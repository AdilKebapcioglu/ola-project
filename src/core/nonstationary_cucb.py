"""
nonstationary_cucb.py
---------------------
Non-stationary variants of the Phase 2 combinatorial UCB-like bidder, for
Phase 4 (slightly non-stationary, piecewise-stationary environments).

Both agents subclass `CombUCBLikeBidderAgent` and change ONLY how the
per-(campaign, bid) statistics avg_f / avg_c / n_pulls are maintained; the
per-IS LPs, budget safety (hard affordability cap), and pacing floor are
inherited untouched.

- `SWCombUCBLikeBidderAgent` (CUCB-SW): statistics computed over a sliding
  window of the last W rounds (course notebook 10, SW-UCB pattern). Old
  observations are forgotten; arms with no pulls left in the window fall
  back to the base class's optimistic seeding, so forgetting automatically
  re-triggers exploration.

- `CDCombUCBLikeBidderAgent` (CUCB-CD): all-time statistics plus a CUSUM
  change detector per (campaign, bid) on the win indicator (notebook 10,
  CUSUM-UCB pattern). On detection the WHOLE campaign row is reset: all K
  bids of campaign i face the same competing bid m_i, so a change detected
  at any (i, k) means every (i, .) estimate is stale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.conflict_graph import ConflictGraph
from src.core.cucb_bidder import CombUCBLikeBidderAgent


class SWCombUCBLikeBidderAgent(CombUCBLikeBidderAgent):
    """Phase 4: combinatorial UCB-like bidder with a sliding window (CUCB-SW).

    Windowed statistics are kept in circular buffers with rolling sums:
    each round the observations that fall out of the window are subtracted
    and the new ones added (O(N*K) per round) — equivalent to notebook 10's
    NaN-cache but without rescanning the window.

    The UCB confidence width uses log(W) instead of log(T) (notebook 10's
    correction of the slides): W is the effective sample horizon of every
    estimate.

    Parameters
    ----------
    bid_grid : (K,) array-like
        Shared discrete bid levels; bid_grid[0] must be 0 (opt-out arm).
    values : (N,) array-like
        Per-campaign private values v_i.
    budget : float
        Total budget B_total.
    T : int
        Horizon.
    conflict_graph : ConflictGraph
        Feasible actions are independent sets of this graph.
    W : int
        Window length in rounds. Theory (notebook 10, abrupt changes):
        W = floor(2 * sqrt(T log T / Upsilon_T)) with Upsilon_T the number
        of change points.
    rng : np.random.Generator, optional
        RNG for sampling from LP-derived mixed strategies.
    force_init : bool, default False
        Forced round-robin warm-up is off by default: it costs N*K rounds
        whose knowledge is discarded by the window anyway; the optimistic
        seeding of unexplored arms does the warm-up instead.
    """

    def __init__(
        self,
        bid_grid,
        values,
        budget: float,
        T: int,
        conflict_graph: ConflictGraph,
        W: int,
        rng: np.random.Generator | None = None,
        force_init: bool = False,
    ) -> None:
        super().__init__(
            bid_grid=bid_grid,
            values=values,
            budget=budget,
            T=T,
            conflict_graph=conflict_graph,
            rng=rng,
            force_init=force_init,
        )
        if W < 2:
            raise ValueError(f"W must be >= 2, got {W}")
        self.W = int(W)
        self._log_horizon = np.log(self.W)

        # Circular buffers over the last W rounds and their rolling sums
        self._buf_f = np.zeros((self.W, self.N, self.K))
        self._buf_c = np.zeros((self.W, self.N, self.K))
        self._buf_pull = np.zeros((self.W, self.N, self.K))
        self._buf_pos = 0
        self._sum_f = np.zeros((self.N, self.K))
        self._sum_c = np.zeros((self.N, self.K))
        self._cnt = np.zeros((self.N, self.K))

    def reset(self) -> None:
        super().reset()
        self._buf_f[:] = 0.0
        self._buf_c[:] = 0.0
        self._buf_pull[:] = 0.0
        self._buf_pos = 0
        self._sum_f[:] = 0.0
        self._sum_c[:] = 0.0
        self._cnt[:] = 0.0

    def update(self, feedback: dict) -> None:
        played = dict(self._selected_arms)
        # Base update handles budget accounting, init bookkeeping and _t; the
        # all-time incremental means it writes into avg_f/avg_c/n_pulls are
        # overwritten below with the windowed statistics.
        super().update(feedback)

        utilities = np.asarray(feedback["utilities"], dtype=float)
        costs = np.asarray(feedback["costs"], dtype=float)

        # Evict the observations of the round that falls out of the window
        pos = self._buf_pos
        self._sum_f -= self._buf_f[pos]
        self._sum_c -= self._buf_c[pos]
        self._cnt -= self._buf_pull[pos]
        self._buf_f[pos] = 0.0
        self._buf_c[pos] = 0.0
        self._buf_pull[pos] = 0.0

        # Insert this round's semi-bandit observations
        for i, k in played.items():
            self._buf_f[pos, i, k] = utilities[i]
            self._buf_c[pos, i, k] = costs[i]
            self._buf_pull[pos, i, k] = 1.0
            self._sum_f[i, k] += utilities[i]
            self._sum_c[i, k] += costs[i]
            self._cnt[i, k] += 1.0
        self._buf_pos = (pos + 1) % self.W

        # Windowed statistics replace the all-time ones. Arms with zero pulls
        # inside the window get n_pulls = 0 and are re-seeded optimistically
        # by the base class — forgetting re-triggers exploration for free.
        safe_cnt = np.where(self._cnt > 0, self._cnt, 1.0)
        self.n_pulls = self._cnt.copy()
        self.avg_f = np.where(self._cnt > 0, self._sum_f / safe_cnt, 0.0)
        self.avg_c = np.where(self._cnt > 0, self._sum_c / safe_cnt, 0.0)


class CDCombUCBLikeBidderAgent(CombUCBLikeBidderAgent):
    """Phase 4: combinatorial UCB-like bidder with CUSUM change detection (CUCB-CD).

    Detection signal: the win indicator w = 1{b_k >= m_i} of each PLAYED
    (campaign, bid) pair. Utility and cost of a played arm are deterministic
    functions of w, so w is the sufficient statistic, and a change in
    campaign i's competing-bid distribution is exactly a change in the win
    rates of its arms. Per-arm Bernoulli CUSUM (notebook 10 pattern):

        first M post-reset samples of arm (i, k)  ->  baseline u_0
        then   g+ = max(0, g+ + (w - u_0)),  g- = max(0, g- + (u_0 - w))
        change detected when max(g+, g-) >= h

    Reset scope — whole campaign row: all K bids of campaign i share the
    same m_i, so a change detected at any (i, k) invalidates every (i, .)
    estimate; row i of avg_f / avg_c / n_pulls and all row-i detectors are
    zeroed (the base class's optimistic seeding then re-explores the row).
    This deviates from notebook 10's per-arm reset and exploits the shared
    structure — more sample-efficient than resetting one arm at a time.

    There are no forced post-reset warm-up counters (the notebook's M forced
    pulls per arm): the LP plus optimistic seeding re-explore naturally, and
    each arm's detector simply stays inactive until it has M post-reset
    samples. Instead, alpha-exploration keeps the detectors fed: with
    probability `alpha` per round, one randomly chosen campaign of the
    selected independent set plays a uniformly random AFFORDABLE bid instead
    of its LP-sampled one (same per-campaign affordability cap as the LP, so
    the budget guarantee is untouched).

    Parameters
    ----------
    bid_grid : (K,) array-like
        Shared discrete bid levels; bid_grid[0] must be 0 (opt-out arm).
    values : (N,) array-like
        Per-campaign private values v_i.
    budget : float
        Total budget B_total.
    T : int
        Horizon.
    conflict_graph : ConflictGraph
        Feasible actions are independent sets of this graph.
    M : int
        Detector warm-up: samples per arm used to form the CUSUM baseline
        u_0. Notebook 10 default: int(log(T / Upsilon_T)).
    h : float
        CUSUM detection threshold. Notebook 10 default: 2 * log(T / Upsilon_T).
    alpha : float
        Probability of the extra uniform exploration round. Notebook 10
        default: sqrt(Upsilon_T * log(T / Upsilon_T) / T).
    rng : np.random.Generator, optional
        RNG for LP sampling and alpha-exploration.
    force_init : bool, default False
        As in `SWCombUCBLikeBidderAgent`.

    Attributes
    ----------
    n_resets : (N,) array
        Number of detector-triggered resets per campaign.
    reset_history : list[list[int]]
        Rounds at which each campaign was reset (for detection-delay plots).
    """

    def __init__(
        self,
        bid_grid,
        values,
        budget: float,
        T: int,
        conflict_graph: ConflictGraph,
        M: int,
        h: float,
        alpha: float,
        rng: np.random.Generator | None = None,
        force_init: bool = False,
    ) -> None:
        super().__init__(
            bid_grid=bid_grid,
            values=values,
            budget=budget,
            T=T,
            conflict_graph=conflict_graph,
            rng=rng,
            force_init=force_init,
        )
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        self.M = int(M)
        self.h = float(h)
        self.alpha = float(alpha)

        # Per-arm CUSUM state
        self._cd_n = np.zeros((self.N, self.K))      # post-reset sample count
        self._cd_sum0 = np.zeros((self.N, self.K))   # sum of first M samples
        self._cd_u0 = np.zeros((self.N, self.K))     # baseline mean
        self._gp = np.zeros((self.N, self.K))
        self._gm = np.zeros((self.N, self.K))

        self.n_resets = np.zeros(self.N)
        self.reset_history: list[list[int]] = [[] for _ in range(self.N)]

    def reset(self) -> None:
        super().reset()
        self._cd_n = np.zeros((self.N, self.K))
        self._cd_sum0 = np.zeros((self.N, self.K))
        self._cd_u0 = np.zeros((self.N, self.K))
        self._gp = np.zeros((self.N, self.K))
        self._gm = np.zeros((self.N, self.K))
        self.n_resets = np.zeros(self.N)
        self.reset_history = [[] for _ in range(self.N)]

    def select_action(self) -> NDArray[np.float64]:
        bids = super().select_action()

        # Alpha-exploration: replace one campaign's LP-sampled bid with a
        # uniformly random affordable one, keeping the detectors fed on arms
        # the LP has stopped playing. Uses the same affordability cap as the
        # LP (bid <= remaining / |S|), so the worst-case-cost guarantee holds.
        if (
            self._selected_arms
            and self._init_idx >= self._n_init
            and self._rng.random() < self.alpha
        ):
            i = int(self._rng.choice(sorted(self._selected_arms)))
            max_affordable = self._budget / len(self._selected_arms)
            eligible = np.flatnonzero(self.bid_grid <= max_affordable)
            k = int(self._rng.choice(eligible))
            self._selected_arms[i] = k
            bids[i] = self.bid_grid[k]
        return bids

    def update(self, feedback: dict) -> None:
        played = dict(self._selected_arms)
        t_now = self._t  # round index of this feedback (base update increments it)
        super().update(feedback)

        won = np.asarray(feedback["won"], dtype=float)

        to_reset: set[int] = set()
        for i, k in played.items():
            w = won[i]
            if self._cd_n[i, k] < self.M:
                # Warm-up: accumulate the baseline
                self._cd_sum0[i, k] += w
                self._cd_n[i, k] += 1
                if self._cd_n[i, k] == self.M:
                    self._cd_u0[i, k] = self._cd_sum0[i, k] / self.M
            else:
                # Active detection
                u0 = self._cd_u0[i, k]
                self._gp[i, k] = max(0.0, self._gp[i, k] + (w - u0))
                self._gm[i, k] = max(0.0, self._gm[i, k] + (u0 - w))
                self._cd_n[i, k] += 1
                if max(self._gp[i, k], self._gm[i, k]) >= self.h:
                    to_reset.add(i)

        for i in to_reset:
            self._reset_campaign(i, t_now)

    def _reset_campaign(self, i: int, t: int) -> None:
        """Forget everything about campaign i (statistics and detectors)."""
        self.avg_f[i] = 0.0
        self.avg_c[i] = 0.0
        self.n_pulls[i] = 0.0
        self._cd_n[i] = 0.0
        self._cd_sum0[i] = 0.0
        self._cd_u0[i] = 0.0
        self._gp[i] = 0.0
        self._gm[i] = 0.0
        self.n_resets[i] += 1
        self.reset_history[i].append(t)
