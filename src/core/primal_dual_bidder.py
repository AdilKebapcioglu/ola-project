"""
primal_dual_bidder.py
---------------------
Primal-dual bidding agent for Phase 3 (best-of-both-worlds, multi-campaign).

Extends the single-knapsack primal-dual scheme (Hedge primal + OGD dual,
notebook 08 pattern) to N campaigns with a conflict graph and a shared
budget, under FULL FEEDBACK: after each round the highest competing bid
m_{i,t} of every campaign is observed, so the counterfactual utility and
cost of every (campaign, bid) pair can be computed — Hedge applies, no
importance weighting or exploration bonus needed.

Per round:
  1. Each campaign i has a Hedge learner over the K bids; its distribution
     x_{i,t} is computed from cumulative Lagrangian losses.
  2. A max-weight independent set is selected, weighting campaign i by its
     expected Lagrangian value mu_i = x_{i,t} . (avg_f_i - lambda_t * avg_c_i)
     under the current dual variable; campaigns with mu_i <= 0 are dropped
     (bidding 0 dominates them). Campaigns outside the set bid 0.
  3. One bid per selected campaign is sampled from x_{i,t}.
  4. After observing m_t: every campaign's Hedge is updated with the
     full-feedback Lagrangian loss, and the dual variable takes an OGD step
     on the budget constraint violation.

The dual variable performs the pacing automatically: overspending raises
lambda, which penalises costly bids in every Hedge; underspending drives
lambda to 0, which lets the agent bid up to the unconstrained optimum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.conflict_graph import ConflictGraph


class PrimalDualBidderAgent:
    """Phase 3: primal-dual bidder (Hedge + OGD) with conflict graph and budget.

    Lagrangian (per campaign i, bid k, round t):
        L_{i,k,t} = f_{i,k,t} - lambda_t * c_{i,k,t}
        f_{i,k,t} = (v_i - b_k) * 1{b_k >= m_{i,t}}
        c_{i,k,t} = b_k * 1{b_k >= m_{i,t}}
    Hedge losses are the negated Lagrangian payoffs, affinely rescaled to
    [0, 1] using the a-priori payoff range (this absorbs a constant factor
    into the learning rate, as in the course formulation).

    Dual update (shared scalar, one budget constraint):
        lambda_{t+1} = clip(lambda_t - eta_dual * (rho - E_{x_t}[cost_t]), 0, 1/rho)
    where the expected cost is taken under the played distributions of the
    selected campaigns (unselected campaigns bid 0 and cost nothing).

    Budget safety: bidding stops entirely (all-zero bids) once the remaining
    budget drops below the worst-case cost of one round, i.e. the largest
    independent set size times the maximum bid — after that point no
    realization can exceed the budget, whatever the estimates say.

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
    rng : np.random.Generator, optional
        RNG for sampling bids from the Hedge distributions.
    hedge_eta : float, optional
        Primal (Hedge) learning rate; defaults to sqrt(log K / T). Any fixed
        multiple of the default keeps the O(sqrt(T)) Hedge guarantee (with a
        proportionally larger constant); empirically larger values sharpen
        the bid distributions substantially on this problem.
    ogd_eta : float, optional
        Dual (OGD) step size; defaults to 1 / sqrt(T).
    lmbd_init : float, default 1.0
        Initial dual variable. The course formulation starts at 1
        (maximally cautious); starting at 0 avoids an early under-bidding
        phase whose spend deficit fixed-rho pacing can never recover.
    adaptive_rho : bool, default False
        If True, the dual gradient targets rho_t = remaining_budget /
        remaining_rounds instead of the fixed rho, so a spend deficit
        raises the target and pushes lambda down (catch-up pacing).
        The lambda projection ceiling stays at 1/rho.
    """

    def __init__(
        self,
        bid_grid,
        values,
        budget: float,
        T: int,
        conflict_graph: ConflictGraph,
        rng: np.random.Generator | None = None,
        hedge_eta: float | None = None,
        ogd_eta: float | None = None,
        lmbd_init: float = 1.0,
        adaptive_rho: bool = False,
    ) -> None:
        self.bid_grid = np.asarray(bid_grid, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.K = len(self.bid_grid)
        self.N = len(self.values)
        self.T = T
        self._budget_total = float(budget)
        self.rho = budget / T
        self.lmbd_max = 1.0 / self.rho
        self._rng = rng or np.random.default_rng()
        self._cg = conflict_graph

        self.hedge_eta = hedge_eta if hedge_eta is not None else np.sqrt(np.log(self.K) / T)
        self.ogd_eta = ogd_eta if ogd_eta is not None else 1.0 / np.sqrt(T)
        self._lmbd_init = float(lmbd_init)
        self._adaptive_rho = bool(adaptive_rho)

        # Worst-case cost of a single round: every campaign in the largest
        # feasible independent set wins at the maximum bid.
        max_is_size = max(len(S) for S in conflict_graph.all_independent_sets())
        self._max_round_cost = max_is_size * float(self.bid_grid.max())

        # A-priori Lagrangian payoff range, used to rescale Hedge losses to
        # [0, 1]. Utility of a won auction lies in [min_i v_i - max_b, max_i v_i]
        # (negative when bidding above value), cost in [0, max_b], lambda in
        # [0, 1/rho].
        max_bid = float(self.bid_grid.max())
        f_min = min(0.0, float(self.values.min()) - max_bid)
        self._L_min = f_min - self.lmbd_max * max_bid
        self._L_max = float(self.values.max())

        # Per-(campaign, bid) cumulative statistics — exact under full feedback
        self.cum_loss = np.zeros((self.N, self.K))  # scaled Hedge losses
        self.cum_f = np.zeros((self.N, self.K))     # raw utilities
        self.cum_c = np.zeros((self.N, self.K))     # raw costs

        self.lmbd = self._lmbd_init
        self._budget = self._budget_total
        self._t = 0
        self._x_t = np.full((self.N, self.K), 1.0 / self.K)
        self._selected: frozenset = frozenset()

    # ------------------------------------------------------------------
    # Runner protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.cum_loss = np.zeros((self.N, self.K))
        self.cum_f = np.zeros((self.N, self.K))
        self.cum_c = np.zeros((self.N, self.K))
        self.lmbd = self._lmbd_init
        self._budget = self._budget_total
        self._t = 0
        self._x_t = np.full((self.N, self.K), 1.0 / self.K)
        self._selected = frozenset()

    def select_action(self) -> NDArray[np.float64]:
        """Return (N,) bid vector; 0 for campaigns not selected this round."""
        bids = np.zeros(self.N)

        if self._budget < self._max_round_cost:
            self._selected = frozenset()
            return bids

        # Hedge distributions from cumulative losses (numerically stable
        # softmax; weights w = exp(-eta * cum_loss) normalised per campaign).
        z = -self.hedge_eta * self.cum_loss
        z -= z.max(axis=1, keepdims=True)
        w = np.exp(z)
        self._x_t = w / w.sum(axis=1, keepdims=True)

        # Weighted MIS: campaign i's weight is its expected Lagrangian value
        # under the current dual variable and its Hedge distribution.
        if self._t == 0:
            mu = self.values.copy()  # no data yet — prefer high-value campaigns
        else:
            avg_L = (self.cum_f - self.lmbd * self.cum_c) / self._t  # (N, K)
            mu = np.einsum("nk,nk->n", self._x_t, avg_L)

        positive = mu > 0.0
        if not positive.any():
            # Every campaign has non-positive expected Lagrangian value:
            # bidding 0 everywhere dominates.
            self._selected = frozenset()
            return bids

        S, _ = self._cg.max_weight_independent_set(np.where(positive, mu, 0.0))
        # Drop non-positive members: any subset of an IS is an IS, and a
        # campaign with mu_i <= 0 only lowers the set's value.
        S = frozenset(i for i in S if positive[i])
        self._selected = S

        for i in S:
            k = int(self._rng.choice(self.K, p=self._x_t[i]))
            bids[i] = self.bid_grid[k]
        return bids

    def update(self, feedback: dict) -> None:
        """Full-feedback update: every (campaign, bid) pair, every round."""
        m = np.asarray(feedback["competing_bids"], dtype=float)  # (N,)

        # Counterfactual outcome of every (campaign, bid) pair this round
        win = self.bid_grid[None, :] >= m[:, None]                     # (N, K)
        f_mat = (self.values[:, None] - self.bid_grid[None, :]) * win  # (N, K)
        c_mat = self.bid_grid[None, :] * win                           # (N, K)

        # Primal: Hedge loss = 1 - rescaled Lagrangian payoff, in [0, 1]
        L = f_mat - self.lmbd * c_mat
        loss = 1.0 - (L - self._L_min) / (self._L_max - self._L_min)
        self.cum_loss += loss
        self.cum_f += f_mat
        self.cum_c += c_mat

        # Dual: OGD on the budget constraint, expected cost under the played
        # distributions (campaigns outside the selected set bid 0). With
        # adaptive_rho the target is the remaining per-round allowance, so a
        # spend deficit raises it and lambda is pushed down to catch up.
        exp_cost = sum(float(self._x_t[i] @ c_mat[i]) for i in self._selected)
        rho_target = (
            self._budget / max(self.T - self._t, 1) if self._adaptive_rho else self.rho
        )
        self.lmbd = float(
            np.clip(self.lmbd - self.ogd_eta * (rho_target - exp_cost), 0.0, self.lmbd_max)
        )

        self._budget -= float(feedback["cost"])
        self._t += 1
