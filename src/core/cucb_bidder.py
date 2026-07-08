"""
cucb_bidder.py
--------------
Combinatorial UCB-like bidding agent for Phase 2 (multi-campaign, stochastic).

Extends the Phase 1B UCB-like approach (UCBLikeBidderAgent) to N campaigns
with a conflict-graph feasibility constraint and a shared budget.

Core idea (notebook 09 pattern + Phase 1B LP extension):
  - Maintain (N, K) tables of UCB estimates for utility and LCB for cost.
  - Each round, enumerate feasible independent sets; for each IS solve a joint
    LP over per-campaign mixed strategies subject to the per-round budget.
  - Pick the IS with the highest LP objective; sample one bid per campaign.
  - Semi-bandit feedback: update only the (campaign, bid) pairs actually played.

Initialization:
  By default (force_init=True), systematically explore each (campaign, bid)
  pair once before UCB kicks in. This takes N*K rounds and ensures every arm
  has at least one observation. With force_init=False, skip this and rely on
  the optimistic seeding already built into the UCB estimates instead (Phase
  1B style).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from src.core.conflict_graph import ConflictGraph


class CombUCBLikeBidderAgent:
    """Phase 2: Combinatorial UCB-like bidder with budget and IS constraint.

    Each round's LP is bounded on both sides of the pace line rho * t: an
    affordability cap prevents cumulative spend from ever exceeding the
    budget (see `_solve_lp_for_set`), and a pacing floor requires this
    round's expected cost to be at least rho whenever cumulative spend has
    fallen behind schedule (see `select_action`). Without the floor, the
    LP can perpetually prefer cheap-looking options and never catch up,
    leaving most of the budget unused even though nothing prevents it from
    being spent.

    Parameters
    ----------
    bid_grid : (K,) array-like
        Shared discrete bid levels; bid_grid[0] must be 0 (opt-out arm).
    values : (N,) array-like
        Per-campaign private values v_i. Used as reward-range in UCB widths.
    budget : float
        Total budget B_total.
    T : int
        Horizon.
    conflict_graph : ConflictGraph
        Encodes which campaigns conflict. Feasible actions are independent sets.
    rng : np.random.Generator, optional
        RNG for sampling from LP-derived mixed strategies.
    force_init : bool, default True
        If True, spend the first N*K rounds exploring every (campaign, bid)
        pair once in a fixed round-robin order before the UCB+LP phase
        starts. If False, skip this forced warm-up and rely purely on the
        optimistic seeding already built into the UCB estimates (Phase 1B
        style — see `UCBLikeBidderAgent`): unpulled arms get f_UCB set to
        their maximum possible reward, so the LP is naturally drawn to
        explore them without a separate phase.
    """

    def __init__(
        self,
        bid_grid,
        values,
        budget: float,
        T: int,
        conflict_graph: ConflictGraph,
        rng: np.random.Generator | None = None,
        force_init: bool = True,
    ) -> None:
        self.bid_grid = np.asarray(bid_grid, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.K = len(self.bid_grid)
        self.N = len(self.values)
        self.T = T
        self._budget_total = float(budget)
        self.rho = budget / T
        self._rng = rng or np.random.default_rng()

        # All feasible independent sets, precomputed once at init.
        # Sorted largest-first by ConflictGraph.all_independent_sets().
        self.feasible_sets: list[frozenset] = conflict_graph.all_independent_sets()

        # Per-(campaign, bid) statistics
        self.avg_f = np.zeros((self.N, self.K))   # average utility
        self.avg_c = np.zeros((self.N, self.K))   # average cost
        self.n_pulls = np.zeros((self.N, self.K))

        self._budget = self._budget_total
        self._t = 0

        # Initialization: explore each (i, k) pair once before UCB
        self._force_init = force_init
        self._n_init = self.N * self.K if force_init else 0
        self._init_idx = 0

        # Current round's selection, stored for update()
        self._selected_arms: dict[int, int] = {}  # campaign i -> arm index k

    # ------------------------------------------------------------------
    # Runner protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.avg_f = np.zeros((self.N, self.K))
        self.avg_c = np.zeros((self.N, self.K))
        self.n_pulls = np.zeros((self.N, self.K))
        self._budget = self._budget_total
        self._t = 0
        self._init_idx = 0
        self._selected_arms = {}

    def select_action(self) -> NDArray[np.float64]:
        """Return (N,) bid vector; 0 for campaigns not selected this round."""
        bids = np.zeros(self.N)

        if self._budget <= 0.0:
            self._selected_arms = {}
            return bids

        # Initialization: explore arms in a fixed (campaign, bid) order
        if self._init_idx < self._n_init:
            i = self._init_idx // self.K
            k = self._init_idx % self.K
            self._selected_arms = {i: k}
            bids[i] = self.bid_grid[k]
            return bids

        # UCB+LP phase
        # Use min(rho, remaining budget) so we never over-commit if budget is
        # nearly exhausted mid-horizon.
        effective_rho = min(self.rho, self._budget)

        # Pacing floor: if cumulative spend has fallen behind the smooth
        # target line rho * t, require this round's expected cost to be at
        # least rho (not just capped at rho). Without this, the LP can
        # perpetually prefer safe-looking low-cost options and never
        # actually catch up to the budget it's allowed to use — the mirror
        # image of the pacing ceiling in UCBLikeBidderAgent, which instead
        # reins in bursts of overspending.
        spent_so_far = self._budget_total - self._budget
        pace_line = self.rho * self._t
        floor_rho = self.rho if spent_so_far < pace_line else 0.0

        f_ucb, c_lcb = self._compute_ucb_lcb()
        best_val = -np.inf
        best_arms: dict[int, int] = {}

        for S in self.feasible_sets:
            S_list = sorted(S)
            lp_val, gamma_dict = self._solve_lp_for_set(
                S_list, f_ucb, c_lcb, effective_rho, floor_rho
            )
            if lp_val > best_val:
                best_val = lp_val
                best_arms = {
                    i: int(self._rng.choice(self.K, p=gamma_dict[i]))
                    for i in S_list
                }

        if not best_arms and floor_rho > 0:
            # The floor made every feasible set infeasible this round (e.g.
            # affordability cap too tight to guarantee floor_rho) — fall
            # back to the unconstrained-floor LP rather than bidding nothing.
            for S in self.feasible_sets:
                S_list = sorted(S)
                lp_val, gamma_dict = self._solve_lp_for_set(
                    S_list, f_ucb, c_lcb, effective_rho, 0.0
                )
                if lp_val > best_val:
                    best_val = lp_val
                    best_arms = {
                        i: int(self._rng.choice(self.K, p=gamma_dict[i]))
                        for i in S_list
                    }

        self._selected_arms = best_arms
        for i, k in best_arms.items():
            bids[i] = self.bid_grid[k]
        return bids

    def update(self, feedback: dict) -> None:
        """Semi-bandit update: only update arms that were played this round."""
        utilities = np.asarray(feedback["utilities"], dtype=float)
        costs = np.asarray(feedback["costs"], dtype=float)

        for i, k in self._selected_arms.items():
            self.n_pulls[i, k] += 1
            n = self.n_pulls[i, k]
            self.avg_f[i, k] += (utilities[i] - self.avg_f[i, k]) / n
            self.avg_c[i, k] += (costs[i] - self.avg_c[i, k]) / n

        self._budget -= float(feedback["cost"])
        if self._init_idx < self._n_init:
            self._init_idx += 1
        self._t += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ucb_lcb(self) -> tuple[NDArray, NDArray]:
        """Compute f_UCB and c_budget matrices of shape (N, K).

        f_UCB: standard UCB index — optimistic reward estimate.
        c_budget: Laplace-smoothed cost estimate used in the LP budget constraint.
          For unexplored arms (n_pulls == 0): c_budget = 0.5 * bid_grid[k] (assume a
          neutral 50% win rate — not "never wins" and not "always wins", either of
          which biases the LP toward one extreme before any real data exists).
          For explored arms: (n * avg_c + 0.5 * bid) / (n + 1), which converges to
          avg_c while staying above zero early on.
        Using c_budget instead of pure LCB prevents budget over-consumption:
        with the raw LCB the confidence width dominates avg_c for hundreds of rounds,
        yielding c_lcb ≈ 0 and making the LP budget constraint non-binding. The
        earlier "assume win" version (pseudo-cost = bid) over-corrected the other
        way: it made every untried bid look maximally expensive, which biased the
        agent away from ever trying higher bids and left it chronically under-
        spending relative to the budget.
        """
        mask_unexp = self.n_pulls == 0
        safe_n = np.where(mask_unexp, 1.0, self.n_pulls)

        # Tighter Hoeffding range: arm (i,k) reward lies in [0, max(v_i - b_k, 0)].
        # Bids above v_i can only give non-positive utility, so their range is 0 —
        # the LP assigns zero weight to them after a handful of pulls.
        # This prevents high-value campaigns (large v_i) from being persistently
        # over-optimistic relative to lower-value campaigns with better win rates.
        reward_range = np.maximum(
            self.values[:, None] - self.bid_grid[None, :], 0.0
        )  # (N, K)
        # bid=0 never wins (P(m_i ≤ 0) = 0); zero its range so it gets no
        # UCB bonus and the LP treats it as worthless from the start.
        reward_range = np.where(
            self.bid_grid[None, :] == 0.0, 0.0, reward_range
        )
        width = reward_range * np.sqrt(2.0 * np.log(self.T) / safe_n)

        f_ucb = np.where(
            mask_unexp,
            reward_range * np.sqrt(2.0 * np.log(self.T)),
            self.avg_f + width,
        )
        # Laplace-smoothed cost: (n * avg_c + 0.5 * bid) / (n + 1)
        # Equivalent to a Bayesian estimate with one pseudo-observation assuming
        # a neutral 50% win rate (not "always wins", which biased the LP against
        # ever trying higher bids).
        c_budget = (self.n_pulls * self.avg_c + 0.5 * self.bid_grid[None, :]) / (
            self.n_pulls + 1.0
        )
        return f_ucb, c_budget

    def _solve_lp_for_set(
        self,
        S_list: list[int],
        f_ucb: NDArray,
        c_lcb: NDArray,
        budget_rhs: float | None = None,
        floor_rho: float = 0.0,
    ) -> tuple[float, dict[int, NDArray]]:
        """Solve the per-IS LP for a fixed independent set S.

        Variables: gamma_{i,k} for i in S, k in 0..K-1 (flattened).
        Objective: max  sum_{i in S} sum_k gamma_{i,k} * f_UCB[i,k]
        Budget:    s.t. rho_floor <= sum_{i in S} sum_k gamma_{i,k} * c_budget[i,k] <= rho
        Simplex:        sum_k gamma_{i,k} = 1  for each i in S
        Bounds:         0 <= gamma_{i,k} <= 1, further capped per hard
                        affordability constraint below.

        Hard affordability cap: the LP's own budget constraint only bounds
        *expected* cost via c_lcb, which underestimates cost early on — so a
        single round where every campaign in S happens to win could still
        cost more than what's left (e.g. see UCBLikeBidderAgent, Phase 1B).
        With up to |S| campaigns won simultaneously in one round, the
        worst-case cost of this round is the sum of the |S| chosen bids.
        Restricting every campaign's eligible bids to
        <= remaining_budget / |S| guarantees that even if all of them win,
        total cost cannot exceed what remains, regardless of estimation
        error in c_lcb.

        floor_rho: when > 0, additionally requires expected cost >= floor_rho
        (passed by select_action when cumulative spend has fallen behind the
        pace line), so the LP can't perpetually settle for cheap-looking
        options while chronically under-using the budget. Can make some
        candidate sets infeasible; select_action falls back to floor_rho=0
        if every set becomes infeasible.

        Returns
        -------
        lp_value : float          — LP objective (-inf if infeasible)
        gamma_dict : dict[int, NDArray]  — per-campaign bid distributions
        """
        ns = len(S_list)
        n_vars = ns * self.K

        rhs = budget_rhs if budget_rhs is not None else self.rho
        c_obj = -f_ucb[S_list, :].ravel()
        cost_row = c_lcb[S_list, :].ravel()
        if floor_rho > 0:
            A_ub = np.stack([cost_row, -cost_row])
            b_ub = np.array([rhs, -floor_rho])
        else:
            A_ub = cost_row.reshape(1, -1)
            b_ub = np.array([rhs])

        A_eq = np.zeros((ns, n_vars))
        for j in range(ns):
            A_eq[j, j * self.K : (j + 1) * self.K] = 1.0
        b_eq = np.ones(ns)

        max_affordable = self._budget / ns
        eligible = self.bid_grid <= max_affordable  # (K,) bool; bid 0 always eligible
        bounds = [
            (0.0, 1.0) if eligible[idx % self.K] else (0.0, 0.0)
            for idx in range(n_vars)
        ]

        res = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            return -np.inf, {}

        gamma_flat = np.maximum(res.x, 0.0)
        gamma_dict: dict[int, NDArray] = {}
        for j, i in enumerate(S_list):
            gamma_i = gamma_flat[j * self.K : (j + 1) * self.K]
            total = gamma_i.sum()
            gamma_dict[i] = gamma_i / total if total > 1e-12 else np.ones(self.K) / self.K

        return float(-res.fun), gamma_dict
