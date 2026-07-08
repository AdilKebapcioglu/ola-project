"""
ucb_bidder.py
-------------
UCB-based bidding agents for Phase 1 (single campaign, stochastic).

Algorithm A — UCB1BidderAgent:
    Treats each discrete bid as an arm; ignores budget.
    Reward = (v - b) * 1{b >= m_t} in [0, v].

Algorithm B — UCBLikeBidderAgent:
    UCB on utility + LCB on cost; solves an LP each round to pick a mixed
    strategy subject to the per-round budget constraint rho = B_total / T.
    Bids zero when budget is exhausted.
    Reference: notebook 07 UCBLikeAgent.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


class UCB1BidderAgent:
    """Phase 1A: UCB1 over a discrete bid grid, no budget constraint.

    Parameters
    ----------
    bid_grid : array-like
        Discrete bid levels K; bid_grid[0] should be 0 (opt-out).
    value : float
        Private value v. Used as the reward-range parameter in UCB widths.
    T : int
        Horizon; used in UCB confidence width sqrt(2 log T / N_t(b)).
    """

    def __init__(self, bid_grid, value: float, T: int) -> None:
        self.bid_grid = np.asarray(bid_grid, dtype=float)
        self.K = len(self.bid_grid)
        self.value = float(value)
        self.T = T
        self._arm: int = 0
        self.avg_reward: NDArray[np.float64] = np.zeros(self.K)
        self.n_pulls: NDArray[np.float64] = np.zeros(self.K)
        self._t: int = 0

    def reset(self) -> None:
        self.avg_reward = np.zeros(self.K)
        self.n_pulls = np.zeros(self.K)
        self._arm = 0
        self._t = 0

    def select_action(self) -> NDArray[np.float64]:
        if self._t < self.K:
            self._arm = self._t
        else:
            ucbs = self.avg_reward + self.value * np.sqrt(
                2.0 * np.log(self.T) / self.n_pulls
            )
            self._arm = int(np.argmax(ucbs))
        return np.array([self.bid_grid[self._arm]])

    def update(self, feedback: dict) -> None:
        r = float(feedback["reward"])
        self.n_pulls[self._arm] += 1
        self.avg_reward[self._arm] += (
            r - self.avg_reward[self._arm]
        ) / self.n_pulls[self._arm]
        self._t += 1


class UCBLikeBidderAgent:
    """Phase 1B: UCB on utility + LCB on cost, LP mixed strategy, budget-aware.

    Each round computes, for arms tried at least once:
        f_UCB(b) = avg_f(b) + max(v-b, 0) * sqrt(2 log T / N(b))
        c_LCB(b) = max(avg_c(b) - b * sqrt(2 log T / N(b)), 0)
    Untried arms are seeded optimistically instead of forced round-robin:
    f_UCB(b) = max(v-b, 0) (the max possible profit for that bid) and
    c_LCB(b) = 0.5 * b (assume a neutral 50% win rate), so the LP is
    naturally drawn to explore them without a separate warm-up phase, and
    without assuming they are free (which caused the budget to be spent
    almost entirely in the first few percent of the horizon, since every
    untried bid looked costless until it happened to be pulled).

    The confidence width is scaled by max(v-b, 0) — the true best-case
    profit for that specific bid — not by the flat value v. Scaling by v
    everywhere meant a single lucky win on an expensive, barely-tried bid
    produced a wildly inflated f_UCB (since even one pull gives a large
    bonus, and that bonus was multiplied by the full v regardless of how
    little that bid could ever actually pay out), causing the LP to dump
    most of the budget into that one arm for many consecutive rounds.

    Then solves, restricted to arms whose bid does not exceed the currently
    remaining budget (a hard cap, independent of the LP's own soft
    expected-cost constraint below):
        max_{gamma in Delta(B)} sum_b gamma(b) * f_UCB(b)
        s.t. sum_b gamma(b) * c_LCB(b) <= rho_t

    and samples a bid from gamma. Stops bidding (bid = 0) when budget
    exhausted.

    The hard per-arm affordability cap is what guarantees the budget can
    never be violated: the LP's own constraint only bounds *expected* cost
    via the (optimistic, underestimating) c_LCB, so without the cap a single
    unlucky draw of an expensive-but-undersampled arm could win and cost
    more than what's left. Restricting the arm set to bid <= budget_remaining
    makes that impossible regardless of estimation error.

    A second guard, the pacing ceiling, forces bid = 0 whenever cumulative
    spend has gotten more than one max-bid ahead of the smooth target line
    rho * t. This exists because bids are lumpy (cost is 0 or the full bid,
    never a smooth trickle): a single early win on an expensive bid can put
    the agent far ahead of schedule, after which rho_t collapses and starves
    it for most of the remaining horizon. The ceiling spreads spend evenly
    over time instead of letting it arrive in one early burst.

    Parameters
    ----------
    bid_grid : array-like
        Discrete bid levels; bid_grid[0] must be 0 (opt-out arm).
    value : float
        Private value v.
    budget : float
        Total budget B_total.
    T : int
        Horizon.
    rng : np.random.Generator, optional
        RNG for sampling from the LP-derived mixed strategy.
    """

    def __init__(
        self,
        bid_grid,
        value: float,
        budget: float,
        T: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.bid_grid = np.asarray(bid_grid, dtype=float)
        self.K = len(self.bid_grid)
        self.value = float(value)
        self.T = T
        self._budget_total = float(budget)
        self.rho = budget / T
        self._budget: float = self._budget_total
        self._arm: int = 0
        self.avg_f: NDArray[np.float64] = np.zeros(self.K)
        self.avg_c: NDArray[np.float64] = np.zeros(self.K)
        self.n_pulls: NDArray[np.float64] = np.zeros(self.K)
        self._t: int = 0
        self._rng = rng or np.random.default_rng()

    def reset(self) -> None:
        self.avg_f = np.zeros(self.K)
        self.avg_c = np.zeros(self.K)
        self.n_pulls = np.zeros(self.K)
        self._arm = 0
        self._budget = self._budget_total
        self._t = 0

    def select_action(self) -> NDArray[np.float64]:
        if self._budget <= 0.0:
            self._arm = 0
            return np.array([self.bid_grid[0]])

        rho_t = self._budget / max(self.T - self._t, 1)

        # Pacing ceiling: don't let realised spend get too far ahead of the
        # smooth target line rho * t. Without this, a lucky early win on an
        # expensive bid (a single lump of cost, not a smooth trickle) can put
        # the agent way ahead of schedule; the recalculated rho_t then starves
        # it for most of the remaining horizon to compensate, which is what
        # produced front-loaded spending and a long stretch of negative
        # regret. Capping how far ahead it's allowed to get keeps spend paced
        # evenly instead of bursty. Slack of one max-bid's worth avoids
        # penalising the first lucky win itself, only repeats of it.
        spent_so_far = self._budget_total - self._budget
        pace_line = self.rho * self._t
        ahead_of_pace = spent_so_far - pace_line
        slack = self.bid_grid.max()

        # Hard cap: only arms whose bid cannot exceed the remaining budget are
        # eligible, regardless of what the LP's soft cost constraint allows.
        # bid_grid[0] == 0 is always in here since self._budget > 0 here.
        if ahead_of_pace > slack:
            idx = np.array([0])
        else:
            idx = np.flatnonzero(self.bid_grid <= self._budget)

        log_t = np.log(self.T)
        n_pulls = self.n_pulls[idx]
        tried = n_pulls > 0
        width = np.zeros(len(idx))
        width[tried] = np.sqrt(2.0 * log_t / n_pulls[tried])

        # Reward for bid b lies in [0, max(v-b, 0)] (profit if won, 0 if lost) —
        # scale the confidence bonus by that per-bid range, not the flat value
        # v, so one lucky early win on an expensive bid can't look better than
        # that bid could ever actually pay off.
        reward_range = np.maximum(self.value - self.bid_grid[idx], 0.0)
        f_ucb = np.where(tried, self.avg_f[idx] + reward_range * width, reward_range)
        c_lcb = np.where(
            tried,
            np.maximum(self.avg_c[idx] - self.bid_grid[idx] * width, 0.0),
            0.5 * self.bid_grid[idx],
        )

        gamma = np.zeros(self.K)
        gamma[idx] = self._solve_lp(f_ucb, c_lcb, rho_t)
        self._arm = int(self._rng.choice(self.K, p=gamma))
        return np.array([self.bid_grid[self._arm]])

    def _solve_lp(
        self,
        f_ucb: NDArray[np.float64],
        c_lcb: NDArray[np.float64],
        rho: float,
    ) -> NDArray[np.float64]:
        n = len(f_ucb)
        res = optimize.linprog(
            -f_ucb,
            A_ub=[c_lcb],
            b_ub=[rho],
            A_eq=[np.ones(n)],
            b_eq=[1.0],
            bounds=(0.0, 1.0),
            method="highs",
        )
        if not res.success:
            gamma = np.zeros(n)
            gamma[int(np.argmax(f_ucb))] = 1.0
            return gamma
        gamma = np.maximum(res.x, 0.0)
        total = gamma.sum()
        if total < 1e-12:
            gamma = np.zeros(n)
            gamma[0] = 1.0
            return gamma
        return gamma / total

    def update(self, feedback: dict) -> None:
        f_t = float(feedback["reward"])
        c_t = float(feedback["cost"])
        self.n_pulls[self._arm] += 1
        self.avg_f[self._arm] += (f_t - self.avg_f[self._arm]) / self.n_pulls[self._arm]
        self.avg_c[self._arm] += (c_t - self.avg_c[self._arm]) / self.n_pulls[self._arm]
        self._budget -= c_t
        self._t += 1
