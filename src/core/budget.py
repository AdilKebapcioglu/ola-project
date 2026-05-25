"""
budget.py
---------
Budget tracker for shared-budget bidding experiments.

The tracker is a plain stateful object that environments and agents can
consult. It does NOT modify bids itself — agents are responsible for
zeroing bids when budget is exhausted.
"""

from __future__ import annotations


class BudgetTracker:
    """Track cumulative spend against a finite budget.

    Parameters
    ----------
    total_budget : float  — B, the total budget for T rounds
    T            : int    — horizon (used to compute per-round budget rho)
    """

    def __init__(self, total_budget: float, T: int):
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        self.total_budget = total_budget
        self.T = T
        self.rho = total_budget / T  # per-round budget
        self._spent = 0.0
        self._t = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._spent = 0.0
        self._t = 0

    def consume(self, cost: float) -> None:
        """Record spending of `cost` in the current round."""
        if cost < 0:
            raise ValueError(f"Cost must be non-negative, got {cost}")
        self._spent += cost
        self._t += 1

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self._spent)

    @property
    def is_exhausted(self) -> bool:
        return self._spent >= self.total_budget

    @property
    def t(self) -> int:
        return self._t

    def clamp_bids(self, bids, costs_per_unit=None):
        """Zero out all bids if budget is exhausted (hard coercion).

        If costs_per_unit is provided (per-campaign expected cost), scale
        down bids proportionally so total expected cost ≤ remaining budget.
        Simple version: just zero everything out when exhausted.
        """
        import numpy as np
        bids = np.asarray(bids, dtype=float).copy()
        if self.is_exhausted:
            bids[:] = 0.0
        return bids

    def __repr__(self) -> str:
        return (
            f"BudgetTracker(total={self.total_budget:.2f}, "
            f"spent={self._spent:.2f}, remaining={self.remaining:.2f}, "
            f"rho={self.rho:.4f})"
        )
