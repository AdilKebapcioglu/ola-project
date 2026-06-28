"""
environment.py
--------------
BiddingEnvironment base class and first-price single-slot auction logic.

All concrete environments (stochastic, adversarial, non-stationary) extend
BiddingEnvironment and implement _sample_competing_bids().
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any


# ---------------------------------------------------------------------------
# Auction primitives (stateless, pure functions)
# ---------------------------------------------------------------------------

def first_price_outcome(
    bid: float,
    competing_bid: float,
    value: float,
) -> tuple[bool, float, float]:
    """Resolve a first-price single-slot auction for one campaign.

    The advertiser wins iff bid >= competing_bid (highest others' bid).

    Parameters
    ----------
    bid : float          — our submitted bid
    competing_bid : float — maximum competing bid m_t
    value : float        — our private value v for winning

    Returns
    -------
    won   : bool
    utility : float   — (v - bid) if won, else 0
    cost    : float   — bid if won, else 0
    """
    won = bid >= competing_bid
    if won:
        return True, float(value - bid), float(bid)
    return False, 0.0, 0.0


def multi_campaign_outcomes(
    bids: NDArray[np.float64],
    competing_bids: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised first-price outcomes for N campaigns.

    Parameters
    ----------
    bids, competing_bids, values : (N,) arrays

    Returns
    -------
    won       : (N,) bool array
    utilities : (N,) float array
    costs     : (N,) float array
    """
    won = bids >= competing_bids
    utilities = np.where(won, values - bids, 0.0)
    costs = np.where(won, bids, 0.0)
    return won, utilities, costs


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class BiddingEnvironment(ABC):
    """Abstract base class for all bidding environments.

    Subclasses implement `_sample_competing_bids()` to draw m_t each round.

    Parameters
    ----------
    values : (N,) array  — per-campaign private values (fixed)
    T      : int         — horizon (informational; env does not enforce it)
    rng    : np.random.Generator
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        T: int,
        rng: np.random.Generator,
    ):
        self.values = np.asarray(values, dtype=float)
        self.N = len(self.values)
        self.T = T
        self.rng = rng
        self._t = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset round counter (RNG state is NOT reset for independence)."""
        self._t = 0

    def step(
        self,
        bids: NDArray[np.float64],
    ) -> tuple[float, float, dict[str, Any]]:
        """Execute one round.

        Parameters
        ----------
        bids : (N,) array — submitted bids for each campaign

        Returns
        -------
        total_utility : float
        total_cost    : float
        info : dict with keys
            't', 'competing_bids', 'won', 'utilities', 'costs'
        """
        bids = np.asarray(bids, dtype=float)
        assert bids.shape == (self.N,), f"Expected {self.N} bids, got {bids.shape}"

        m = self._sample_competing_bids()
        won, utilities, costs = multi_campaign_outcomes(bids, m, self.values)

        self._t += 1
        info = {
            "t": self._t,
            "competing_bids": m,
            "won": won,
            "utilities": utilities,
            "costs": costs,
        }
        return float(utilities.sum()), float(costs.sum()), info

    def round(self, bids: NDArray[np.float64]) -> dict[str, Any]:
        """Runner-protocol wrapper around step().

        Parameters
        ----------
        bids : (N,) array — submitted bids for each campaign

        Returns
        -------
        dict with keys: "reward", "cost", "competing_bids", "won",
                        "utilities", "costs", "t"
        """
        total_utility, total_cost, info = self.step(bids)
        return {
            "reward": total_utility,
            "cost": total_cost,
            **info,
        }

    # ------------------------------------------------------------------
    # To implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _sample_competing_bids(self) -> NDArray[np.float64]:
        """Draw the competing bid vector m_t for the current round.

        Returns
        -------
        m : (N,) array
        """
