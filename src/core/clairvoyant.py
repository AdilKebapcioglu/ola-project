"""
clairvoyant.py
--------------
Clairvoyant benchmark computations for Phase 1 (single campaign, stochastic).

No-budget clairvoyant:   best fixed bid b* in expectation.
Budget clairvoyant:      LP over mixed bid strategies (notebook 07 style).
Win probability helpers: analytical formulas for common distributional assumptions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


def win_probs_beta_uniform(bid_grid: NDArray, n_competitors: int) -> NDArray:
    """P(m <= b) when m = max of n_competitors iid Uniform(0, 1) bids.

    m ~ Beta(n_competitors, 1), so CDF(b) = b^n_competitors.

    Parameters
    ----------
    bid_grid : (K,) array
    n_competitors : int

    Returns
    -------
    (K,) array of win probabilities
    """
    return np.asarray(bid_grid, dtype=float) ** n_competitors


def clairvoyant_no_budget(
    bid_grid: NDArray,
    value: float,
    win_probs: NDArray,
) -> tuple[float, float]:
    """Per-round clairvoyant reward with no budget constraint.

    Returns the expected per-round reward of the best fixed bid:
        max_b (v - b) * P(m <= b)

    Parameters
    ----------
    bid_grid : (K,) array
    value : float
    win_probs : (K,) array   — P(m <= b) for each bid b

    Returns
    -------
    best_bid : float
    per_round_reward : float
    """
    expected_rewards = (value - np.asarray(bid_grid, dtype=float)) * np.asarray(
        win_probs, dtype=float
    )
    expected_rewards = np.maximum(expected_rewards, 0.0)
    best_k = int(np.argmax(expected_rewards))
    return float(bid_grid[best_k]), float(expected_rewards[best_k])


def clairvoyant_with_budget(
    bid_grid: NDArray,
    value: float,
    rho: float,
    win_probs: NDArray,
) -> tuple[NDArray, float, float]:
    """LP clairvoyant subject to per-round budget rho.

    Solves:
        max_{gamma in Delta(B)}  sum_b gamma(b) * (v - b) * P(m <= b)
        s.t.  sum_b gamma(b) * b * P(m <= b) <= rho
              sum_b gamma(b) = 1,  gamma(b) in [0, 1]

    This is the stochastic-setting clairvoyant from notebook 07
    (compute_clairvoyant). It knows the true win probabilities and is
    allowed to violate the budget constraint on individual rounds as long
    as the expected cost satisfies the constraint.

    Parameters
    ----------
    bid_grid : (K,) array
    value : float
    rho : float           — per-round budget = B_total / T
    win_probs : (K,) array

    Returns
    -------
    gamma : (K,) array    — optimal mixed strategy over bids
    per_round_reward : float
    per_round_cost : float
    """
    bid_grid = np.asarray(bid_grid, dtype=float)
    win_probs = np.asarray(win_probs, dtype=float)
    f = (value - bid_grid) * win_probs
    c = bid_grid * win_probs
    res = optimize.linprog(
        -f,
        A_ub=[c],
        b_ub=[rho],
        A_eq=[np.ones(len(bid_grid))],
        b_eq=[1.0],
        bounds=(0.0, 1.0),
        method="highs",
    )
    gamma = np.maximum(res.x, 0.0)
    return gamma, float(-res.fun), float(c @ gamma)
