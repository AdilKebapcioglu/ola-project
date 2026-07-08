"""
clairvoyant.py
--------------
Clairvoyant benchmark computations for Phases 1 and 2.

Phase 1 (single campaign):
  No-budget clairvoyant:   best fixed bid b* in expectation.
  Budget clairvoyant:      LP over mixed bid strategies (notebook 07 style).

Phase 2 (multi-campaign):
  win_probs_multi_campaign: per-campaign win-probability matrices.
  clairvoyant_multi_campaign: LP over per-IS mixed strategies subject to
    shared budget and conflict-graph IS constraint.
    Exact (LP) for N <= 6; document fallback for larger N.

Win probability helpers: analytical formulas for Beta(n, 1) competing bids.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from src.core.conflict_graph import ConflictGraph


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


# ---------------------------------------------------------------------------
# Phase 2 helpers — multi-campaign, stochastic
# ---------------------------------------------------------------------------

def win_probs_multi_campaign(
    bid_grid: NDArray,
    n_competitors_list,
) -> NDArray[np.float64]:
    """Win-probability matrix for N independent campaigns.

    Each campaign i has competing bids drawn from Beta(n_competitors[i], 1),
    so P(m_i <= b) = b^{n_competitors[i]}.

    Parameters
    ----------
    bid_grid : (K,) array
    n_competitors_list : (N,) array-like of int

    Returns
    -------
    win_probs : (N, K) array   — win_probs[i, k] = P(m_i <= bid_grid[k])
    """
    bid_grid = np.asarray(bid_grid, dtype=float)
    n_competitors_list = np.asarray(n_competitors_list, dtype=int)
    return np.stack([
        win_probs_beta_uniform(bid_grid, int(nc))
        for nc in n_competitors_list
    ])  # (N, K)


def clairvoyant_multi_campaign(
    bid_grid: NDArray,
    values: NDArray,
    rho: float,
    win_probs: NDArray,
    conflict_graph: ConflictGraph,
) -> tuple[dict[int, NDArray], float, float]:
    """LP clairvoyant for multi-campaign bidding with budget and IS constraint.

    For each feasible independent set S, solves the joint LP over per-campaign
    mixed strategies (same structure as the agent LP, but with true win
    probabilities instead of UCB/LCB estimates). Returns the best IS.

    NOTE: Exact enumeration is only feasible for N <= 6.
    For larger N, use the best fixed pure action in expectation instead and
    document this explicitly in the notebook.

    Parameters
    ----------
    bid_grid  : (K,) array
    values    : (N,) array   — per-campaign private values v_i
    rho       : float        — per-round budget = B_total / T
    win_probs : (N, K) array — P(m_i <= b_k) for each campaign i and bid k
    conflict_graph : ConflictGraph

    Returns
    -------
    best_gamma : dict[int, NDArray]  — per-campaign bid distributions for best IS
    per_round_reward : float
    per_round_cost   : float
    """
    bid_grid = np.asarray(bid_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    win_probs = np.asarray(win_probs, dtype=float)
    K = len(bid_grid)

    all_IS = conflict_graph.all_independent_sets()

    best_reward = -np.inf
    best_gamma: dict[int, NDArray] = {}
    best_cost = 0.0

    for S in all_IS:
        S_list = sorted(S)
        ns = len(S_list)
        n_vars = ns * K

        # True expected utility and cost per (campaign, bid)
        f_true = np.array([
            (values[i] - bid_grid) * win_probs[i]
            for i in S_list
        ])  # (ns, K)
        c_true = np.array([
            bid_grid * win_probs[i]
            for i in S_list
        ])  # (ns, K)

        c_obj = -f_true.ravel()
        A_ub = c_true.ravel().reshape(1, -1)
        b_ub = np.array([rho])

        A_eq = np.zeros((ns, n_vars))
        for j in range(ns):
            A_eq[j, j * K : (j + 1) * K] = 1.0
        b_eq = np.ones(ns)

        res = optimize.linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0.0, 1.0),
            method="highs",
        )

        if not res.success:
            continue

        lp_reward = float(-res.fun)
        if lp_reward > best_reward:
            best_reward = lp_reward
            gamma_flat = np.maximum(res.x, 0.0)
            best_cost = float(c_true.ravel() @ gamma_flat)
            best_gamma = {}
            for j, i in enumerate(S_list):
                gamma_i = gamma_flat[j * K : (j + 1) * K]
                total = gamma_i.sum()
                best_gamma[i] = (
                    gamma_i / total if total > 1e-12 else np.ones(K) / K
                )

    return best_gamma, float(best_reward), float(best_cost)
