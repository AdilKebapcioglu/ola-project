"""
src/utils/regret.py
--------------------
Helpers for computing and summarising pseudo-regret across trials.

Terminology (consistent with project notation)
-----------------------------------------------
- clairvoyant_reward : float
    Expected reward per round of the best fixed action in hindsight (or the
    LP clairvoyant value).  Multiplied by T to give the cumulative clairvoyant.
- rewards : array of shape (T,)
    Realised rewards collected by the algorithm in one trial.
- regret : array of shape (T,)
    Cumulative pseudo-regret up to each round.

Usage
-----
    from src.utils.regret import cumulative_regret, summarise_regret

    # Single trial
    reg = cumulative_regret(rewards, clairvoyant_reward=0.35)

    # Across trials  →  shape (n_trials, T)
    all_regrets = np.stack([cumulative_regret(r, 0.35) for r in all_rewards])
    mean, std = summarise_regret(all_regrets)
"""

import numpy as np


def cumulative_regret(
    rewards: np.ndarray,
    clairvoyant_reward: float | np.ndarray,
) -> np.ndarray:
    """
    Compute cumulative pseudo-regret for a single trial.

    Regret_t = sum_{s=1}^{t} clairvoyant_reward_s - sum_{s=1}^{t} reward_s

    Parameters
    ----------
    rewards : np.ndarray, shape (T,)
        Per-round realised rewards from the algorithm.
    clairvoyant_reward : float or np.ndarray of shape (T,)
        Expected per-round reward of the clairvoyant benchmark. A scalar for
        stationary benchmarks; a (T,) array for dynamic benchmarks (e.g. the
        per-interval clairvoyant of a piecewise-stationary environment).

    Returns
    -------
    np.ndarray, shape (T,)
        Cumulative regret at each round.
    """
    rewards = np.asarray(rewards, dtype=float)
    T = len(rewards)
    clairvoyant_reward = np.asarray(clairvoyant_reward, dtype=float)
    if clairvoyant_reward.ndim == 0:
        clairvoyant_cumulative = float(clairvoyant_reward) * np.arange(1, T + 1)
    else:
        assert clairvoyant_reward.shape == (T,), (
            f"clairvoyant_reward must be scalar or shape ({T},), "
            f"got {clairvoyant_reward.shape}"
        )
        clairvoyant_cumulative = np.cumsum(clairvoyant_reward)
    return clairvoyant_cumulative - np.cumsum(rewards)


def summarise_regret(
    regret_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of cumulative regret across trials.

    Parameters
    ----------
    regret_matrix : np.ndarray, shape (n_trials, T)
        Each row is the cumulative regret array for one trial.

    Returns
    -------
    mean : np.ndarray, shape (T,)
    std  : np.ndarray, shape (T,)
    """
    regret_matrix = np.asarray(regret_matrix, dtype=float)
    return regret_matrix.mean(axis=0), regret_matrix.std(axis=0)
