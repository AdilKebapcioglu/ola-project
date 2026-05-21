"""
src/utils/seed_manager.py
--------------------------
Thin wrapper around np.random.Generator for reproducible experiments.

Usage
-----
    from src.utils.seed_manager import make_rng, make_trial_rngs

    # Single experiment
    rng = make_rng(seed=42)

    # Multi-trial: one independent rng per trial derived from a master seed
    rngs = make_trial_rngs(master_seed=42, n_trials=20)
"""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh Generator seeded deterministically from `seed`."""
    return np.random.default_rng(seed)


def make_trial_rngs(master_seed: int, n_trials: int) -> list[np.random.Generator]:
    """
    Return a list of `n_trials` independent Generators.

    Each trial gets its own seed derived from `master_seed` via SeedSequence
    spawning, which guarantees statistical independence between trials.

    Parameters
    ----------
    master_seed : int
        Top-level seed stored in data/seeds/ for the experiment.
    n_trials : int
        Number of independent trials to run.

    Returns
    -------
    list of np.random.Generator, length n_trials
    """
    ss = np.random.SeedSequence(master_seed)
    child_sequences = ss.spawn(n_trials)
    return [np.random.default_rng(child) for child in child_sequences]
