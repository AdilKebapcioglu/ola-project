"""
src/utils/runner.py
--------------------
Generic multi-trial runner.

The runner is deliberately thin: it knows nothing about the specific
environment or agent classes — it just calls the protocol methods and
collects results. This keeps it reusable across all four requirements.

Protocol expected from `env` and `agent`
-----------------------------------------
env:
    env.reset(rng)          → None          (re-initialises env with a fresh rng)
    env.round(bids)         → dict          (at minimum keys: "rewards", "costs")

agent:
    agent.reset()           → None          (re-initialises agent state)
    agent.select_action()   → action        (whatever the env expects as `bids`)
    agent.update(feedback)  → None          (feedback = dict returned by env.round)

Usage
-----
    from src.utils.runner import run_trials
    from src.utils.regret import summarise_regret

    results = run_trials(env, agent, T=1000, n_trials=20, master_seed=42)

    mean_regret, std_regret = summarise_regret(results["regret_matrix"])
"""

import numpy as np
from typing import Any

from src.utils.seed_manager import make_trial_rngs
from src.utils.regret import cumulative_regret


def run_trials(
    env: Any,
    agent: Any,
    T: int,
    n_trials: int,
    master_seed: int,
    clairvoyant_reward: float,
) -> dict[str, np.ndarray]:
    """
    Run `n_trials` independent repetitions of the env–agent interaction loop.

    Each trial uses an independent rng derived from `master_seed` via
    SeedSequence spawning (see seed_manager.py).

    Parameters
    ----------
    env : object
        Must implement reset(rng) and round(action).
    agent : object
        Must implement reset(), select_action(), and update(feedback).
    T : int
        Number of rounds per trial.
    n_trials : int
        Number of independent trials.
    master_seed : int
        Top-level seed for the experiment (store in data/seeds/).
    clairvoyant_reward : float
        Expected per-round reward of the benchmark. Used to compute regret.

    Returns
    -------
    dict with keys:
        "reward_matrix"  : np.ndarray, shape (n_trials, T)
        "cost_matrix"    : np.ndarray, shape (n_trials, T)
        "regret_matrix"  : np.ndarray, shape (n_trials, T)  ← cumulative regret
    """
    # NOTE on seeding: the course notebooks use np.random.seed(i) per trial.
    # We use SeedSequence.spawn() instead — it gives statistically independent
    # streams with no risk of overlap, which np.random.seed(i) does not guarantee.
    # Do not revert to the notebook pattern.
    rngs = make_trial_rngs(master_seed, n_trials)

    reward_matrix = np.zeros((n_trials, T))
    cost_matrix   = np.zeros((n_trials, T))

    for trial_idx, rng in enumerate(rngs):
        env.reset(rng)
        agent.reset()

        for t in range(T):
            action   = agent.select_action()
            feedback = env.round(action)

            reward_matrix[trial_idx, t] = feedback["reward"]
            cost_matrix[trial_idx, t]   = feedback["cost"]

            agent.update(feedback)

    regret_matrix = np.stack([
        cumulative_regret(reward_matrix[i], clairvoyant_reward)
        for i in range(n_trials)
    ])

    return {
        "reward_matrix": reward_matrix,
        "cost_matrix":   cost_matrix,
        "regret_matrix": regret_matrix,
    }
