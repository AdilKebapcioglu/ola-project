"""
src/utils/parallel_runner.py
-----------------------------
Multi-process variant of `run_trials` for agents where per-round cost (e.g.
solving one LP per feasible independent set, as in CombUCBLikeBidderAgent)
dominates process-spawn overhead. Trials are independent by construction, so
running them across worker processes changes only *how* the same n_trials
are executed, not what is computed per trial — same seeding guarantees,
same result shapes.

Protocol
--------
Callers pass the env/agent *classes* plus their constructor kwargs (not
instances): each worker constructs its own env/agent, since instances
cannot be shared across processes. `rng` must be omitted from
`env_kwargs` / `agent_kwargs` — an independent Generator is injected per
trial from two SeedSequence streams spawned off `master_seed`.

Usage
-----
    from src.utils.parallel_runner import run_trials_parallel

    results = run_trials_parallel(
        env_cls=MultiCampaignEnv, env_kwargs=dict(values=values, T=T, n_competitors=n_competitors),
        agent_cls=CombUCBLikeBidderAgent, agent_kwargs=dict(bid_grid=bid_grid, values=values, budget=B_total, T=T, conflict_graph=cg),
        T=T, n_trials=25, master_seed=42, clairvoyant_reward=0.35,
    )
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from src.utils.regret import cumulative_regret


def _run_single_trial(
    env_cls: type,
    env_kwargs: dict[str, Any],
    agent_cls: type,
    agent_kwargs: dict[str, Any],
    T: int,
    env_rng: np.random.Generator,
    agent_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one full trial in a worker process; returns (rewards, costs)."""
    env = env_cls(**env_kwargs)
    agent = agent_cls(**agent_kwargs, rng=agent_rng)
    env.reset(env_rng)
    agent.reset()

    rewards = np.zeros(T)
    costs = np.zeros(T)
    for t in range(T):
        action = agent.select_action()
        feedback = env.round(action)
        rewards[t] = feedback["reward"]
        costs[t] = feedback["cost"]
        agent.update(feedback)
    return rewards, costs


def run_trials_parallel(
    env_cls: type,
    env_kwargs: dict[str, Any],
    agent_cls: type,
    agent_kwargs: dict[str, Any],
    T: int,
    n_trials: int,
    master_seed: int,
    clairvoyant_reward: float,
    n_jobs: int | None = None,
) -> dict[str, np.ndarray]:
    """Parallel drop-in replacement for `run_trials` (see module docstring).

    Parameters
    ----------
    env_cls, agent_cls : type
        Environment/agent classes, constructed fresh inside each worker.
    env_kwargs, agent_kwargs : dict
        Constructor kwargs, excluding `rng`.
    T : int
        Rounds per trial.
    n_trials : int
        Number of independent trials.
    master_seed : int
        Top-level seed; env and agent rngs are spawned from two independent
        child SeedSequences of this seed, so results are fully reproducible.
    clairvoyant_reward : float
        Expected per-round reward of the benchmark, for regret computation.
    n_jobs : int, optional
        Worker processes; defaults to `ProcessPoolExecutor`'s own default
        (`os.cpu_count()`).

    Returns
    -------
    dict with keys: "reward_matrix", "cost_matrix", "regret_matrix"
        Same shapes/semantics as `run_trials`.
    """
    env_seed, agent_seed = np.random.SeedSequence(master_seed).spawn(2)
    env_rngs = [np.random.default_rng(c) for c in env_seed.spawn(n_trials)]
    agent_rngs = [np.random.default_rng(c) for c in agent_seed.spawn(n_trials)]

    reward_matrix = np.zeros((n_trials, T))
    cost_matrix = np.zeros((n_trials, T))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _run_single_trial,
                env_cls, env_kwargs, agent_cls, agent_kwargs,
                T, env_rngs[i], agent_rngs[i],
            )
            for i in range(n_trials)
        ]
        for i, future in enumerate(futures):
            rewards, costs = future.result()
            reward_matrix[i] = rewards
            cost_matrix[i] = costs

    regret_matrix = np.stack([
        cumulative_regret(reward_matrix[i], clairvoyant_reward)
        for i in range(n_trials)
    ])

    return {
        "reward_matrix": reward_matrix,
        "cost_matrix": cost_matrix,
        "regret_matrix": regret_matrix,
    }
