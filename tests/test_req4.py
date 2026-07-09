"""
tests/test_req4.py
------------------
Phase 4 sanity checks:
  1. SlightlyNonStationaryMultiCampaignEnv: interval structure is seed-stable,
     every boundary is a change point, samples are resampled per trial.
  2. clairvoyant_piecewise returns a valid per-round reward sequence.
  3. cumulative_regret accepts scalar and (T,) benchmarks consistently.
  4. CUCB-SW and CUCB-CD run end-to-end without violating the budget.

Run with:  python -m pytest tests/test_req4.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.core.nonstationary_env import SlightlyNonStationaryMultiCampaignEnv
from src.core.conflict_graph import ConflictGraph
from src.core.clairvoyant import clairvoyant_piecewise
from src.core.nonstationary_cucb import (
    SWCombUCBLikeBidderAgent,
    CDCombUCBLikeBidderAgent,
)
from src.utils.seed_manager import make_rng
from src.utils.regret import cumulative_regret


N = 3
T = 400
VALUES = np.array([0.9, 0.8, 0.7])
BID_GRID = np.linspace(0.0, 1.0, 10)
EDGES = [(0, 1)]
RHO = 0.3


def _make_env(**kwargs):
    return SlightlyNonStationaryMultiCampaignEnv(
        values=VALUES, T=T, sequence_seed=123, n_intervals=4, **kwargs
    )


def test_env_structure_seed_stable_and_changing():
    env1 = _make_env()
    env2 = _make_env()
    # Same sequence_seed -> identical structure
    assert np.array_equal(env1.k_matrix, env2.k_matrix)
    assert np.array_equal(env1.boundaries, env2.boundaries)
    assert env1.k_seq.shape == (T, N)
    assert env1.boundaries[0] == 0 and env1.boundaries[-1] == T
    # Every boundary is a genuine change point for every campaign
    assert (np.diff(env1.k_matrix, axis=0) != 0).all()


def test_env_resamples_per_trial():
    env = _make_env()
    bids = np.zeros(N)

    env.reset(make_rng(1))
    m_a = np.array([env.round(bids)["competing_bids"] for _ in range(50)])
    env.reset(make_rng(2))
    m_b = np.array([env.round(bids)["competing_bids"] for _ in range(50)])
    env.reset(make_rng(1))
    m_c = np.array([env.round(bids)["competing_bids"] for _ in range(50)])

    assert not np.allclose(m_a, m_b)   # different trial rngs -> different bids
    assert np.allclose(m_a, m_c)       # same trial rng -> reproducible


def test_clairvoyant_piecewise_shape_and_values():
    env = _make_env()
    cg = ConflictGraph(N, EDGES)
    reward_seq, info = clairvoyant_piecewise(
        BID_GRID, VALUES, RHO, env.k_matrix, env.boundaries, cg
    )
    assert reward_seq.shape == (T,)
    assert len(info) == env.n_intervals
    assert (reward_seq >= 0).all()
    # Piecewise constant: constant within each interval
    for j in range(env.n_intervals):
        seg = reward_seq[env.boundaries[j]:env.boundaries[j + 1]]
        assert np.allclose(seg, seg[0])
        assert np.isclose(seg[0], info[j]["reward"])


def test_cumulative_regret_scalar_vs_array():
    rewards = np.array([0.1, 0.2, 0.3])
    scalar = cumulative_regret(rewards, 0.5)
    array = cumulative_regret(rewards, np.full(3, 0.5))
    assert np.allclose(scalar, array)
    assert np.isclose(scalar[-1], 1.5 - 0.6)


def _run_agent(agent, budget):
    env = _make_env()
    env.reset(make_rng(7))
    agent.reset()
    total_cost = 0.0
    for _ in range(T):
        bids = agent.select_action()
        assert (bids >= 0).all() and (bids <= BID_GRID.max()).all()
        fb = env.round(bids)
        total_cost += fb["cost"]
        agent.update(fb)
    assert total_cost <= budget + 1e-9
    return total_cost


def test_sw_agent_runs_and_respects_budget():
    budget = RHO * T
    cg = ConflictGraph(N, EDGES)
    agent = SWCombUCBLikeBidderAgent(
        bid_grid=BID_GRID, values=VALUES, budget=budget, T=T,
        conflict_graph=cg, W=50, rng=make_rng(0),
    )
    _run_agent(agent, budget)
    # Windowed counts can never exceed the window length
    assert agent.n_pulls.max() <= agent.W
    assert np.log(agent.W) == agent._log_horizon


def test_cd_agent_runs_and_respects_budget():
    budget = RHO * T
    cg = ConflictGraph(N, EDGES)
    agent = CDCombUCBLikeBidderAgent(
        bid_grid=BID_GRID, values=VALUES, budget=budget, T=T,
        conflict_graph=cg, M=5, h=4.0, alpha=0.05, rng=make_rng(0),
    )
    _run_agent(agent, budget)
    # Detectors saw data; reset bookkeeping is consistent
    assert agent._cd_n.sum() > 0
    assert all(len(h) == int(n) for h, n in zip(agent.reset_history, agent.n_resets))
