"""
tests/test_req3.py
------------------
Phase 3 sanity checks:
  1. PrimalDualBidderAgent runs end-to-end without violating the budget.
  2. Dual variable lambda always stays within its projection bounds [0, 1/rho].
  3. Agent bids all-zero once the remaining budget can't cover a worst-case round.

Run with:  python -m pytest tests/test_req3.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.core.multi_campaign_env import MultiCampaignEnv
from src.core.conflict_graph import ConflictGraph
from src.core.primal_dual_bidder import PrimalDualBidderAgent
from src.utils.seed_manager import make_rng


N = 3
K = 10
T = 400
VALUES = np.array([0.9, 0.8, 0.7])
N_COMPETITORS = np.array([3, 2, 4])
BID_GRID = np.linspace(0.0, 1.0, K)
EDGES = [(0, 1)]
RHO = 0.3


def _make_env():
    return MultiCampaignEnv(values=VALUES, T=T, n_competitors=N_COMPETITORS)


def _make_agent(rng_seed=0, **kwargs):
    cg = ConflictGraph(N, EDGES)
    return PrimalDualBidderAgent(
        bid_grid=BID_GRID, values=VALUES, budget=RHO * T, T=T,
        conflict_graph=cg, rng=make_rng(rng_seed), **kwargs
    )


def test_pd_agent_runs_and_respects_budget():
    budget = RHO * T
    env = _make_env()
    agent = _make_agent()
    env.reset(make_rng(1))
    agent.reset()

    total_cost = 0.0
    for _ in range(T):
        bids = agent.select_action()
        assert (bids >= 0).all() and (bids <= BID_GRID.max()).all()
        fb = env.round(bids)
        total_cost += fb["cost"]
        agent.update(fb)
    assert total_cost <= budget + 1e-9


def test_pd_agent_lambda_stays_in_bounds():
    env = _make_env()
    agent = _make_agent(lmbd_init=0.0, adaptive_rho=True)
    env.reset(make_rng(2))
    agent.reset()

    for _ in range(T):
        fb = env.round(agent.select_action())
        agent.update(fb)
        assert 0.0 <= agent.lmbd <= agent.lmbd_max


def test_pd_agent_stops_bidding_when_budget_exhausted():
    agent = _make_agent(rng_seed=3)
    agent.reset()
    agent._budget = 0.0  # below the worst-case cost of any round
    bids = agent.select_action()
    assert (bids == 0).all()
