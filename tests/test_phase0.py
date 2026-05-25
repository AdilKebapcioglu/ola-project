"""
tests/test_phase0.py
--------------------
Phase 0 sanity checks:
  1. Random-bidding baseline runs end-to-end on a toy 2-campaign instance.
  2. Budget exhaustion behaves correctly.
  3. Conflict graph rejects infeasible subsets.

Run with:  python -m pytest tests/test_phase0.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.core.environment import BiddingEnvironment, first_price_outcome, multi_campaign_outcomes
from src.core.budget import BudgetTracker
from src.core.conflict_graph import ConflictGraph
from src.utils.seed_manager import make_rng, make_trial_rngs
from src.utils.regret import cumulative_regret, summarise_regret


# ---------------------------------------------------------------------------
# Minimal concrete environment for testing
# (matches the protocol expected by runner.py: reset(rng) and round(action))
# ---------------------------------------------------------------------------

class ToyStochasticEnv(BiddingEnvironment):
    """Competing bids drawn i.i.d. Uniform(0, 1) per campaign."""

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._t = 0

    def _sample_competing_bids(self) -> np.ndarray:
        return self.rng.uniform(0, 1, size=self.N)


# ---------------------------------------------------------------------------
# 1. Random-bidding end-to-end test
# ---------------------------------------------------------------------------

def test_random_bidding_end_to_end():
    """Run 100 rounds of random bidding on a 2-campaign instance."""
    N = 2
    T = 100
    values = np.array([0.8, 0.6])
    bid_grid = np.linspace(0, 1, 10)

    rng = make_rng(seed=42)
    env = ToyStochasticEnv(values=values, T=T, rng=rng)
    env.reset(rng)

    total_utility = 0.0
    total_cost = 0.0

    for _ in range(T):
        bids = rng.choice(bid_grid, size=N)
        utility, cost, info = env.step(bids)
        assert "competing_bids" in info
        assert "won" in info
        assert len(info["won"]) == N
        assert cost >= 0.0
        total_utility += utility
        total_cost += cost

    assert env._t == T


# ---------------------------------------------------------------------------
# 2. Budget exhaustion
# ---------------------------------------------------------------------------

def test_budget_exhaustion():
    """BudgetTracker.is_exhausted triggers correctly; clamp zeros bids."""
    tracker = BudgetTracker(total_budget=10.0, T=100)

    tracker.consume(9.99)
    assert not tracker.is_exhausted

    tracker.consume(0.02)
    assert tracker.is_exhausted
    assert tracker.remaining == 0.0

    bids = np.array([0.3, 0.5, 0.7])
    clamped = tracker.clamp_bids(bids)
    np.testing.assert_array_equal(clamped, np.zeros(3))


def test_budget_reset():
    tracker = BudgetTracker(total_budget=5.0, T=50)
    tracker.consume(3.0)
    tracker.reset()
    assert tracker.spent == 0.0
    assert not tracker.is_exhausted


def test_budget_rho():
    tracker = BudgetTracker(total_budget=100.0, T=200)
    assert tracker.rho == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. Conflict graph
# ---------------------------------------------------------------------------

def test_conflict_graph_feasibility():
    """Conflict graph correctly rejects subsets with conflict edges."""
    cg = ConflictGraph(n_campaigns=3, edges=[(0, 1), (1, 2), (0, 2)])

    assert cg.is_independent_set([0]) is True
    assert cg.is_independent_set([0, 1]) is False
    assert cg.is_independent_set([0, 2]) is False
    assert cg.is_independent_set([1, 2]) is False
    assert cg.is_independent_set([0, 1, 2]) is False


def test_conflict_graph_independent_sets():
    """all_independent_sets returns only valid sets."""
    cg = ConflictGraph(n_campaigns=4, edges=[(0, 1), (1, 2), (2, 3)])
    all_IS = cg.all_independent_sets()

    for s in all_IS:
        assert cg.is_independent_set(s)

    max_size = max(len(s) for s in all_IS)
    assert max_size == 2


def test_conflict_graph_max_weight_IS():
    """max_weight_independent_set picks highest-weight feasible set."""
    cg = ConflictGraph(n_campaigns=4, edges=[(0, 1), (0, 2), (0, 3)])
    weights = np.array([10.0, 4.0, 4.0, 4.0])

    best_set, best_val = cg.max_weight_independent_set(weights)

    assert best_set == frozenset({1, 2, 3})
    assert best_val == pytest.approx(12.0)


def test_conflict_graph_no_edges():
    """With no conflict edges, all campaigns can be selected together."""
    cg = ConflictGraph(n_campaigns=5)
    assert cg.is_independent_set(list(range(5))) is True


# ---------------------------------------------------------------------------
# 4. Auction logic unit tests
# ---------------------------------------------------------------------------

def test_first_price_win():
    won, utility, cost = first_price_outcome(bid=0.5, competing_bid=0.4, value=1.0)
    assert won is True
    assert utility == pytest.approx(0.5)
    assert cost == pytest.approx(0.5)


def test_first_price_loss():
    won, utility, cost = first_price_outcome(bid=0.3, competing_bid=0.4, value=1.0)
    assert won is False
    assert utility == pytest.approx(0.0)
    assert cost == pytest.approx(0.0)


def test_first_price_tie_wins():
    won, utility, cost = first_price_outcome(bid=0.5, competing_bid=0.5, value=1.0)
    assert won is True


def test_multi_campaign_outcomes():
    bids      = np.array([0.6, 0.3, 0.8])
    competing = np.array([0.5, 0.4, 0.7])
    values    = np.array([1.0, 1.0, 1.0])
    won, utilities, costs = multi_campaign_outcomes(bids, competing, values)

    np.testing.assert_array_equal(won, [True, False, True])
    np.testing.assert_allclose(utilities, [0.4, 0.0, 0.2])
    np.testing.assert_allclose(costs,     [0.6, 0.0, 0.8])


# ---------------------------------------------------------------------------
# 5. seed_manager tests (matches colleague's API: make_rng / make_trial_rngs)
# ---------------------------------------------------------------------------

def test_make_rng_reproducibility():
    """Same seed → same sequence."""
    rng_a = make_rng(seed=99)
    rng_b = make_rng(seed=99)
    assert rng_a.random() == rng_b.random()


def test_make_trial_rngs_independence():
    """Different trial indices → different sequences."""
    rngs = make_trial_rngs(master_seed=99, n_trials=2)
    assert rngs[0].random() != rngs[1].random()


def test_make_trial_rngs_reproducibility():
    """Same master seed → same list of sequences."""
    rngs_a = make_trial_rngs(master_seed=42, n_trials=3)
    rngs_b = make_trial_rngs(master_seed=42, n_trials=3)
    for a, b in zip(rngs_a, rngs_b):
        assert a.random() == b.random()


# ---------------------------------------------------------------------------
# 6. regret helpers (matches colleague's API: cumulative_regret / summarise_regret)
# ---------------------------------------------------------------------------

def test_cumulative_regret_shape():
    rewards = np.ones(100) * 0.3
    reg = cumulative_regret(rewards, clairvoyant_reward=0.5)
    assert reg.shape == (100,)


def test_cumulative_regret_values():
    """With constant reward 0 and clairvoyant 1, regret = t."""
    rewards = np.zeros(10)
    reg = cumulative_regret(rewards, clairvoyant_reward=1.0)
    np.testing.assert_allclose(reg, np.arange(1, 11))


def test_summarise_regret():
    matrix = np.array([[1.0, 2.0, 3.0],
                       [3.0, 4.0, 5.0]])
    mean, std = summarise_regret(matrix)
    np.testing.assert_allclose(mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(std,  [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])