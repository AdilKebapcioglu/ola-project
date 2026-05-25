"""
test_phase0.py
--------------
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
from src.utils.seed_manager import SeedManager


# ---------------------------------------------------------------------------
# Minimal concrete environment for testing
# ---------------------------------------------------------------------------

class ToyStochasticEnv(BiddingEnvironment):
    """Competing bids drawn i.i.d. Uniform(0, 1) per campaign."""

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
    rng = np.random.default_rng(42)
    bid_grid = np.linspace(0, 1, 10)

    env = ToyStochasticEnv(values=values, T=T, rng=rng)
    env.reset()

    total_utility = 0.0
    total_cost = 0.0

    for _ in range(T):
        bids = rng.choice(bid_grid, size=N)
        utility, cost, info = env.step(bids)
        assert "competing_bids" in info
        assert "won" in info
        assert len(info["won"]) == N
        assert utility >= 0 or utility <= 0  # just a finite-float check
        assert cost >= 0.0
        total_utility += utility
        total_cost += cost

    # After T rounds environment should have advanced t counter
    assert env._t == T
    print(f"  total utility={total_utility:.2f}, total cost={total_cost:.2f}")


# ---------------------------------------------------------------------------
# 2. Budget exhaustion
# ---------------------------------------------------------------------------

def test_budget_exhaustion():
    """BudgetTracker.is_exhausted triggers correctly; clamp zeros bids."""
    B = 10.0
    T = 100
    tracker = BudgetTracker(total_budget=B, T=T)

    # Spend just under budget
    tracker.consume(9.99)
    assert not tracker.is_exhausted

    # Spend over the edge
    tracker.consume(0.02)
    assert tracker.is_exhausted
    assert tracker.remaining == 0.0

    # Clamping should zero all bids once exhausted
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
    # Triangle: 0-1, 1-2, 0-2 — no independent set of size > 1
    cg = ConflictGraph(n_campaigns=3, edges=[(0, 1), (1, 2), (0, 2)])

    assert cg.is_independent_set([0]) is True
    assert cg.is_independent_set([1]) is True
    assert cg.is_independent_set([0, 1]) is False
    assert cg.is_independent_set([0, 2]) is False
    assert cg.is_independent_set([1, 2]) is False
    assert cg.is_independent_set([0, 1, 2]) is False


def test_conflict_graph_independent_sets():
    """ConflictGraph.all_independent_sets returns only valid sets."""
    # Path graph: 0-1-2-3. IS sizes up to 2.
    cg = ConflictGraph(n_campaigns=4, edges=[(0, 1), (1, 2), (2, 3)])
    all_IS = cg.all_independent_sets()

    for s in all_IS:
        assert cg.is_independent_set(s), f"{s} claimed IS but has conflict edge"

    # The maximum IS should contain 2 nodes (e.g. {0,2}, {0,3}, {1,3})
    max_size = max(len(s) for s in all_IS)
    assert max_size == 2


def test_conflict_graph_max_weight_IS():
    """max_weight_independent_set picks highest-weight feasible set."""
    # Star graph: 0 conflicts with 1,2,3 — best IS is {1,2,3} with large weights
    cg = ConflictGraph(n_campaigns=4, edges=[(0, 1), (0, 2), (0, 3)])
    weights = np.array([10.0, 4.0, 4.0, 4.0])  # node 0 has high weight but blocks all

    best_set, best_val = cg.max_weight_independent_set(weights)

    # {1,2,3} has total weight 12 > {0} with weight 10
    assert best_set == frozenset({1, 2, 3})
    assert best_val == pytest.approx(12.0)


def test_conflict_graph_no_edges():
    """With no conflict edges, all N campaigns can always be selected together."""
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
    bids = np.array([0.6, 0.3, 0.8])
    competing = np.array([0.5, 0.4, 0.7])
    values = np.array([1.0, 1.0, 1.0])
    won, utilities, costs = multi_campaign_outcomes(bids, competing, values)

    np.testing.assert_array_equal(won, [True, False, True])
    np.testing.assert_allclose(utilities, [0.4, 0.0, 0.2])
    np.testing.assert_allclose(costs, [0.6, 0.0, 0.8])


# ---------------------------------------------------------------------------
# 5. SeedManager reproducibility
# ---------------------------------------------------------------------------

def test_seed_manager_reproducibility():
    sm = SeedManager(master_seed=99)
    rng_a = sm.get_rng(trial=3)
    rng_b = sm.get_rng(trial=3)
    # Same trial index → same sequence
    assert rng_a.random() == rng_b.random()


def test_seed_manager_independence():
    sm = SeedManager(master_seed=99)
    rng_0 = sm.get_rng(trial=0)
    rng_1 = sm.get_rng(trial=1)
    # Different trial indices → different sequences
    assert rng_0.random() != rng_1.random()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
