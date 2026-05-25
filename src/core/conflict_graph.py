"""
conflict_graph.py
-----------------
Conflict-graph helper built on networkx.

A conflict graph G = (V, E) encodes which campaigns cannot be bid on
simultaneously. A valid action is a subset S ⊆ V that is an **independent
set** of G (no two campaigns in S share a conflict edge).

For small N (≤ ~12 campaigns), full enumeration of independent sets is
cheap. For larger N, use the greedy or maximum-weight-IS heuristics.
"""

from __future__ import annotations

import itertools
from typing import Iterable

import networkx as nx
import numpy as np
from numpy.typing import NDArray


class ConflictGraph:
    """Wrapper around networkx.Graph for campaign-conflict modelling.

    Parameters
    ----------
    n_campaigns : int
        Number of campaigns (nodes are 0-indexed integers 0..n_campaigns-1).
    edges : iterable of (i, j) pairs, optional
        Conflict edges to add at construction time.
    """

    def __init__(
        self,
        n_campaigns: int,
        edges: Iterable[tuple[int, int]] | None = None,
    ):
        self.n = n_campaigns
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n_campaigns))
        if edges:
            for u, v in edges:
                self._validate_node(u)
                self._validate_node(v)
                self.G.add_edge(u, v)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_conflict(self, i: int, j: int) -> None:
        self._validate_node(i)
        self._validate_node(j)
        self.G.add_edge(i, j)

    # ------------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------------

    def is_independent_set(self, subset: Iterable[int]) -> bool:
        """Return True iff no two nodes in `subset` share a conflict edge."""
        nodes = list(subset)
        for u, v in itertools.combinations(nodes, 2):
            if self.G.has_edge(u, v):
                return False
        return True

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def all_independent_sets(self, include_empty: bool = False) -> list[frozenset]:
        """Enumerate all independent sets of G.

        WARNING: exponential in n; only use for small n (≤ ~12).

        Returns
        -------
        list of frozensets, sorted by size descending
        """
        result = []
        for r in range(1, self.n + 1):
            for subset in itertools.combinations(range(self.n), r):
                if self.is_independent_set(subset):
                    result.append(frozenset(subset))
        if include_empty:
            result.append(frozenset())
        # Largest subsets first (usually most useful)
        result.sort(key=len, reverse=True)
        return result

    def maximal_independent_sets(self) -> list[frozenset]:
        """Return all maximal independent sets (cannot add any more node)."""
        return [
            frozenset(s) for s in nx.find_cliques(nx.complement(self.G))
        ]

    # ------------------------------------------------------------------
    # Maximum-weight independent set (exact, small N)
    # ------------------------------------------------------------------

    def max_weight_independent_set(
        self,
        weights: NDArray[np.float64],
    ) -> tuple[frozenset, float]:
        """Return the independent set maximising sum of weights.

        Exact exhaustive search — only suitable for small n.

        Parameters
        ----------
        weights : (n,) array of per-campaign weights (e.g. UCB utilities)

        Returns
        -------
        best_set  : frozenset of campaign indices
        best_value : float
        """
        weights = np.asarray(weights, dtype=float)
        assert len(weights) == self.n

        best_set: frozenset = frozenset()
        best_val: float = -np.inf

        for r in range(1, self.n + 1):
            for subset in itertools.combinations(range(self.n), r):
                if self.is_independent_set(subset):
                    val = weights[list(subset)].sum()
                    if val > best_val:
                        best_val = val
                        best_set = frozenset(subset)

        return best_set, float(best_val)

    # ------------------------------------------------------------------
    # Greedy feasibility repair
    # ------------------------------------------------------------------

    def greedy_independent_set(
        self,
        weights: NDArray[np.float64],
    ) -> frozenset:
        """Fast greedy max-weight IS: add nodes in weight-descending order,
        skip if adding would violate independence.

        Use as a cheap approximation when n is large.
        """
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(-weights)  # descending
        selected: set[int] = set()
        for node in order:
            if not any(self.G.has_edge(node, s) for s in selected):
                selected.add(int(node))
        return frozenset(selected)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_node(self, v: int) -> None:
        if not (0 <= v < self.n):
            raise ValueError(f"Node {v} out of range [0, {self.n})")

    def __repr__(self) -> str:
        return (
            f"ConflictGraph(n={self.n}, "
            f"edges={list(self.G.edges())})"
        )
