"""
multi_campaign_env.py
---------------------
Stochastic bidding environment for N independent campaigns (Phase 2).

Each campaign i has its own competing-bid distribution:
  m_{i,t} ~ Beta(n_competitors[i], 1)   i.i.d. across rounds and campaigns.

This equals the distribution of the maximum of n_competitors[i] independent
Uniform(0,1) bids, matching the single-campaign model from Phase 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.environment import BiddingEnvironment


class MultiCampaignEnv(BiddingEnvironment):
    """Multi-campaign stochastic bidding environment.

    Parameters
    ----------
    values : (N,) array-like
        Per-campaign private values v_i.
    T : int
        Horizon (informational; env does not enforce it).
    n_competitors : (N,) array-like of int
        Number of competing advertisers per campaign. Higher → harder to win.
    rng : np.random.Generator, optional
        Initial RNG; overridden by reset(rng) before each trial.
    """

    def __init__(
        self,
        values,
        T: int,
        n_competitors,
        rng: np.random.Generator | None = None,
    ) -> None:
        values = np.asarray(values, dtype=float)
        super().__init__(
            values=values,
            T=T,
            rng=rng or np.random.default_rng(),
        )
        self.n_competitors = np.asarray(n_competitors, dtype=int)
        assert len(self.n_competitors) == self.N, (
            f"n_competitors length {len(self.n_competitors)} != N={self.N}"
        )

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._t = 0

    def _sample_competing_bids(self) -> NDArray[np.float64]:
        return self.rng.beta(self.n_competitors, np.ones(self.N, dtype=float))
