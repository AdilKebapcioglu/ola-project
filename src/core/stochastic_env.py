"""
stochastic_env.py
-----------------
Concrete stochastic bidding environments for Phases 1 and 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.environment import BiddingEnvironment


class SingleCampaignEnv(BiddingEnvironment):
    """Single-campaign stochastic bidding environment.

    Competing bid m_t is drawn i.i.d. from Beta(n_competitors, 1) each round,
    which equals the distribution of the maximum of n_competitors independent
    Uniform(0, 1) bids.

    Parameters
    ----------
    value : float
        Private value v for winning the campaign.
    T : int
        Horizon (informational; env does not enforce it).
    n_competitors : int
        Number of competing advertisers. Higher → harder to win.
    rng : np.random.Generator, optional
        Initial RNG; overridden by reset(rng) before each trial.
    """

    def __init__(
        self,
        value: float,
        T: int,
        n_competitors: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            values=np.array([float(value)]),
            T=T,
            rng=rng or np.random.default_rng(),
        )
        self.n_competitors = n_competitors

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._t = 0

    def _sample_competing_bids(self) -> NDArray[np.float64]:
        return np.array([self.rng.beta(self.n_competitors, 1)])
