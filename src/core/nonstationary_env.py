"""
nonstationary_env.py
--------------------
Highly non-stationary multi-campaign bidding environment for Phase 3.

The competing bid for campaign i at round t is drawn from Beta(k_i(t), 1),
where the parameter sequence k_i(t) combines:
  - abrupt piecewise-constant jumps: each campaign's level re-draws every
    ~change_interval rounds (change points are staggered across campaigns,
    so the identity of the "best" independent set rotates over time), and
  - a slow sinusoidal drift with a per-campaign period and phase.

The whole sequence is generated once at construction from `sequence_seed`
and is NOT resampled by reset(): every trial faces the same fixed
(non-stochastic) sequence, so the only randomness across trials is the
agent's own. This matches the adversarial evaluation protocol of the
course material (fixed loss sequence, uncertainty from the algorithm
only) and makes the best-fixed-in-hindsight baseline well defined.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.environment import BiddingEnvironment


class NonStationaryMultiCampaignEnv(BiddingEnvironment):
    """Fixed highly non-stationary sequence of competing bids for N campaigns.

    m_{i,t} ~ Beta(k_i(t), 1), with k_i(t) = clip(base_i(t) + drift_i(t), k_min, k_max):
      base_i(t)  — piecewise constant, jumps to Uniform(k_range) at change
                   points whose spacing is Uniform(change_interval) rounds.
      drift_i(t) — drift_amplitude * sin(2*pi*t/period_i + phase_i).

    Larger k means stiffer competition (m concentrates near 1), so a jump in
    k_i(t) abruptly changes both the win probability and the price needed to
    win campaign i.

    Parameters
    ----------
    values : (N,) array
        Per-campaign private values v_i (fixed over time).
    T : int
        Horizon; the full (T, N) sequence of competing bids is pre-generated.
    sequence_seed : int
        Seed for the sequence generator. Same seed => identical sequence, so
        trials are comparable and the hindsight baseline is computed once.
    change_interval : tuple[int, int], default (50, 150)
        (min, max) spacing in rounds between abrupt jumps of base_i(t).
    k_range : tuple[float, float], default (1.0, 6.0)
        Range from which each new base level is drawn.
    drift_amplitude : float, default 1.0
        Amplitude of the sinusoidal drift added on top of the base level.
    drift_period_range : tuple[int, int], default (500, 2000)
        (min, max) period in rounds of each campaign's sinusoidal drift.

    Attributes
    ----------
    m_seq : (T, N) array
        The realized competing-bid sequence (read-only; used by the
        hindsight baseline and for plotting).
    k_seq : (T, N) array
        The underlying Beta parameter sequence (for plotting the
        non-stationarity).
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        T: int,
        sequence_seed: int,
        change_interval: tuple[int, int] = (50, 150),
        k_range: tuple[float, float] = (1.0, 6.0),
        drift_amplitude: float = 1.0,
        drift_period_range: tuple[int, int] = (500, 2000),
    ) -> None:
        gen_rng = np.random.default_rng(sequence_seed)
        super().__init__(values=np.asarray(values, dtype=float), T=T, rng=gen_rng)

        k_min = 0.3  # keep Beta parameter strictly positive after drift
        k_seq = np.zeros((T, self.N))
        for i in range(self.N):
            # Piecewise-constant base with staggered change points
            base = np.empty(T)
            t0 = 0
            while t0 < T:
                seg_len = int(gen_rng.integers(change_interval[0], change_interval[1] + 1))
                level = gen_rng.uniform(k_range[0], k_range[1])
                base[t0 : t0 + seg_len] = level
                t0 += seg_len
            # Sinusoidal drift
            period = gen_rng.uniform(drift_period_range[0], drift_period_range[1])
            phase = gen_rng.uniform(0.0, 2.0 * np.pi)
            drift = drift_amplitude * np.sin(2.0 * np.pi * np.arange(T) / period + phase)
            k_seq[:, i] = np.clip(base + drift, k_min, None)

        self.k_seq: NDArray[np.float64] = k_seq
        self.m_seq: NDArray[np.float64] = gen_rng.beta(k_seq, 1.0)  # (T, N)

    def reset(self, rng: np.random.Generator) -> None:
        """Reset the round counter only.

        The competing-bid sequence is fixed at construction; `rng` is
        accepted for runner-protocol compatibility but intentionally unused
        (a non-stochastic sequence must not change across trials).
        """
        self._t = 0

    def _sample_competing_bids(self) -> NDArray[np.float64]:
        return self.m_seq[min(self._t, self.T - 1)]


class SlightlyNonStationaryMultiCampaignEnv(BiddingEnvironment):
    """Piecewise-stationary multi-campaign environment for Phase 4.

    The horizon is partitioned into `n_intervals` equal intervals. Within
    interval j, campaign i's competing bid is drawn i.i.d. from
    Beta(k_matrix[j, i], 1); each interval has a different parameter vector.
    This is the "slightly non-stationary" regime: a handful of persistent
    regime shifts (Upsilon_T = n_intervals - 1 change points), in contrast
    to the ~hundreds of fast changes of NonStationaryMultiCampaignEnv.

    The interval STRUCTURE (boundaries and k_matrix) is generated once at
    construction from `sequence_seed` and shared by every trial, but the
    competing bids themselves are RESAMPLED per trial from the trial rng
    (course notebook 10 protocol for piecewise-stationary stochastic
    environments). Pseudo-regret is therefore well defined against the
    expected-value per-interval clairvoyant (`clairvoyant_piecewise`).

    Consecutive intervals are guaranteed to differ in every campaign's
    parameter (redrawn on collision), so each boundary is a genuine change
    point for every campaign.

    Parameters
    ----------
    values : (N,) array
        Per-campaign private values v_i (fixed over time).
    T : int
        Horizon.
    sequence_seed : int
        Seed for the interval structure. Same seed => identical k_matrix and
        boundaries, so baselines are computed once and trials are comparable.
    n_intervals : int, default 5
        Number of stationary intervals (equal length, last one absorbs the
        remainder of T).
    k_levels : tuple[int, int], default (1, 6)
        Inclusive integer range from which each k_matrix entry is drawn
        (number of Uniform(0,1) competitors; higher = stiffer competition).

    Attributes
    ----------
    k_matrix : (n_intervals, N) array
        Beta parameter of each (interval, campaign) pair.
    boundaries : (n_intervals + 1,) array of int
        Interval edges; interval j covers rounds [boundaries[j], boundaries[j+1]).
    k_seq : (T, N) array
        Per-round Beta parameters (k_matrix expanded along the horizon).
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        T: int,
        sequence_seed: int,
        n_intervals: int = 5,
        k_levels: tuple[int, int] = (1, 6),
    ) -> None:
        gen_rng = np.random.default_rng(sequence_seed)
        super().__init__(values=np.asarray(values, dtype=float), T=T, rng=gen_rng)

        self.n_intervals = int(n_intervals)
        self.boundaries: NDArray[np.int_] = np.linspace(
            0, T, self.n_intervals + 1, dtype=int
        )

        k_matrix = np.zeros((self.n_intervals, self.N))
        for j in range(self.n_intervals):
            for i in range(self.N):
                k = int(gen_rng.integers(k_levels[0], k_levels[1] + 1))
                # Every boundary must be a real change point for campaign i
                while j > 0 and k == k_matrix[j - 1, i]:
                    k = int(gen_rng.integers(k_levels[0], k_levels[1] + 1))
                k_matrix[j, i] = k
        self.k_matrix: NDArray[np.float64] = k_matrix

        self.k_seq: NDArray[np.float64] = np.repeat(
            k_matrix,
            np.diff(self.boundaries),
            axis=0,
        )  # (T, N)

    def reset(self, rng: np.random.Generator) -> None:
        """Store the trial rng; competing bids are resampled per trial."""
        self.rng = rng
        self._t = 0

    def _sample_competing_bids(self) -> NDArray[np.float64]:
        k_row = self.k_seq[min(self._t, self.T - 1)]
        return self.rng.beta(k_row, np.ones(self.N))
