"""
src/utils/plotting.py
----------------------
Shared plotting helpers for regret curves and other experiment figures.
All functions return the Axes object so callers can overlay multiple curves
or add custom annotations.

Usage
-----
    from src.utils.plotting import plot_regret, new_figure, save_figure

    fig, ax = new_figure("Cumulative Regret — Req 1")
    plot_regret(ax, mean, std, label="UCB1 (no budget)")
    plot_regret(ax, mean2, std2, label="UCB1 + budget")
    save_figure(fig, "report/figures/req1_regret.png")
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ── Shared style ────────────────────────────────────────────────────────────

# Colour cycle: accessible, distinct, works in greyscale print
_COLOURS = [
    "#2166ac",  # blue
    "#d6604d",  # red-orange
    "#4dac26",  # green
    "#8073ac",  # purple
    "#f4a582",  # light salmon (for a 5th curve if needed)
]

_ALPHA_BAND = 0.20   # transparency of the ±1 std shading


def _apply_shared_style(ax: Axes, title: str) -> None:
    """Apply project-wide axis style."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Round $t$", fontsize=11)
    ax.set_ylabel("Cumulative regret", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Public API ───────────────────────────────────────────────────────────────

def new_figure(title: str = "", figsize: tuple[float, float] = (8, 5)) -> tuple[Figure, Axes]:
    """
    Create a fresh (fig, ax) pair with the shared style applied.

    Parameters
    ----------
    title : str
        Figure / axes title.
    figsize : tuple
        Matplotlib figsize.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(color=_COLOURS)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Round $t$", fontsize=11)
    ax.set_ylabel("Cumulative regret", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_regret(
    ax: Axes,
    mean: np.ndarray,
    std: np.ndarray,
    n_trials: int,
    label: str = "",
    colour: str | None = None,
) -> Axes:
    """
    Plot a mean regret curve with a shaded ± std/sqrt(n_trials) band.

    The band shows the standard error of the mean (SEM), matching the
    convention used throughout the course notebooks (01, 02, etc.):
        fill_between(mean - std/sqrt(n_trials), mean + std/sqrt(n_trials))

    Parameters
    ----------
    ax : Axes
        Target axes (from new_figure or any plt.subplots call).
    mean : np.ndarray, shape (T,)
        Mean cumulative regret over trials.
    std : np.ndarray, shape (T,)
        Std of cumulative regret over trials.
    n_trials : int
        Number of trials used to compute mean and std. Used to derive SEM.
    label : str
        Legend label.
    colour : str or None
        Hex / named colour. If None, uses the axes' current colour cycle.

    Returns
    -------
    ax (for chaining)
    """
    # adapted from notebook 01/02 convention: uncertainty = std / sqrt(n_trials)
    sem = std / np.sqrt(n_trials)
    rounds = np.arange(1, len(mean) + 1)
    line, = ax.plot(rounds, mean, label=label, color=colour, linewidth=1.8)
    c = line.get_color()
    ax.fill_between(
        rounds,
        mean - sem,
        mean + sem,
        alpha=_ALPHA_BAND,
        color=c,
    )
    ax.legend(fontsize=10, framealpha=0.9)
    return ax


def plot_budget_consumption(
    ax: Axes,
    cumulative_costs: np.ndarray,
    budget_total: float,
    label: str = "",
    colour: str | None = None,
) -> Axes:
    """
    Plot cumulative cost over time with a horizontal budget ceiling.

    Parameters
    ----------
    ax : Axes
    cumulative_costs : np.ndarray, shape (T,)
        Cumulative spend up to each round (single trial or mean over trials).
    budget_total : float
        Total budget B_total — drawn as a dashed red line.
    label : str
    colour : str or None

    Returns
    -------
    ax
    """
    rounds = np.arange(1, len(cumulative_costs) + 1)
    ax.plot(rounds, cumulative_costs, label=label, color=colour, linewidth=1.8)
    ax.axhline(budget_total, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Budget = {budget_total:.2f}")
    ax.set_xlabel("Round $t$", fontsize=11)
    ax.set_ylabel("Cumulative cost", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save_figure(fig: Figure, path: str, dpi: int = 150) -> None:
    """
    Save a figure to disk (PNG or PDF depending on extension).

    Parameters
    ----------
    fig : Figure
    path : str
        e.g. "report/figures/req1_regret.png"
    dpi : int
    """
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path}")
