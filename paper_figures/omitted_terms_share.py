"""Draw the two figures of Appendix G.2: the share of the omitted terms of a truncated chain.

For the chained RBF kernel of Appendix G.2, with terms φ_n(t) = c τ^(−n) (1 − exp(−s₀ ρ^n t)),
the share of the terms omitted by truncating at index K is

    R_K^(N)(t) = Σ_{K ≤ n ≤ N} φ_n(t) / Σ_{0 ≤ n ≤ N} φ_n(t).

Each figure shows R_K^(N)(t) for every truncation index K and argument t, and the curve
K_α(t) where it falls to α. The first figure has τ = 1.10 ρ, the second τ = 1.01 ρ. The shares
are computed by the package, with `TruncatedChainedRBFKernel.compute_omitted_shares`.

Usage:
    python paper_figures/omitted_terms_share.py --output figures
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from figure_style import BACKGROUND_COLOR, TEXT_COLOR
from jaxtyping import Float
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from mmd_contraction.kernels import TruncatedChainedRBFKernel

jax.config.update("jax_enable_x64", True)

DISCOUNT = 0.95  # γ
SQUARED_DISCOUNT = DISCOUNT**2  # ρ = γ²
SCALE = 1.0  # s₀
WEIGHT = 1.0  # c
LAST_INDEX = 2500  # N, the last index of the finite sums
LEVEL = 1e-3  # α of the curve K_α(t)
SHARE_FLOOR = 1e-8  # shares below this value are drawn with the color of this value
LEGEND_EDGE_COLOR = "#D9D9D9"


@dataclass(frozen=True)
class ChainFigure:
    """One of the two chains of Appendix G.2 and the file its figure goes to."""

    target: float
    """τ, the contraction factor the chain targets."""
    output_name: str
    """The PDF file name in the output folder."""


FIGURES = (
    ChainFigure(target=1.10 * SQUARED_DISCOUNT, output_name="tail_share_default.pdf"),
    ChainFigure(target=1.01 * SQUARED_DISCOUNT, output_name="tail_share_close.pdf"),
)


def find_level_crossings(
    shares: Float[np.ndarray, "indices arguments"], level: float
) -> Float[np.ndarray, " arguments"]:
    """Find, for each t, where R_K^(N)(t) first falls to the level, between integer K.

    The crossing is placed between the last K above the level and the first K at or below
    it, by linear interpolation of log R; NaN if no K reaches the level.

    Args:
        shares: R_K^(N)(t): shape (N + 1, T).
        level: α.

    Returns:
        The crossing index for each t: shape (T,).
    """
    tiny = np.finfo(np.float64).tiny
    crossings = np.full(shares.shape[1], np.nan)
    for column in range(shares.shape[1]):
        # Indices K with R_K^(N)(t) ≤ α
        reached = np.flatnonzero(shares[:, column] <= level)
        if reached.size == 0:
            continue
        # First index K at or below the level
        first = int(reached[0])
        if first == 0:
            crossings[column] = 0.0
            continue
        # log R at the last index above the level and at the first one at or below it
        above = np.log(max(float(shares[first - 1, column]), tiny))
        below = np.log(max(float(shares[first, column]), tiny))
        # Fraction of the step from K − 1 to K where log R reaches log α
        fraction = 1.0 if above == below else (np.log(level) - above) / (below - above)
        crossings[column] = first - 1 + float(np.clip(fraction, 0.0, 1.0))
    return crossings


def build_log_cell_edges(
    centers: Float[np.ndarray, " cells"],
) -> Float[np.ndarray, " edges"]:
    """Build the cell boundaries of a geometric grid: midpoints in log scale, half steps at ends.

    Args:
        centers: The grid values, increasing and geometric: shape (cells,).

    Returns:
        The boundaries: shape (cells + 1,).
    """
    # Logarithms of the centers, shape (cells,)
    log_centers = np.log(centers)
    # Midpoints between neighbours in log scale, shape (cells − 1,)
    midpoints = 0.5 * (log_centers[1:] + log_centers[:-1])
    # First and last edges, half a step beyond the end centers
    first = log_centers[0] - 0.5 * (log_centers[1] - log_centers[0])
    last = log_centers[-1] + 0.5 * (log_centers[-1] - log_centers[-2])
    # All edges, shape (cells + 1,)
    return np.exp(np.concatenate(([first], midpoints, [last])))


def plot_omitted_share(chain: ChainFigure, arguments: Float[np.ndarray, " arguments"]) -> Figure:
    """Draw one figure: R_K^(N)(t) as a heatmap, the curve K_α(t), and the chain's parameters.

    Args:
        chain: The chain.
        arguments: The arguments t, geometric: shape (T,).

    Returns:
        The figure.
    """
    # The chain with its N + 1 terms n = 0, …, N
    kernel = TruncatedChainedRBFKernel(
        discount=DISCOUNT,
        target_contraction_factor=chain.target,
        scale=SCALE,
        weight=WEIGHT,
        term_count=LAST_INDEX + 1,
    )
    # Shares R_K^(N)(t) at row K and column t, shape (N + 1, T)
    shares: Float[np.ndarray, "indices arguments"] = np.asarray(
        kernel.compute_omitted_shares(jnp.asarray(arguments))
    )
    crossings = find_level_crossings(shares, LEVEL)
    figure, axes = plt.subplots(figsize=(7.2, 4.9))
    mesh = axes.pcolormesh(
        build_log_cell_edges(arguments),
        np.arange(LAST_INDEX + 2) - 0.5,
        np.clip(shares, SHARE_FLOOR, 1.0),
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=SHARE_FLOOR, vmax=1.0),
        rasterized=True,
    )
    reached = np.isfinite(crossings)
    axes.plot(
        arguments[reached],
        crossings[reached],
        color="white",
        linewidth=2.2,
        label=r"$K_{10^{-3}}(t)$",
    )
    axes.legend(loc="upper right")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(TEXT_COLOR)
        axes.spines[side].set_linewidth(1.45)
    axes.set_xscale("log")
    axes.set_xlim(float(arguments[0]), float(arguments[-1]))
    axes.set_ylim(-0.5, LAST_INDEX + 0.5)
    axes.set_xlabel("t")
    axes.set_ylabel("Truncation index $K$")
    axes.set_title("Share of the Omitted Terms", fontweight="bold", pad=10)
    axes.text(
        0.02,
        0.98,
        "\n".join(
            (
                rf"$\rho={SQUARED_DISCOUNT:.6g}$",
                rf"$\tau={chain.target:.6g}$",
                rf"$s_0={SCALE:.6g}$",
                rf"$c={WEIGHT:.6g}$",
                rf"$N={LAST_INDEX}$",
            )
        ),
        transform=axes.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": BACKGROUND_COLOR,
            "edgecolor": LEGEND_EDGE_COLOR,
            "linewidth": 1.0,
            "alpha": 0.96,
        },
    )
    color_bar = figure.colorbar(mesh, ax=axes)
    color_bar.set_label(r"share of the omitted terms $R_K^{(N)}(t)$")
    color_bar.ax.spines["outline"].set_edgecolor(LEGEND_EDGE_COLOR)
    color_bar.ax.spines["outline"].set_linewidth(1.0)
    figure.tight_layout()
    return figure


def main() -> None:
    """Compute and draw both figures as PDFs."""
    parser = argparse.ArgumentParser(description="Draw the two figures of Appendix G.2.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figures")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    # Arguments t, 241 geometric values from 10⁻⁶ to 10⁶
    arguments = np.logspace(-6.0, 6.0, 241)
    style = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "font.size": 9.0,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "figure.facecolor": BACKGROUND_COLOR,
        "axes.facecolor": BACKGROUND_COLOR,
    }
    with plt.rc_context(style):
        for chain in FIGURES:
            figure = plot_omitted_share(chain, arguments)
            # No creation date, so the same settings always give the same file
            figure.savefig(
                output / chain.output_name,
                format="pdf",
                dpi=400,
                bbox_inches="tight",
                metadata={"CreationDate": None},
            )
            plt.close(figure)


if __name__ == "__main__":
    main()
