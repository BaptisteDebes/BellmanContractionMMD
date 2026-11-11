"""Draw Figures 12 and 13 of the paper: the chained RBF kernel for three target factors τ.

For γ = 0.90, a = 0.20, c = 1 and τ ∈ {0.84, 0.90, 0.96}, Figure 12 shows the Bernstein
functions φ_τ(t) = Σ_{n=0}^{K} c τ^(−n) (1 − e^(−a γ^(2n) t)) with K = 700 on t ∈ [0, 500], and
Figure 13 the first 35 atoms (s_n, c τ^(−n)) of their Lévy measures, s_n = a γ^(2n). Each
figure sits at 0.48 of the text width. It is drawn 13.8/5.5 times larger, with the style and
fonts of Figures 3 and 17, so that on the page its text has their size. This script only draws.

Usage:
    python paper_figures/chained_rbf_kernel_figures.py --output figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from figure_style import style_axes
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from tradeoff_figures import TRADEOFF_RC, save_figure

from mmd_contraction.kernels import TruncatedChainedRBFKernel

jax.config.update("jax_enable_x64", True)

DISCOUNT = 0.90  # γ
SCALE = 0.20  # a
WEIGHT = 1.0  # c
TARGETS = (0.84, 0.90, 0.96)  # τ, one curve each
LAST_TERM_INDEX = 700  # K, the last index of the sums of Figure 12
SHOWN_ATOM_COUNT = 35  # atoms n = 0, …, 34 in Figure 13
COLORS = ("#4E79A7", "#59A14F", "#E15759")  # one per τ
# Width of Figure 3 divided by its width in the paper (13.8 in shown at the 5.5 in text width)
DRAWING_SCALE = 13.8 / 5.5
FIGURE_SIZE = (DRAWING_SCALE * 0.48 * 5.5, DRAWING_SCALE * 0.48 * 5.5 * 4.2 / 5.2)  # inches
BERNSTEIN_FILE_NAME = "gaussian_chain_tau_bernstein.pdf"
MEASURE_FILE_NAME = "gaussian_chain_tau_measures.pdf"


def plot_bernstein_functions() -> Figure:
    """Draw φ_τ for the three targets τ on t ∈ [0, 500] (Figure 12).

    φ_τ(t) is minus the truncated chained RBF kernel at squared distance t, with K + 1 terms.

    Returns:
        The figure.
    """
    # Arguments t, shape (1800,)
    arguments: Float[np.ndarray, " arguments"] = np.linspace(0.0, 500.0, 1800)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    style_axes(axes)
    for target, color in zip(TARGETS, COLORS, strict=True):
        kernel = TruncatedChainedRBFKernel(
            discount=DISCOUNT,
            target_contraction_factor=target,
            scale=SCALE,
            weight=WEIGHT,
            term_count=LAST_TERM_INDEX + 1,
        )
        # φ_τ(t) = −k(t), shape (1800,)
        values: Float[np.ndarray, " arguments"] = -np.asarray(
            kernel.evaluate_on_squared_distances(jnp.asarray(arguments))
        )
        axes.plot(arguments, values, color=color, linewidth=3.0, label=rf"$\tau={target:.2f}$")
    axes.set_xlabel(r"$t$")
    axes.set_ylabel(r"$\phi_\tau(t)$")
    axes.legend(loc="upper left", frameon=False)
    return figure


def plot_levy_measures() -> Figure:
    """Draw the first atoms (s_n, c τ^(−n)) for the three targets τ, on log axes (Figure 13).

    Returns:
        The figure.
    """
    # Atom indices n, shape (atoms,)
    indices: Int[np.ndarray, " atoms"] = np.arange(SHOWN_ATOM_COUNT)
    # Atom locations s_n = a γ^(2n), shape (atoms,)
    locations: Float[np.ndarray, " atoms"] = SCALE * DISCOUNT ** (2 * indices)
    # Atom masses c τ^(−n) for each τ, shape (targets, atoms)
    masses: Float[np.ndarray, "targets atoms"] = np.asarray(
        [WEIGHT * target ** (-indices) for target in TARGETS]
    )
    # Bottom of the stems, below the smallest mass
    floor = float(np.min(masses)) / 1.6
    figure, axes = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    style_axes(axes)
    for target, color, target_masses in zip(TARGETS, COLORS, masses, strict=True):
        axes.vlines(locations, floor, target_masses, color=color, linewidth=2.2, alpha=0.45)
        axes.scatter(
            locations,
            target_masses,
            s=60,
            facecolors="white",
            edgecolors=color,
            linewidths=2.0,
            label=rf"$\tau={target:.2f}$",
            zorder=3,
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel(r"$s_n = a\gamma^{2n}$")
    axes.set_ylabel(r"$c\,\tau^{-n}$")
    axes.legend(loc="upper right", frameon=False)
    return figure


def main() -> None:
    """Write Figures 12 and 13 as PDFs."""
    parser = argparse.ArgumentParser(description="Draw Figures 12 and 13, the chained RBF kernel.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figures")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(TRADEOFF_RC):
        save_figure(plot_bernstein_functions(), arguments.output / BERNSTEIN_FILE_NAME)
        save_figure(plot_levy_measures(), arguments.output / MEASURE_FILE_NAME)


if __name__ == "__main__":
    main()
