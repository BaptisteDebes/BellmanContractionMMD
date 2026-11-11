"""Draw Figure 5 of the paper: a single geometric chain of atoms and its Bernstein function.

For λ² = 0.7, τ = 0.9, s₀ = 0.2 and c = 1, the left panel shows the atoms s_n = λ^(2n) s₀ with
weights a_n = c τ^(−n) for n = 0, …, 25, on logarithmic axes. The right panel shows the
Bernstein function φ_chain(t) = Σ_{n=0}^{250} c τ^(−n) (1 − e^(−s_n t)) on t ∈ [0, 1000], which
is minus the truncated chained RBF kernel with discount λ and scale s₀. The figure spans the
text width. It has the style and fonts of Figures 3 and 17 but is drawn 12 in wide, the width of
the image it replaces, so that its text keeps that image's size on the page, about one point
larger than in Figures 3 and 17. This script only draws.

Usage:
    python paper_figures/single_chain_figure.py --output figures
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

SQUARED_DISCOUNT = 0.7  # λ²
TARGET = 0.9  # τ
FIRST_LOCATION = 0.2  # s₀
WEIGHT = 1.0  # c
LAST_SHOWN_INDEX = 25  # the left panel shows atoms n = 0, …, 25
LAST_TERM_INDEX = 250  # the right panel sums terms n = 0, …, 250
COLOR = "#4E79A7"
FIGURE_SIZE = (12.0, 4.0)  # inches, the size of the image it replaces
OUTPUT_NAME = "single_chain.pdf"


def plot_single_chain() -> Figure:
    """Draw the atoms of the chain (left) and its Bernstein function (right).

    Returns:
        The figure.
    """
    # Atom indices n, shape (26,)
    indices: Int[np.ndarray, " atoms"] = np.arange(LAST_SHOWN_INDEX + 1)
    # Atom locations s_n = λ^(2n) s₀, shape (26,)
    locations: Float[np.ndarray, " atoms"] = FIRST_LOCATION * SQUARED_DISCOUNT**indices
    # Atom weights a_n = c τ^(−n), shape (26,)
    weights: Float[np.ndarray, " atoms"] = WEIGHT * TARGET ** (-indices)
    # Arguments t, shape (1200,)
    arguments: Float[np.ndarray, " arguments"] = np.linspace(0.0, 1000.0, 1200)
    kernel = TruncatedChainedRBFKernel(
        discount=SQUARED_DISCOUNT**0.5,
        target_contraction_factor=TARGET,
        scale=FIRST_LOCATION,
        weight=WEIGHT,
        term_count=LAST_TERM_INDEX + 1,
    )
    # φ_chain(t) = −k(t), shape (1200,)
    values: Float[np.ndarray, " arguments"] = -np.asarray(
        kernel.evaluate_on_squared_distances(jnp.asarray(arguments))
    )
    figure = plt.figure(figsize=FIGURE_SIZE, layout="constrained")
    # The two panels, left to right
    atom_axes, function_axes = list(figure.subplots(1, 2, squeeze=False)[0])
    style_axes(atom_axes)
    # Bottom of the stems, below the smallest weight
    floor = float(np.min(weights)) / 1.6
    atom_axes.vlines(locations, floor, weights, color=COLOR, linewidth=2.2, alpha=0.45)
    atom_axes.scatter(
        locations, weights, s=60, facecolors="white", edgecolors=COLOR, linewidths=2.0, zorder=3
    )
    atom_axes.set_xscale("log")
    atom_axes.set_yscale("log")
    atom_axes.set_xlabel(r"atom locations $s_n=\lambda^{2n} s_0$")
    atom_axes.set_ylabel(r"atom weights $a_n=c\,\tau^{-n}$")
    atom_axes.set_title(
        "Single geometric chain of atoms\n"
        rf"$\lambda^2={SQUARED_DISCOUNT}$, $\tau={TARGET}$, $s_0={FIRST_LOCATION}$, $c={WEIGHT:g}$"
    )
    style_axes(function_axes)
    function_axes.plot(arguments, values, color=COLOR, linewidth=3.0)
    function_axes.set_xlabel(r"$t$")
    function_axes.set_ylabel(r"$\phi_{\mathrm{chain}}(t)$")
    function_axes.set_title("Bernstein function from the chain")
    return figure


def main() -> None:
    """Write Figure 5 as a PDF."""
    parser = argparse.ArgumentParser(description="Draw Figure 5, a geometric chain of atoms.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figure")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(TRADEOFF_RC):
        save_figure(plot_single_chain(), arguments.output / OUTPUT_NAME)


if __name__ == "__main__":
    main()
