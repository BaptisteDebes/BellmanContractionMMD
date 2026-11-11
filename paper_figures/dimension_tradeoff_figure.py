"""Draw the figure of the tradeoff experiments in higher dimensions (Appendix H) from its archive.

For the power-distance kernels −‖x − y‖^α, whose smallest contraction factor is γ^α, each
panel is the heatmap of Algorithm 4 for one distribution pair and one dimension d. The style
and the fonts are those of Figures 3 and 17, and the figure has their width, so that its text
has their size in the paper. This script only draws.

Usage:
    python paper_figures/dimension_tradeoff_figure.py --archive <dimension_tradeoff.npz> \
        --output figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import convert_to_bold_mathtext
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from tradeoff_figures import TRADEOFF_RC, draw_tradeoff_heatmap, save_figure

PATTERN_TITLES = {  # pattern name in the archive → row title, in row order
    "parity_checkerboard": "Cube\nCheckerboard",
    "gaussian_mean_shift": "Mean Shift",
}
OUTPUT_NAME = "empirical_envelope__dimension_generalization.pdf"


def plot_dimension_tradeoff(archive: Path) -> Figure:
    """Draw one row per distribution pair and one column per dimension, with one color bar.

    The figure is as wide as Figures 3 and 17, so that at the same width in the paper its
    text has the same size.

    Args:
        archive: The archive `dimension_tradeoff.npz`.

    Returns:
        The figure.
    """
    with np.load(archive, allow_pickle=False) as arrays:
        dimensions = [int(d) for d in arrays["dimensions"]]
        discounts = np.asarray(arrays["discounts"], dtype=np.float64)
        budgets = np.asarray(arrays["budgets"], dtype=np.float64)
        # Heatmaps per pattern, shape (dimensions, budgets, discounts)
        heatmaps = {
            pattern: np.asarray(arrays[f"{pattern}__heatmaps"], dtype=np.float64)
            for pattern in PATTERN_TITLES
        }
    figure = plt.figure(figsize=(13.8, 6.2), layout="constrained")
    # One row of axes per pattern, one column per dimension
    axes: list[list[Axes]] = [
        list(row) for row in figure.subplots(len(PATTERN_TITLES), len(dimensions), squeeze=False)
    ]
    meshes: list[QuadMesh] = []
    for row, (pattern, title) in enumerate(PATTERN_TITLES.items()):
        for column, dimension in enumerate(dimensions):
            panel = axes[row][column]
            meshes.append(
                draw_tradeoff_heatmap(panel, heatmaps[pattern][column], discounts, budgets)
            )
            if row == 0:
                panel.set_title(f"$d={dimension}$")
            panel.set_xlabel(
                r"Discount" "\n" r"factor $\gamma$" if row == len(PATTERN_TITLES) - 1 else ""
            )
            panel.set_ylabel(
                f"{convert_to_bold_mathtext(title)}\n" + r"Contraction" "\n" r"Budget $\bar{\tau}$"
                if column == 0
                else ""
            )
    color_bar = figure.colorbar(
        meshes[-1],
        ax=[panel for row_axes in axes for panel in row_axes],
        fraction=0.028,
        pad=0.01,
    )
    color_bar.set_label(r"Best power under $\tau^\star \leq \bar{\tau}$")
    color_bar.ax.spines["outline"].set_visible(False)
    # One legend for the γ² curve, centered below the panels
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", frameon=False)
    return figure


def main() -> None:
    """Read the archive and write the figure as a PDF."""
    parser = argparse.ArgumentParser(description="Draw the higher-dimension tradeoff figure.")
    parser.add_argument("--archive", type=Path, required=True, help="dimension_tradeoff.npz")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figure")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(TRADEOFF_RC):
        save_figure(plot_dimension_tradeoff(arguments.archive), arguments.output / OUTPUT_NAME)


if __name__ == "__main__":
    main()
