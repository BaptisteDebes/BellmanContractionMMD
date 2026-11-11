"""Draw Figure 1 of the paper, examples of Bernstein functions φ, on t ∈ [0, 4].

The four functions are those of the power-distance kernel with α = 1, φ(t) = √t; the
logarithmic kernel, φ(t) = log(1 + t); the multiquadric kernel, φ(t) = √(1 + t) − 1; and the
RBF kernel, φ(t) = 1 − e^(−t), each with scale 1. The legend names them as the text does, with
their φ(t), in the order of the curves at t = 4, from top to bottom. The figure sits next to
Definition 4.1, at 0.43 of the text width. It is drawn 2.5 times larger, with the style and
fonts of Figures 3 and 17, so that on the page its text has their size. This script only draws.

Usage:
    python paper_figures/bernstein_function_examples.py --output figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import style_axes
from matplotlib.figure import Figure
from tradeoff_figures import TRADEOFF_RC, save_figure

# Width of Figure 3 divided by its width in the paper (13.8 in shown at the 5.5 in text width)
DRAWING_SCALE = 13.8 / 5.5
FIGURE_WIDTH = DRAWING_SCALE * 0.43 * 5.5  # inches; 0.43 of the text width in the paper
FIGURE_HEIGHT = DRAWING_SCALE * 1.6  # inches; about the height of Definition 4.1 beside it
OUTPUT_NAME = "main_bernstein_functions.pdf"


def plot_bernstein_function_examples() -> Figure:
    """Draw the four Bernstein functions on t ∈ [0, 4], with one legend.

    Returns:
        The figure.
    """
    # Arguments t, shape (500,)
    arguments = np.linspace(0.0, 4.0, 500)
    examples = (  # (legend label, color, φ(t) on the arguments), top to bottom at t = 4
        (r"power distance, $\sqrt{t}$", "#F28E2B", np.sqrt(arguments)),
        (r"logarithmic, $\log(1+t)$", "#E15759", np.log1p(arguments)),
        (r"multiquadric, $\sqrt{1+t}-1$", "#59A14F", np.sqrt(1.0 + arguments) - 1.0),
        (r"RBF, $1-e^{-t}$", "#4E79A7", 1.0 - np.exp(-arguments)),
    )
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), layout="constrained")
    style_axes(axes)
    for label, color, values in examples:
        axes.plot(arguments, values, color=color, linewidth=3.0, label=label)
    axes.set_xlabel(r"$t$")
    axes.set_ylabel(r"$\phi(t)$")
    axes.set_xlim(0.0, 4.0)
    axes.set_ylim(0.0, 2.5)
    axes.set_xticks(np.arange(0.0, 4.1, 1.0))
    axes.set_yticks(np.arange(0.0, 2.51, 1.0))
    axes.legend(loc="upper left", frameon=False)
    return figure


def main() -> None:
    """Write Figure 1 as a PDF."""
    parser = argparse.ArgumentParser(description="Draw Figure 1, examples of Bernstein functions.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figure")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(TRADEOFF_RC):
        save_figure(plot_bernstein_function_examples(), arguments.output / OUTPUT_NAME)


if __name__ == "__main__":
    main()
