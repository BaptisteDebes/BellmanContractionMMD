"""Draw Figure 4 of the paper, estimates on balls for kernels with known factors, from its archive.

Each panel is one kernel of experiments/bounded_support_known_factors.py: the estimate τ̂⋆ on
the ball B(0, R) against the discount λ, one curve per radius R, and the smallest contraction
factor across dimensions as a dashed curve, λ^(2β) for the generalized multiquadric and λ^α
for the power distance. The figure has the width, style and fonts of Figures 3 and 17, so that
at the text width its text has their size. This script only draws.

Usage:
    python paper_figures/bounded_support_known_factors_figure.py \
        --archive <folder>/bounded_support_known_factors.npz --output figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import TEXT_COLOR, style_axes
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tradeoff_figures import TRADEOFF_RC, save_figure

FAMILY_TITLES = {  # family in the archive → (panel title, name of its exponent)
    "gmq": ("GMQ", r"\beta"),
    "power_distance": ("Power Distance", r"\alpha"),
}
THEORY_COLOR = "#BAB0AC"
OUTPUT_NAME = "bounded_support_theoretical_confirmation.pdf"


def compute_known_factor_power(family: str, exponent: float) -> float:
    """Compute the power p of the smallest contraction factor across dimensions, τ⋆(λ) = λ^p.

    Args:
        family: "gmq" or "power_distance".
        exponent: β or α.

    Returns:
        2β for the generalized multiquadric, α for the power distance.
    """
    return 2.0 * exponent if family == "gmq" else exponent


def plot_known_factor_estimates(archive: Path) -> Figure:
    """Draw one panel per kernel, two rows of three, with one legend below them.

    Args:
        archive: The archive `bounded_support_known_factors.npz`.

    Returns:
        The figure.
    """
    with np.load(archive, allow_pickle=False) as arrays:
        families = [str(family) for family in arrays["families"]]
        exponents = [float(exponent) for exponent in arrays["exponents"]]
        radii: Float[np.ndarray, " radii"] = np.asarray(arrays["radii"], dtype=np.float64)
        discounts: Float[np.ndarray, " discounts"] = np.asarray(
            arrays["discounts"], dtype=np.float64
        )
        # τ̂⋆ for each kernel, radius and discount, shape (kernels, radii, discounts)
        thresholds: Float[np.ndarray, "kernels radii discounts"] = np.asarray(
            arrays["thresholds"], dtype=np.float64
        )
    # One color per radius, from dark (small R) to light (large R)
    colors = [plt.get_cmap("viridis")(level) for level in np.linspace(0.18, 0.92, radii.size)]
    figure = plt.figure(figsize=(13.8, 8.1), layout="constrained")
    # The six panels, row by row
    panels: list[Axes] = list(figure.subplots(2, 3, sharex=True, sharey=True, squeeze=False).flat)
    for kernel_index, (panel, family, exponent) in enumerate(
        zip(panels, families, exponents, strict=True)
    ):
        style_axes(panel)
        # Power p of the known factor τ⋆(λ) = λ^p
        power = compute_known_factor_power(family, exponent)
        for color, curve in zip(colors, thresholds[kernel_index], strict=True):
            panel.plot(discounts, curve, color=color, linewidth=3.6, alpha=0.98)
        panel.plot(
            discounts,
            discounts**power,
            color=THEORY_COLOR,
            linewidth=3.2,
            linestyle="--",
            dashes=(6, 3),
        )
        title, exponent_name = FAMILY_TITLES[family]
        panel.set_title(rf"{title}: ${exponent_name}={exponent:.2f}$")
        panel.text(
            0.03,
            0.90,
            rf"$\tau^\star(\lambda)=\lambda^{{{power:.2f}}}$",
            transform=panel.transAxes,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": THEORY_COLOR,
                "linewidth": 1.4,
            },
        )
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.05)
        panel.set_xticks(np.linspace(0.0, 1.0, 6))
        panel.set_xlabel(r"$\lambda$")
        if kernel_index % 3 == 0:
            panel.set_ylabel(r"$\widehat{\tau}^\star$")
    handles = [
        Line2D([0], [0], color=color, linewidth=3.6, label=rf"$R={radius:g}$")
        for color, radius in zip(colors, radii, strict=True)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=THEORY_COLOR,
            linewidth=3.2,
            linestyle="--",
            label=r"$\tau^\star(\lambda)$",
        )
    )
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    return figure


def main() -> None:
    """Read the archive and write Figure 4 as a PDF."""
    parser = argparse.ArgumentParser(description="Draw Figure 4, estimates on balls.")
    parser.add_argument(
        "--archive", type=Path, required=True, help="bounded_support_known_factors.npz"
    )
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figure")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({**TRADEOFF_RC, "text.color": TEXT_COLOR}):
        save_figure(plot_known_factor_estimates(arguments.archive), arguments.output / OUTPUT_NAME)


if __name__ == "__main__":
    main()
