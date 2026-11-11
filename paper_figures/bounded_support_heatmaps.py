"""Draw Figure 2 of the paper, contraction thresholds on a ball, from the saved heatmap archives.

Each archive holds, for every discount γ and every ratio R/ℓ, the largest contraction
threshold τ⋆_S(γ) (Proposition 5.2) over the sampled finite supports S ⊆ B(0, R). This
script only draws them; it computes no threshold.

Usage:
    python paper_figures/bounded_support_heatmaps.py --archives <folder> --output figures
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import TEXT_COLOR, build_cell_edges, style_axes
from jaxtyping import Float
from matplotlib.figure import Figure
from tradeoff_figures import TRADEOFF_RC


@dataclass(frozen=True)
class HeatmapArchive:
    """One saved heatmap: the largest contraction threshold over the sampled supports."""

    discounts: Float[np.ndarray, " discounts"]
    """The discounts γ, one per column."""
    ratios: Float[np.ndarray, " ratios"]
    """The ratios R/ℓ, one per row, with ℓ as in the paper's kernel formulas."""
    thresholds: Float[np.ndarray, "ratios discounts"]
    """The largest contraction threshold τ⋆_S(γ) over the sampled supports."""
    rejection_tolerance: float
    """A threshold above 1 + rejection_tolerance counts as exceeding 1."""


@dataclass(frozen=True)
class HeatmapPanel:
    """One panel of Figure 2: which archive to read, how to label it, where to write it."""

    archive_name: str
    title: str
    output_name: str


# Both archives hold R/ℓ with the paper's ℓ: exp(−‖x − y‖²/ℓ²) and −log(1 + ‖x − y‖²/ℓ²).
FIGURE_2_PANELS = (
    HeatmapPanel(
        archive_name="rbf_fixed_support_heatmap_data.npz",
        title="RBF kernel",
        output_name="rbf_fixed_support_tau_heatmap.pdf",
    ),
    HeatmapPanel(
        archive_name="log_fixed_support_heatmap_data.npz",
        title="Logarithmic kernel",
        output_name="log_fixed_support_tau_heatmap.pdf",
    ),
)


def load_heatmap_archive(path: Path) -> HeatmapArchive:
    """Read one saved heatmap archive.

    Args:
        path: An `.npz` archive with the arrays `lambda_values` (the discounts),
            `ratio_values`, `tau_grid` and the scalar `rejection_tol`.

    Returns:
        The discounts, shape (discounts,), the ratios R/ℓ, shape (ratios,), the
        thresholds, shape (ratios, discounts), and the rejection tolerance.

    Raises:
        ValueError: If the threshold grid does not have one row per ratio and one
            column per discount.
    """
    with np.load(path, allow_pickle=False) as archive:
        discounts: Float[np.ndarray, " discounts"] = np.asarray(
            archive["lambda_values"], dtype=np.float64
        )
        ratios: Float[np.ndarray, " ratios"] = np.asarray(archive["ratio_values"], dtype=np.float64)
        thresholds: Float[np.ndarray, "ratios discounts"] = np.asarray(
            archive["tau_grid"], dtype=np.float64
        )
        rejection_tolerance = float(archive["rejection_tol"])
    if thresholds.shape != (ratios.size, discounts.size):
        raise ValueError(
            f"{path}: threshold grid of shape {thresholds.shape}, expected "
            f"({ratios.size}, {discounts.size})"
        )
    return HeatmapArchive(discounts, ratios, thresholds, rejection_tolerance)


def plot_contraction_thresholds(archive: HeatmapArchive, *, title: str) -> Figure:
    """Draw one panel of Figure 2.

    The color is the threshold. Where it exceeds 1 + the archive's rejection tolerance,
    the region is hatched and bounded by a black line.

    Args:
        archive: The saved heatmap.
        title: The panel title.

    Returns:
        The figure, with one heatmap and its color bar.
    """
    ratios = archive.ratios
    exceeds_from = 1.0 + archive.rejection_tolerance
    figure, axes = plt.subplots(figsize=(6.8, 5.6))
    discount_edges = build_cell_edges(archive.discounts)
    ratio_edges = build_cell_edges(ratios)
    image = axes.pcolormesh(
        discount_edges,
        ratio_edges,
        archive.thresholds,
        cmap="viridis",
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )
    if np.any(archive.thresholds > exceeds_from):
        # Hatch every value from exceeds_from up to the largest one; the black line is the
        # boundary of the same region.
        axes.contourf(
            archive.discounts,
            ratios,
            archive.thresholds,
            levels=[exceeds_from, float(np.max(archive.thresholds))],
            colors="none",
            hatches=["//"],
        )
        axes.contour(
            archive.discounts,
            ratios,
            archive.thresholds,
            levels=[exceeds_from],
            colors=[TEXT_COLOR],
            linewidths=1.6,
        )
    axes.set_title(title)
    axes.set_xlabel(r"Discount factor $\gamma$")
    axes.set_ylabel(r"$R / \ell$")
    axes.set_xlim(float(discount_edges[0]), float(discount_edges[-1]))
    axes.set_ylim(float(ratio_edges[0]), float(ratio_edges[-1]))
    axes.set_xticks(np.linspace(0.0, 1.0, 6))
    style_axes(axes)
    axes.grid(False)
    color_bar = figure.colorbar(image, ax=axes, fraction=0.055, pad=0.03)
    color_bar.set_label("contraction threshold")
    figure.tight_layout()
    return figure


def main() -> None:
    """Read both archives and write both panels of Figure 2 as PDFs."""
    parser = argparse.ArgumentParser(description="Draw Figure 2 from the saved heatmap archives.")
    parser.add_argument("--archives", type=Path, required=True, help="folder of the .npz archives")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF panels")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # The style and text sizes of Figures 3 and 17, which are drawn as much larger than in the
    # paper, so that the text has their size on the page
    plt.rcParams.update({**TRADEOFF_RC, "hatch.color": TEXT_COLOR, "hatch.linewidth": 0.9})
    for panel in FIGURE_2_PANELS:
        archive = load_heatmap_archive(arguments.archives / panel.archive_name)
        figure = plot_contraction_thresholds(archive, title=panel.title)
        # No creation date, so the same archive always gives the same file.
        figure.savefig(
            arguments.output / panel.output_name,
            format="pdf",
            dpi=300,
            bbox_inches="tight",
            metadata={"CreationDate": None},
        )
        plt.close(figure)
        hatched_cells = int(np.sum(archive.thresholds > 1.0 + archive.rejection_tolerance))
        print(f"{panel.output_name}: {hatched_cells} cells above 1 + rejection tolerance")


if __name__ == "__main__":
    main()
