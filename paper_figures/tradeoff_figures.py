"""Draw Figures 3, 16 and 17 of the paper, the tradeoff experiments, from the saved archives.

Figures 3 and 17 show, for each task of Table 12 and each kernel family of Table 11, the
heatmap of Algorithm 4: the best power among the family's kernels whose contraction
threshold fits the budget τ̄ at discount γ. Figure 3 has the three 1D tasks, Figure 17 the
four 2D tasks. Figure 16 shows one permutation test on a checkerboard dataset. This script
only draws; it computes no threshold and no power.

Usage:
    python paper_figures/tradeoff_figures.py --archives <folder> --output figures
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import (
    BACKGROUND_COLOR,
    TEXT_COLOR,
    build_cell_edges,
    convert_to_bold_mathtext,
    style_axes,
)
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mmd_contraction import tradeoff_tasks
from mmd_contraction.tradeoff_tasks import GaussianMixture

P_COLOR = "#4E79A7"
Q_COLOR = "#E15759"
RADIUS_COLOR = "#BAB0AC"

FAMILY_TITLES = {  # archive name → column title, in column order
    "rbf": "RBF",
    "log": "Log\nKernel",
    "gmq": "Generalized\nMQ",
    "power_distance": "Power\nDistance",
}
TASK_TITLES = {  # task name in the archives → row title
    "MEAN_SHIFT_1D": "1D Mean Shift",
    "SAME_MEAN_DIFFERENT_STD_1D": "1D Same Mean,\nDifferent Std",
    "BIMODAL_VS_UNIMODAL_1D": "1D Bimodal\nvs Unimodal",
    "TWO_GAUSSIAN_LOCATION": "Two-Gaussian\nLocation",
    "COVARIANCE_SWAP": "Covariance\nSwap",
    "BIMODAL_VS_UNIMODAL": "Bimodal vs\nUnimodal",
    "CHECKERBOARD": "Checkerboard",
}
FIGURE_3_TASKS = ("MEAN_SHIFT_1D", "SAME_MEAN_DIFFERENT_STD_1D", "BIMODAL_VS_UNIMODAL_1D")
FIGURE_17_TASKS = (
    "TWO_GAUSSIAN_LOCATION",
    "COVARIANCE_SWAP",
    "BIMODAL_VS_UNIMODAL",
    "CHECKERBOARD",
)
# Figures 3 and 17 are drawn 13.8 in wide and shown at the text width, about 0.4 times smaller:
# these sizes give, on the page, the ticks and labels of Figure 16 (5.4 pt and 6.6 pt), with
# titles as large as the labels
TRADEOFF_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 16.5,
    "axes.titlesize": 16.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 16.5,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 13.5,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "figure.facecolor": BACKGROUND_COLOR,
    "axes.facecolor": BACKGROUND_COLOR,
    "pdf.fonttype": 42,
}
NULL_RC = {
    **TRADEOFF_RC,
    "font.size": 10.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
}


@dataclass(frozen=True)
class TradeoffTask:
    """One task of Table 12 as the archives store it."""

    name: str
    """The task's name in the archives and in `mmd_contraction.tradeoff_tasks`."""
    radius: float
    """R, the radius of the ball B(0, R) the samples are moved onto."""
    first_sample: Float[np.ndarray, "first dimension"]
    """One example sample from P, moved onto the ball."""
    second_sample: Float[np.ndarray, "second dimension"]
    """One example sample from Q, moved onto the ball."""
    heatmaps: dict[str, Float[np.ndarray, "budgets discounts"]]
    """Per family, the best power under τ̂ ≤ τ̄ at each budget and discount, NaN if none fits."""


@dataclass(frozen=True)
class TradeoffArchives:
    """The tradeoff data of all families and tasks."""

    discounts: Float[np.ndarray, " discounts"]
    """The discounts γ, one per heatmap column."""
    budgets: Float[np.ndarray, " budgets"]
    """The contraction budgets τ̄, one per heatmap row."""
    tasks: dict[str, TradeoffTask]
    """The tasks, by name."""


@dataclass(frozen=True)
class PermutationNull:
    """The single permutation test of Figure 16."""

    first_sample: Float[np.ndarray, "first dimension"]
    """The sample from P."""
    second_sample: Float[np.ndarray, "second dimension"]
    """The sample from Q."""
    statistic: float
    """T, the observed statistic."""
    null_statistics: Float[np.ndarray, " permutations"]
    """T_π, one per shuffle of the labels."""
    level: float
    """α, the level of the test."""


def load_tradeoff_archives(folder: Path) -> TradeoffArchives:
    """Read the four family archives `tradeoff__{family}.npz`.

    Args:
        folder: The folder of the archives.

    Returns:
        The discounts, the budgets and every task with one heatmap per family.

    Raises:
        ValueError: If the archives disagree on the discounts, budgets or tasks.
    """
    discounts, budgets = None, None
    heatmaps: dict[str, dict[str, Float[np.ndarray, "budgets discounts"]]] = {}
    samples: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for family in FAMILY_TITLES:
        with np.load(folder / f"tradeoff__{family}.npz", allow_pickle=False) as archive:
            if discounts is None or budgets is None:
                discounts = np.asarray(archive["discounts"], dtype=np.float64)
                budgets = np.asarray(archive["budgets"], dtype=np.float64)
            elif not (
                np.array_equal(discounts, archive["discounts"])
                and np.array_equal(budgets, archive["budgets"])
            ):
                raise ValueError(f"{family}: discounts or budgets differ from the other archives")
            for name in (str(task) for task in archive["task_order"]):
                prefix = name.lower()
                heatmaps.setdefault(name, {})[family] = np.asarray(
                    archive[f"{prefix}__heatmap"], dtype=np.float64
                )
                # The example samples are the same in every archive; keep the first
                samples.setdefault(
                    name,
                    (
                        np.asarray(archive[f"{prefix}__example_first_sample"], dtype=np.float64),
                        np.asarray(archive[f"{prefix}__example_second_sample"], dtype=np.float64),
                        float(archive[f"{prefix}__radius"]),
                    ),
                )
    assert discounts is not None and budgets is not None
    tasks = {
        name: TradeoffTask(name, samples[name][2], samples[name][0], samples[name][1], families)
        for name, families in heatmaps.items()
    }
    for name, task in tasks.items():
        if set(task.heatmaps) != set(FAMILY_TITLES):
            raise ValueError(f"{name}: missing in some archives")
    return TradeoffArchives(discounts, budgets, tasks)


def load_permutation_null(path: Path) -> PermutationNull:
    """Read the Figure 16 archive `figure_14_permutation_null.npz`."""
    with np.load(path, allow_pickle=False) as archive:
        return PermutationNull(
            np.asarray(archive["first_sample"], dtype=np.float64),
            np.asarray(archive["second_sample"], dtype=np.float64),
            float(archive["statistic"]),
            np.asarray(archive["null_statistics"], dtype=np.float64),
            float(archive["level"]),
        )


def compute_mixture_density(
    mixture: GaussianMixture, points: Float[np.ndarray, " points"]
) -> Float[np.ndarray, " points"]:
    """Evaluate the density Σ_c w_c N(x; μ_c, σ_c²) of a 1D Gaussian mixture.

    Args:
        mixture: A mixture on R.
        points: The x values: shape (points,).

    Returns:
        The density at each x: shape (points,).
    """
    # Component weights w_c, shape (C,)
    weights = np.asarray(mixture.weights, dtype=np.float64)
    # Component means μ_c, shape (C,)
    means = np.asarray(mixture.means, dtype=np.float64)[:, 0]
    # Component standard deviations σ_c, shape (C,)
    deviations = np.sqrt(np.asarray(mixture.covariances, dtype=np.float64)[:, 0, 0])
    # Standardized distances (x − μ_c)/σ_c, shape (points, C)
    standardized = (points[:, None] - means[None, :]) / deviations[None, :]
    # Component densities N(x; μ_c, σ_c²), shape (points, C)
    component_densities = np.exp(-0.5 * standardized**2) / (np.sqrt(2.0 * np.pi) * deviations)
    # Mixture density, shape (points,)
    return component_densities @ weights


def draw_1d_task(axes: Axes, task: TradeoffTask) -> None:
    """Draw the densities of P and Q on [−R, R], the example samples below them, and ±R."""
    task_pair = getattr(tradeoff_tasks, task.name)
    # Points of [−R, R] where the densities are drawn, shape (2000,)
    grid = np.linspace(-task.radius, task.radius, 2000)
    # Densities of P and Q, shape (2000,) each
    first_density = compute_mixture_density(task_pair.first_mixture, grid)
    second_density = compute_mixture_density(task_pair.second_mixture, grid)
    style_axes(axes)
    axes.grid(True, axis="y")
    for density, color, label in (
        (first_density, P_COLOR, "P density"),
        (second_density, Q_COLOR, "Q density"),
    ):
        axes.fill_between(grid, density, color=color, alpha=0.18)
        axes.plot(grid, density, color=color, linewidth=3.0, label=label)
    for sample, color, height in (
        (task.first_sample, P_COLOR, -0.012),
        (task.second_sample, Q_COLOR, -0.034),
    ):
        axes.scatter(
            sample[:, 0],
            np.full(sample.shape[0], height),
            s=12,
            color=color,
            edgecolors="none",
            alpha=0.65,
            clip_on=False,
        )
    for edge in (-task.radius, task.radius):
        axes.axvline(edge, color=RADIUS_COLOR, linestyle="--", linewidth=1.6)
    axes.set_xlim(-1.05 * task.radius, 1.05 * task.radius)
    axes.set_ylim(-0.05, 1.10 * max(float(np.max(first_density)), float(np.max(second_density))))
    axes.set_ylabel(f"{convert_to_bold_mathtext(TASK_TITLES[task.name])}\nDensity")


def draw_2d_task(axes: Axes, task: TradeoffTask) -> None:
    """Draw the example samples of P and Q in the plane and the circle of radius R."""
    style_axes(axes)
    axes.add_patch(
        Circle(
            (0.0, 0.0), task.radius, fill=False, linestyle="--", linewidth=1.6, color=RADIUS_COLOR
        )
    )
    for sample, color, marker, label in (
        (task.first_sample, P_COLOR, "o", "P sample"),
        (task.second_sample, Q_COLOR, "s", "Q sample"),
    ):
        axes.scatter(
            sample[:, 0],
            sample[:, 1],
            s=54,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.88,
            label=label,
        )
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(-1.10 * task.radius, 1.10 * task.radius)
    axes.set_ylim(-1.10 * task.radius, 1.10 * task.radius)
    axes.set_ylabel(TASK_TITLES[task.name], fontweight="bold")


def draw_tradeoff_heatmap(
    axes: Axes,
    heatmap: Float[np.ndarray, "budgets discounts"],
    discounts: Float[np.ndarray, " discounts"],
    budgets: Float[np.ndarray, " budgets"],
) -> QuadMesh:
    """Draw one heatmap of Algorithm 4 and the curve τ̄ = γ², the smallest possible factor.

    Cells where no kernel of the family fits the budget (NaN) are left white.

    Returns:
        The drawn mesh, for the shared color bar.
    """
    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad("white")
    style_axes(axes)
    axes.grid(False)
    mesh = axes.pcolormesh(
        build_cell_edges(discounts),
        build_cell_edges(budgets),
        heatmap,
        shading="flat",
        vmin=0.0,
        vmax=1.0,
        cmap=color_map,
        rasterized=True,
    )
    axes.plot(
        discounts,
        discounts**2,
        color=Q_COLOR,
        linestyle=":",
        linewidth=2.2,
        label=r"Smallest possible factor $\gamma^2$",
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks([0.0, 0.5, 1.0])
    axes.set_yticks([0.0, 0.5, 1.0])
    return mesh


def plot_tradeoff_figure(
    archives: TradeoffArchives, task_names: tuple[str, ...], *, two_dimensional: bool
) -> Figure:
    """Draw Figure 3 (1D tasks) or Figure 17 (2D tasks).

    One row per task: the task's data, then one heatmap per kernel family.

    Args:
        archives: The tradeoff data.
        task_names: The tasks, one per row.
        two_dimensional: Whether the tasks are 2D (scatter plots) or 1D (densities).

    Returns:
        The figure, with one shared color bar.
    """
    figure = plt.figure(figsize=(13.8, 10.2 if two_dimensional else 8.4), layout="constrained")
    # One row of axes per task: the task's data, then one heatmap per family
    axes: list[list[Axes]] = [
        list(row)
        for row in figure.subplots(
            len(task_names),
            1 + len(FAMILY_TITLES),
            squeeze=False,
            gridspec_kw={"width_ratios": [1.55] + [1.0] * len(FAMILY_TITLES)},
        )
    ]
    meshes: list[QuadMesh] = []
    for row, name in enumerate(task_names):
        task = archives.tasks[name]
        last_row = row == len(task_names) - 1
        data_axes = axes[row][0]
        if two_dimensional:
            draw_2d_task(data_axes, task)
            data_axes.set_xlabel(r"$x_1$" if last_row else "")
        else:
            draw_1d_task(data_axes, task)
            data_axes.set_xlabel(r"$x$" if last_row else "")
        if row == 0:
            data_axes.set_title("2D Datasets" if two_dimensional else "1D Datasets")
        for column, (family, title) in enumerate(FAMILY_TITLES.items(), start=1):
            heatmap_axes = axes[row][column]
            meshes.append(
                draw_tradeoff_heatmap(
                    heatmap_axes, task.heatmaps[family], archives.discounts, archives.budgets
                )
            )
            if row == 0:
                heatmap_axes.set_title(title)
            heatmap_axes.set_xlabel(r"Discount" "\n" r"factor $\gamma$" if last_row else "")
            heatmap_axes.set_ylabel(
                r"Contraction" "\n" r"Budget $\bar{\tau}$" if column == 1 else ""
            )
    color_bar = figure.colorbar(
        meshes[-1],
        ax=[heatmap_axes for row_axes in axes for heatmap_axes in row_axes[1:]],
        fraction=0.028,
        pad=0.02,
    )
    color_bar.set_label(r"Best power under $\tau^\star \leq \bar{\tau}$")
    color_bar.ax.spines["outline"].set_visible(False)
    # One legend for the samples or densities and the γ² curve, centered below the panels
    data_handles, data_labels = axes[0][0].get_legend_handles_labels()
    curve_handles, curve_labels = axes[0][1].get_legend_handles_labels()
    figure.legend(
        data_handles + curve_handles,
        data_labels + curve_labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    return figure


def compute_permutation_cutoff(null: PermutationNull) -> float:
    """Find the null statistic the observed one must exceed for the test to reject.

    The test rejects when p̂ = (1 + #{T_π ≥ T}) / (B_perm + 1) ≤ α, that is when fewer than
    k = ⌊α (B_perm + 1)⌋ null statistics reach T: when T is above the k-th largest T_π.

    Returns:
        The k-th largest null statistic.
    """
    # Number of shuffles B_perm
    permutation_count = null.null_statistics.size
    # Rank k = ⌊α (B_perm + 1)⌋ of the cutoff among the largest null statistics
    rank = int(np.floor(null.level * (permutation_count + 1)))
    # k-th largest null statistic
    return float(np.sort(null.null_statistics)[permutation_count - rank])


def plot_permutation_null(null: PermutationNull) -> Figure:
    """Draw Figure 16: the checkerboard dataset, and the null statistics with T and the cutoff."""
    figure = plt.figure(figsize=(12.0, 4.3))
    # The dataset on the left, the null statistics on the right
    data_axes, null_axes = figure.subplots(
        1, 2, squeeze=False, gridspec_kw={"width_ratios": [1.0, 1.45]}
    )[0]
    figure.subplots_adjust(wspace=0.14)
    for axes in (data_axes, null_axes):
        style_axes(axes)
        axes.tick_params(axis="both", width=1.1, length=3.5)
    for sample, color, marker, label in (
        (null.first_sample, P_COLOR, "o", r"$P$ sample"),
        (null.second_sample, Q_COLOR, "s", r"$Q$ sample"),
    ):
        data_axes.scatter(
            sample[:, 0],
            sample[:, 1],
            s=56,
            color=color,
            marker=marker,
            edgecolors=TEXT_COLOR,
            linewidths=0.6,
            alpha=0.86,
            label=label,
        )
    data_axes.set_title("Checkerboard: one realization", fontweight="bold")
    data_axes.set_xlabel(r"$x_1$")
    data_axes.set_ylabel(r"$x_2$")
    data_axes.set_aspect("equal", adjustable="box")
    data_axes.legend(frameon=False, loc="best")
    null_axes.hist(
        null.null_statistics,
        bins=36,
        density=True,
        color=P_COLOR,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.78,
    )
    null_axes.axvline(
        compute_permutation_cutoff(null),
        color=RADIUS_COLOR,
        linestyle="--",
        linewidth=2.6,
        label="permutation cutoff",
    )
    null_axes.axvline(null.statistic, color=Q_COLOR, linewidth=3.0, label="observed statistic")
    null_axes.set_title(
        r"$\widehat{\mathrm{MMD}}^2_{\mathrm{RBF}}$: null by label permutation", fontweight="bold"
    )
    null_axes.set_xlabel(r"Biased $\widehat{\mathrm{MMD}}^2$ statistic")
    null_axes.set_ylabel("Density")
    null_axes.legend(frameon=False, loc="best")
    return figure


def save_figure(figure: Figure, path: Path) -> None:
    """Write a figure as PDF, without creation date so the same archives give the same file."""
    figure.savefig(
        path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None},
    )
    plt.close(figure)


def main() -> None:
    """Read the archives and write Figures 3, 16 and 17 as PDFs."""
    parser = argparse.ArgumentParser(description="Draw Figures 3, 16 and 17 from the archives.")
    parser.add_argument("--archives", type=Path, required=True, help="folder of the .npz archives")
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figures")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    archives = load_tradeoff_archives(arguments.archives)
    null = load_permutation_null(arguments.archives / "figure_14_permutation_null.npz")
    with plt.rc_context(TRADEOFF_RC):
        save_figure(
            plot_tradeoff_figure(archives, FIGURE_3_TASKS, two_dimensional=False),
            arguments.output / "empirical_envelope__1d_kernel_comparison.pdf",
        )
        save_figure(
            plot_tradeoff_figure(archives, FIGURE_17_TASKS, two_dimensional=True),
            arguments.output / "empirical_envelope__2d_kernel_comparison.pdf",
        )
    with plt.rc_context(NULL_RC):
        save_figure(
            plot_permutation_null(null),
            arguments.output / "checkerboard_detectability_reference_null.pdf",
        )
    print(
        f"Figure 16: statistic {null.statistic:.6f}, cutoff {compute_permutation_cutoff(null):.6f}"
    )


if __name__ == "__main__":
    main()
