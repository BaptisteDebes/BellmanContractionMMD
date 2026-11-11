"""Draw the estimated τ⋆ curves of the complete Bernstein entries and of their parameters.

One panel per entry of Tables 3–6: its estimated τ⋆(λ) over the discounts λ, with
the line τ⋆ = 1. By the largest value of its curve, and by whether its best points stay inside
the search window, each entry goes into one of three figures: rejected, possibly
non-expansive, accepted. For the seven entries of the parameter sweeps, a fourth figure
shows τ⋆(λ) at a few values of their parameter, and a sixth the largest τ⋆(λ) over the
discounts against the parameter, with the parameter value where it crosses 1. A fifth shows
the profile φ(t) − φ(0) of each accepted entry. This script computes no τ⋆; it only
evaluates φ.

Usage:
    python paper_figures/complete_bernstein_figures.py --archives <folder> --output figures
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import mpmath as mp  # pyright: ignore[reportMissingTypeStubs]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from figure_style import style_axes
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, MaxNLocator
from tradeoff_figures import TRADEOFF_RC, save_figure

from mmd_contraction import complete_bernstein

CLASS_FIGURES = {  # class of an entry → (PDF name, curve color), in the paper's order
    "rejected": ("rejected_tau_star_curves.pdf", "#E15759"),
    "borderline": ("possibly_non_expansive_tau_star_curves.pdf", "#F28E2B"),
    "accepted": ("accepted_tau_star_curves.pdf", "#59A14F"),
}
LINE_AT_ONE_COLOR = "#999999"
CLASSIFICATION_TOLERANCE = 1e-10  # values within this distance of 1 count as 1
FLAT_TOLERANCE = 1e-3  # curves within this distance of 1 everywhere get the axis 0.999–1.001
COLUMN_COUNT = 4
# 13.1 in wide, shown at 0.95 of the text width: the scale of the tradeoff figures, so the
# text of TRADEOFF_RC has their size on the page
FIGURE_WIDTH = 13.1
ROW_HEIGHT = 2.6
PROFILE_POINTS = np.geomspace(1e-3, 1e4, 700)  # arguments t of the profiles, as in the old figure
ZERO_SUBSTITUTE = 1e-40  # φ(0) is taken as φ at this argument
PROFILE_DECIMAL_DIGITS = 30
PROFILE_FILE_NAME = "selected_bernstein_functions_grid_normalized_logx_logy.pdf"
ENTRIES_ARCHIVE_NAME = "complete_bernstein_entries.npz"
PARAMETRIC_FILE_NAME = "parameter_dependent_tau_star_transitions.pdf"
# Colors of the curves of one panel, from the smallest largest value to the largest: blue,
# teal, orange, purple (fifth curve only), red; the curve with the largest values is red
PARAMETRIC_COLORS = ("#4E79A7", "#76B7B2", "#F28E2B", "#B07AA1", "#E15759")
PARAMETRIC_ROW_HEIGHT = 4.4  # taller than ROW_HEIGHT, for the legends under the panels
PARAMETER_SYMBOLS = {"a": "a", "c": "c", "alpha": r"\alpha", "nu": r"\nu"}
SWEEP_ARCHIVE_NAME = "complete_bernstein_parameter_sweeps.npz"
CROSSINGS_FILE_NAME = "complete_bernstein_parameter_crossings.json"
TRANSITIONS_FILE_NAME = "largest_tau_star_by_parameter.pdf"
TRANSITIONS_COLOR = "#4E79A7"
TRANSITIONS_CUT = 2.0  # largest value shown; the curves of Entries 6, 14, 84 and 131 go higher
LOG_SCALE_PARAMETERS = {20}  # entries whose parameter is shown on a logarithmic axis
# The curves of Figure 9, taken from the parameter sweeps: entry → (the parameter that varies,
# its values drawn), in panel order
FIGURE_9_VALUES: dict[int, tuple[str, list[float]]] = {
    6: ("alpha", [0.5, 0.7, 0.75, 0.8]),
    14: ("alpha", [-0.2, -0.05, 0.05, 0.5, 0.75]),
    20: ("a", [0.01, 1.0, 100.0]),
    83: ("alpha", [0.25, 0.5, 0.75]),
    84: ("alpha", [0.5, 0.7, 0.75, 0.8]),
    131: ("nu", [0.2, 0.3, 0.35, 0.5]),
    132: ("nu", [0.25, 0.5, 0.75]),
}


def set_vertical_range_with_one(panel: Axes, curves: list[Float[np.ndarray, " discounts"]]) -> None:
    """Show the curves between their smallest and largest values, with 1 included.

    Args:
        panel: The panel.
        curves: The curves drawn in it, each of shape (discounts,).
    """
    # Range of the curves with 1 included, [min(τ⋆, 1), max(τ⋆, 1)]
    low = min(min(float(np.min(curve)) for curve in curves), 1.0)
    high = max(max(float(np.max(curve)) for curve in curves), 1.0)
    # Margin of 5% of that range on each side
    margin = 0.05 * (high - low)
    panel.set_ylim(low - margin, high + margin)
    # At most 3 ticks, spaced 1, 2 or 5 times a power of ten
    panel.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))


def plot_estimate_curves(
    discounts: Float[np.ndarray, " discounts"],
    curves: dict[int, Float[np.ndarray, " discounts"]],
    color: str,
    flat_tolerance: float,
) -> Figure:
    """Draw one panel per entry, four per row: its curve τ⋆(λ) and the dashed line τ⋆ = 1.

    A curve within flat_tolerance of 1 at every discount is shown between 1 − flat_tolerance
    and 1 + flat_tolerance; any other curve between its smallest and largest values, with 1
    included and 5% margins.

    Args:
        discounts: The discounts λ, shape (discounts,).
        curves: Entry number → its estimated τ⋆(λ), shape (discounts,), in panel order.
        color: The color of the curves.
        flat_tolerance: Largest distance from 1 of a curve drawn on the narrow axis.

    Returns:
        The figure.
    """
    row_count = math.ceil(len(curves) / COLUMN_COUNT)
    figure = plt.figure(figsize=(FIGURE_WIDTH, ROW_HEIGHT * row_count), layout="constrained")
    # All panels, row by row
    panels: list[Axes] = list(figure.subplots(row_count, COLUMN_COUNT, squeeze=False).ravel())
    for panel in panels[len(curves) :]:
        panel.set_visible(False)
    for index, (number, curve) in enumerate(curves.items()):
        panel = panels[index]
        style_axes(panel)
        panel.plot(discounts, curve, color=color, linewidth=2.0)
        panel.axhline(1.0, color=LINE_AT_ONE_COLOR, linestyle="--", linewidth=1.1)
        panel.set_title(f"Entry {number}")
        panel.set_xlim(0.0, 1.0)
        panel.set_xticks([0.0, 0.5, 1.0], labels=["0", "0.5", "1"])
        # Largest distance of the curve from 1, max_λ |τ⋆(λ) − 1|
        distance_from_one = float(np.max(np.abs(curve - 1.0)))
        if distance_from_one < flat_tolerance:
            panel.set_ylim(1.0 - flat_tolerance, 1.0 + flat_tolerance)
            panel.set_yticks([1.0 - flat_tolerance, 1.0, 1.0 + flat_tolerance])
        else:
            set_vertical_range_with_one(panel, [curve])
        panel.ticklabel_format(axis="y", style="plain", useOffset=False)
        if index % COLUMN_COUNT == 0:
            panel.set_ylabel(r"$\tau^\star(\lambda)$")
        # The λ label goes under the lowest panel of each column
        if index + COLUMN_COUNT >= len(curves):
            panel.set_xlabel(r"$\lambda$")
    return figure


def plot_parametric_curves(
    discounts: Float[np.ndarray, " discounts"],
    entries: dict[int, list[tuple[dict[str, float], Float[np.ndarray, " discounts"]]]],
) -> Figure:
    """Draw one panel per entry, four per row: τ⋆(λ) at each parameter value, and τ⋆ = 1.

    A parameter with the same value in all curves of an entry is written in the panel's
    title; the others name each curve in the legend, with the curve's largest value.

    Args:
        discounts: The discounts λ, shape (discounts,).
        entries: Entry number → its curves, each (parameters, estimated τ⋆(λ) of shape
            (discounts,)), in panel and legend order.

    Returns:
        The figure.
    """
    row_count = math.ceil(len(entries) / COLUMN_COUNT)
    figure = plt.figure(
        figsize=(FIGURE_WIDTH, PARAMETRIC_ROW_HEIGHT * row_count), layout="constrained"
    )
    # All panels, row by row
    panels: list[Axes] = list(figure.subplots(row_count, COLUMN_COUNT, squeeze=False).ravel())
    for panel in panels[len(entries) :]:
        panel.set_visible(False)
    for index, (number, curves) in enumerate(entries.items()):
        panel = panels[index]
        style_axes(panel)
        # Parameters with one value in all curves of the entry, name → value
        shared = {
            name: value
            for name, value in curves[0][0].items()
            if all(parameters[name] == value for parameters, _ in curves)
        }
        # The colors of this panel, the last one red, and the rank of each curve's largest value
        palette = PARAMETRIC_COLORS[: len(curves) - 1] + PARAMETRIC_COLORS[-1:]
        ranks = np.argsort(np.argsort([float(np.max(curve)) for _, curve in curves]))
        for (parameters, curve), rank in zip(curves, ranks, strict=True):
            color = palette[int(rank)]
            # The parameters that differ between curves, e.g. "α = 0.5"
            label = ", ".join(
                f"${PARAMETER_SYMBOLS[name]} = {{{value:g}}}$"
                for name, value in parameters.items()
                if name not in shared
            )
            panel.plot(
                discounts,
                curve,
                color=color,
                linewidth=2.0,
                label=f"{label}, max {float(np.max(curve)):.3f}",
            )
        panel.axhline(1.0, color=LINE_AT_ONE_COLOR, linestyle="--", linewidth=1.1)
        title = f"Entry {number}" + "".join(
            f", ${PARAMETER_SYMBOLS[name]} = {{{value:g}}}$" for name, value in shared.items()
        )
        panel.set_title(title)
        panel.set_xlim(0.0, 1.0)
        panel.set_xticks([0.0, 0.5, 1.0], labels=["0", "0.5", "1"])
        set_vertical_range_with_one(panel, [curve for _, curve in curves])
        panel.ticklabel_format(axis="y", style="plain", useOffset=False)
        panel.set_xlabel(r"$\lambda$")
        # The legend under the panel, one line per curve
        panel.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24))
        if index % COLUMN_COUNT == 0:
            panel.set_ylabel(r"$\tau^\star(\lambda)$")
    return figure


def plot_largest_estimate_by_parameter(
    sweeps: dict[
        int,
        tuple[dict[str, float], str, Float[np.ndarray, " values"], Float[np.ndarray, " values"]],
    ],
    crossings: dict[int, float],
) -> Figure:
    """Draw one panel per entry, four per row: the largest τ⋆(λ) over λ against the parameter.

    Each panel shows the line at 1 and, where the root search found one, a dotted vertical line
    at the parameter value where the curve crosses 1. Values above TRANSITIONS_CUT are cut.

    Args:
        sweeps: Entry number → (its fixed parameters, the parameter that varies, the values of
            that parameter, shape (values,), and the largest estimate of τ⋆(λ) over the
            discounts at each value, shape (values,)), in panel order.
        crossings: Entry number → the parameter value where its curve crosses 1.

    Returns:
        The figure.
    """
    row_count = math.ceil(len(sweeps) / COLUMN_COUNT)
    figure = plt.figure(figsize=(FIGURE_WIDTH, ROW_HEIGHT * row_count), layout="constrained")
    # All panels, row by row
    panels: list[Axes] = list(figure.subplots(row_count, COLUMN_COUNT, squeeze=False).ravel())
    for panel in panels[len(sweeps) :]:
        panel.set_visible(False)
    for index, (number, (fixed, varied, values, largest)) in enumerate(sweeps.items()):
        panel = panels[index]
        style_axes(panel)
        panel.plot(
            values, largest, color=TRANSITIONS_COLOR, linewidth=2.0, marker="o", markersize=3
        )
        panel.axhline(1.0, color=LINE_AT_ONE_COLOR, linestyle="--", linewidth=1.1)
        if number in crossings:
            panel.axvline(crossings[number], color=LINE_AT_ONE_COLOR, linestyle=":", linewidth=1.1)
        if number in LOG_SCALE_PARAMETERS:
            panel.set_xscale("log")
        panel.set_title(
            f"Entry {number}"
            + "".join(
                f", ${PARAMETER_SYMBOLS[name]} = {{{value:g}}}$" for name, value in fixed.items()
            )
        )
        # The curve between its smallest value and the cut, with 1 included
        set_vertical_range_with_one(panel, [np.minimum(largest, TRANSITIONS_CUT)])
        panel.ticklabel_format(axis="y", style="plain", useOffset=False)
        panel.set_xlabel(f"${PARAMETER_SYMBOLS[varied]}$")
        if index % COLUMN_COUNT == 0:
            panel.set_ylabel(r"$\max_\lambda\,\tau^\star(\lambda)$")
    return figure


def compute_profile(
    entry: complete_bernstein.CompleteBernsteinEntry,
    points: Float[np.ndarray, " points"],
    zero_substitute: float,
    decimal_digits: int,
) -> Float[np.ndarray, " points"]:
    """Evaluate φ(t) − φ(0) for one entry at its default parameters.

    Several table formulas are undefined at t = 0, so φ(0), the limit of φ(t) as t → 0,
    is taken as φ(zero_substitute).

    Args:
        entry: The entry of Tables 3–6.
        points: The arguments t > 0, shape (points,).
        zero_substitute: The small argument whose value stands for φ(0).
        decimal_digits: The working precision of mpmath.

    Returns:
        φ(t) − φ(0) at each argument, shape (points,).

    Raises:
        ValueError: If a value is not positive, which a logarithmic axis cannot show.
    """
    with mp.workdps(decimal_digits):
        # The entry's parameters as mpmath numbers
        parameters = {name: mp.mpf(value) for name, value in entry.default_parameters.items()}
        # φ(0), taken as φ(zero_substitute)
        value_at_zero = entry.bernstein_function(mp.mpf(zero_substitute), parameters)
        # φ(t) − φ(0) at each argument, shape (points,)
        profile = np.asarray(
            [
                float(entry.bernstein_function(mp.mpf(float(point)), parameters) - value_at_zero)
                for point in points
            ],
            dtype=np.float64,
        )
    if not np.all(profile > 0.0):
        raise ValueError(f"entry {entry.schilling_number}: φ(t) − φ(0) is not positive")
    return profile


def plot_profiles(
    points: Float[np.ndarray, " points"],
    profiles: dict[int, Float[np.ndarray, " points"]],
    color: str,
) -> Figure:
    """Draw one panel per entry, four per row: φ(t) − φ(0) against t, both axes logarithmic.

    Each vertical axis spans the profile's range, widened by a factor 1.25 at both ends.

    Args:
        points: The arguments t, shape (points,).
        profiles: Entry number → its φ(t) − φ(0), shape (points,), in panel order.
        color: The color of the curves.

    Returns:
        The figure.
    """
    row_count = math.ceil(len(profiles) / COLUMN_COUNT)
    figure = plt.figure(figsize=(FIGURE_WIDTH, ROW_HEIGHT * row_count), layout="constrained")
    # All panels, row by row
    panels: list[Axes] = list(figure.subplots(row_count, COLUMN_COUNT, squeeze=False).ravel())
    for panel in panels[len(profiles) :]:
        panel.set_visible(False)
    for index, (number, profile) in enumerate(profiles.items()):
        panel = panels[index]
        style_axes(panel)
        panel.plot(points, profile, color=color, linewidth=2.0)
        panel.set_title(f"Entry {number}")
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(float(points[0]), float(points[-1]))
        panel.set_ylim(float(np.min(profile)) / 1.25, float(np.max(profile)) * 1.25)
        # At most 4 labelled powers of ten on each axis
        panel.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        panel.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        if index % COLUMN_COUNT == 0:
            panel.set_ylabel(r"$\phi(t) - \phi(0)$")
        # The t label goes under the lowest panel of each column
        if index + COLUMN_COUNT >= len(profiles):
            panel.set_xlabel(r"$t$")
    return figure


def main() -> None:
    """Read both archives, classify each entry by its curve and write the five figures."""
    parser = argparse.ArgumentParser(description="Draw the estimated τ⋆ curves of Figures 6–10.")
    parser.add_argument(
        "--archives",
        type=Path,
        required=True,
        help=(f"folder of {ENTRIES_ARCHIVE_NAME}, {SWEEP_ARCHIVE_NAME} and {CROSSINGS_FILE_NAME}"),
    )
    parser.add_argument("--output", type=Path, required=True, help="folder for the PDF figures")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with np.load(arguments.archives / ENTRIES_ARCHIVE_NAME, allow_pickle=False) as arrays:
        # Discounts λ, shape (discounts,)
        discounts = np.asarray(arrays["discounts"], dtype=np.float64)
        # Entry number → estimated τ⋆(λ), shape (discounts,), by increasing entry number
        curves = {
            int(number): np.asarray(arrays[f"entry_{int(number)}__estimate"], dtype=np.float64)
            for number in arrays["entries"]
        }
        # Entry number → whether its best point is an end of the window, shape (discounts,)
        at_window_edge = {
            int(number): np.asarray(arrays[f"entry_{int(number)}__at_window_edge"], dtype=bool)
            for number in arrays["entries"]
        }
    with np.load(arguments.archives / SWEEP_ARCHIVE_NAME, allow_pickle=False) as arrays:
        # Entry number → its rows, each (parameters, estimated τ⋆(λ) of shape (discounts,)),
        # in the archive's order
        sweep_rows: dict[int, list[tuple[dict[str, float], np.ndarray]]] = {}
        for number, parameters, curve in zip(
            arrays["row_entries"], arrays["row_parameters"], arrays["estimate"], strict=True
        ):
            sweep_rows.setdefault(int(number), []).append(
                (json.loads(str(parameters)), np.asarray(curve, dtype=np.float64))
            )
        if not np.array_equal(arrays["discounts"], discounts):
            raise ValueError("the sweep archive has different discounts")
    # Entry number → the curves of Figure 9, each (parameters, estimated τ⋆(λ)), in legend order
    parametric_curves: dict[int, list[tuple[dict[str, float], np.ndarray]]] = {}
    for number, (varied, values) in FIGURE_9_VALUES.items():
        # The sweep row of each value drawn
        by_value = {
            parameters[varied]: (parameters, curve) for parameters, curve in sweep_rows[number]
        }
        parametric_curves[number] = [by_value[value] for value in values]
    # Entry number → (fixed parameters, varied parameter, its values, largest estimates)
    sweeps: dict[int, tuple[dict[str, float], str, np.ndarray, np.ndarray]] = {}
    for number, rows in sweep_rows.items():
        # The parameter whose value differs between rows, and those that keep one value
        varied = next(name for name in rows[0][0] if rows[1][0][name] != rows[0][0][name])
        fixed = {name: value for name, value in rows[0][0].items() if name != varied}
        values = np.asarray([parameters[varied] for parameters, _ in rows])
        # Largest estimate of τ⋆(λ) over the discounts at each value, shape (values,)
        largest = np.asarray([float(np.max(curve)) for _, curve in rows])
        sweeps[number] = (fixed, varied, values, largest)
    # Entry number → the parameter value where its largest estimate crosses 1
    crossings = {
        int(number): float(result["crossing"])
        for number, result in json.loads(
            (arguments.archives / CROSSINGS_FILE_NAME).read_text()
        ).items()
    }
    # Entry number → "accepted", "borderline" or "rejected", by the largest value of its curve
    # and where its best points lie
    classes = {
        number: complete_bernstein.classify_estimate_curve(
            curve.tolist(), at_window_edge[number].tolist(), tolerance=CLASSIFICATION_TOLERANCE
        )
        for number, curve in curves.items()
    }
    # The panel titles keep their size of 19.5, about 7.8 pt on the page
    with plt.rc_context({**TRADEOFF_RC, "axes.titlesize": 19.5}):
        for name, (file_name, color) in CLASS_FIGURES.items():
            # The curves of this class, by increasing entry number
            selected = {
                number: curve for number, curve in curves.items() if classes[number] == name
            }
            figure = plot_estimate_curves(discounts, selected, color, FLAT_TOLERANCE)
            save_figure(figure, arguments.output / file_name)
            print(f"{name}: {len(selected)} entries, {list(selected)}")
        figure = plot_parametric_curves(discounts, parametric_curves)
        save_figure(figure, arguments.output / PARAMETRIC_FILE_NAME)
        print(f"parametric entries: {list(parametric_curves)}")
        figure = plot_largest_estimate_by_parameter(sweeps, crossings)
        save_figure(figure, arguments.output / TRANSITIONS_FILE_NAME)
        print(f"sweeps: {list(sweeps)}, crossings: {crossings}")
        # Entry number → φ(t) − φ(0) at the profile points, for the accepted entries
        profiles = {
            number: compute_profile(
                complete_bernstein.ENTRIES[number],
                PROFILE_POINTS,
                ZERO_SUBSTITUTE,
                PROFILE_DECIMAL_DIGITS,
            )
            for number in curves
            if classes[number] == "accepted"
        }
        figure = plot_profiles(PROFILE_POINTS, profiles, CLASS_FIGURES["accepted"][1])
        save_figure(figure, arguments.output / PROFILE_FILE_NAME)
        print(f"profiles: {list(profiles)}")


if __name__ == "__main__":
    main()
