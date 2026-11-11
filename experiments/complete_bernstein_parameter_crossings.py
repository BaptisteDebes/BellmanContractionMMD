"""Estimate where an entry stops inducing contraction as its parameter varies (Table 8, Appendix G).

For Entries 6, 84 and 131, the parameter sweeps show the largest estimate of τ⋆(λ) over the 100
discounts crossing 1 inside an interval of the parameter. The crossing is the zero of

    f(value) = (largest estimate of τ⋆(λ) over the discounts, at this parameter value) − 1,

negative where the entry induces contraction and positive where it does not. Brent's method
finds it inside that interval, where f has opposite signs at the two ends. Each evaluation of f
computes a full curve with compute_estimate_curves, that is 100 searches run in parallel; the
evaluations follow one another.

Usage:
    python experiments/complete_bernstein_parameter_crossings.py --output <folder> [--processes N]

Running time: about 1 CPU-hour (estimate).
"""

import argparse
import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from complete_bernstein_parameter_sweeps import compute_estimate_curves
from parallel import count_usable_cpus
from scipy.optimize import (  # pyright: ignore[reportMissingTypeStubs]
    RootResults,
    brentq,  # pyright: ignore[reportUnknownVariableType]
)

PARAMETER_CROSSINGS: dict[int, tuple[dict[str, float], str, tuple[float, float]]] = {
    # Schilling number → (its fixed parameters, the parameter that varies, the interval of the
    # parameter where the sweeps show the crossing)
    6: ({"a": 1.0}, "alpha", (0.75, 0.80)),
    84: ({}, "alpha", (0.70, 0.75)),
    131: ({"a": 1.0}, "nu", (0.25, 0.30)),
}
CROSSING_TOLERANCE = 1e-4  # precision on the parameter at which the root search stops
NAME = "complete_bernstein_parameter_crossings"  # of the result file and of the folder of curves


@dataclass(frozen=True)
class ParameterCrossing:
    """Where the largest estimate of τ⋆(λ) over the discounts crosses 1, for one entry."""

    crossing: float
    """The zero of f found by Brent's method."""
    converged: bool
    """Whether Brent's method converged."""
    evaluations: list[tuple[float, float]]
    """Each (parameter value, largest estimate) computed, in order."""


def compute_largest_estimate(
    schilling_number: int, parameters: Mapping[str, float], output: Path, processes: int
) -> float:
    """Compute the largest estimate of τ⋆(λ) over the 100 discounts, for one parameter value.

    The curve is saved in output under a name made of the entry and its parameters, so a
    stopped run reuses the curves already computed.

    Args:
        schilling_number: The entry's number in Schilling, Song and Vondraček's table.
        parameters: The entry's parameters.
        output: The folder of the curves.
        processes: Number of processes running searches at the same time.

    Returns:
        The largest of the 100 estimates.
    """
    # File name of this curve, e.g. "entry_84_alpha_0.7015", without extension
    name = f"entry_{schilling_number}_" + "_".join(
        f"{key}_{value!r}" for key, value in parameters.items()
    )
    arrays = compute_estimate_curves([(schilling_number, parameters)], output, name, processes)
    return float(arrays["estimate"].max())


def find_parameter_crossing(
    schilling_number: int,
    fixed_parameters: Mapping[str, float],
    varied_parameter: str,
    interval: tuple[float, float],
    output: Path,
    processes: int,
) -> ParameterCrossing:
    """Find where the largest estimate of τ⋆(λ) over the discounts crosses 1, by Brent's method.

    Args:
        schilling_number: The entry's number in Schilling, Song and Vondraček's table.
        fixed_parameters: Its parameters that do not vary.
        varied_parameter: The name of the parameter that varies.
        interval: The ends of the interval of that parameter, where f has opposite signs.
        output: The folder of the curves.
        processes: Number of processes running searches at the same time.

    Returns:
        The crossing, whether Brent's method converged, and every value it computed.

    Raises:
        ValueError: If f has the same sign at both ends of the interval.
    """
    # (parameter value, largest estimate), in the order the root search computes them
    evaluations: list[tuple[float, float]] = []

    def compute_distance_above_one(value: float) -> float:
        # f(value) = largest estimate − 1
        parameters = {**fixed_parameters, varied_parameter: float(value)}
        largest = compute_largest_estimate(schilling_number, parameters, output, processes)
        evaluations.append((float(value), largest))
        return largest - 1.0

    crossing, details = cast(
        tuple[float, RootResults],
        brentq(
            compute_distance_above_one,
            interval[0],
            interval[1],
            xtol=CROSSING_TOLERANCE,
            full_output=True,
        ),
    )
    return ParameterCrossing(crossing, bool(details.converged), evaluations)


def main() -> None:
    """Find the three crossings and write complete_bernstein_parameter_crossings.json."""
    parser = argparse.ArgumentParser(
        description="Find where the entries of Table 8 stop contracting."
    )
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    # Folder of the curves computed by the root searches
    curves = arguments.output / NAME
    curves.mkdir(parents=True, exist_ok=True)
    # Schilling number (as text) → its crossing, with what was computed to find it
    crossings: dict[str, dict[str, object]] = {}
    for number, (fixed, varied, interval) in PARAMETER_CROSSINGS.items():
        result = find_parameter_crossing(
            number, fixed, varied, interval, curves, arguments.processes
        )
        crossings[str(number)] = {
            "parameter": varied,
            "interval": list(interval),
            **dataclasses.asdict(result),
        }
        print(f"Entry {number}: {varied} ≈ {result.crossing:.6f}", flush=True)
    (arguments.output / f"{NAME}.json").write_text(json.dumps(crossings, indent=2) + "\n")


if __name__ == "__main__":
    main()
