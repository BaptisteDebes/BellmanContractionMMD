"""Compute τ⋆(λ) over a grid of parameter values for the entries of Table 8 (Appendix G).

In Entries 6, 14, 20, 83, 84, 131 and 132, one parameter changes the shape of the function, so
τ⋆(λ) may depend on it. For each of these entries, that parameter takes the values listed in
PARAMETER_SWEEPS and the other parameters keep their default values. Each value is computed
at the 100 discounts of Figures 6–8, with the same search and settings. Each
(entry, parameters, discount) is one computation.

The largest estimate over the discounts, plotted against the parameter, shows for which values
the entry induces contraction. The other entries of Table 8 need no sweep: Entries 1 and 2 are
settled for every α by Propositions F.1 and F.2, and in Entries 5, 18, 38, 64 and 82 the
parameter only rescales the argument of φ, which leaves τ⋆(λ) unchanged.

Figures 9 and 10 are drawn from these curves. compute_estimate_curves, which computes them,
also serves the root search of complete_bernstein_parameter_crossings.py.

Usage:
    python experiments/complete_bernstein_parameter_sweeps.py --output <folder> [--processes N]

Running time: about 30 CPU-hours (estimate).
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from complete_bernstein_entries import DISCOUNTS, RESULT_FIELDS, search_estimate
from parallel import Computation, ComputationResult, count_usable_cpus, run_computations_in_parallel

# 0.05, 0.10, …, 0.95, for the parameters that range over (0, 1)
VALUES_IN_UNIT_INTERVAL = [round(0.05 * step, 2) for step in range(1, 20)]
PARAMETER_SWEEPS: dict[int, tuple[dict[str, float], str, list[float]]] = {
    # Schilling number → (its fixed parameters, the parameter that varies, its values)
    6: ({"a": 1.0}, "alpha", VALUES_IN_UNIT_INTERVAL),
    # α ranges over (−1, 1); α = 0 is left out, where φ's formula divides by sin(απ) = 0
    14: ({"c": 2.0}, "alpha", [round(0.05 * step, 2) for step in range(-19, 20) if step != 0]),
    # a ranges over (0, ∞)
    20: ({}, "a", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]),
    83: ({}, "alpha", VALUES_IN_UNIT_INTERVAL),
    84: ({}, "alpha", VALUES_IN_UNIT_INTERVAL),
    131: ({"a": 1.0}, "nu", VALUES_IN_UNIT_INTERVAL),
    132: ({"a": 1.0}, "nu", VALUES_IN_UNIT_INTERVAL),
}
NAME = "complete_bernstein_parameter_sweeps"  # of the results file and of the archive


def run_estimate_computation(computation: Computation) -> ComputationResult:
    """One computation: (Schilling number, parameters as JSON text, discount index)."""
    schilling_number, parameters_text, discount_index = computation
    return search_estimate(
        int(schilling_number), json.loads(str(parameters_text)), int(discount_index)
    )


def compute_estimate_curves(
    rows: Sequence[tuple[int, Mapping[str, float]]], output: Path, name: str, processes: int
) -> dict[str, np.ndarray]:
    """Compute τ⋆(λ) of each row at the 100 discounts and write the archive <name>.npz.

    Each (row, discount) is one computation. Its result is appended to <name>.jsonl as soon as
    it is computed, so a stopped run continues where it stopped.

    Args:
        rows: Each (Schilling number, parameters) whose curve is computed, all distinct.
        output: The folder of the results file and of the archive.
        name: The name of both files, without extension.
        processes: Number of processes running computations at the same time.

    Returns:
        The arrays of the archive: "discounts", shape (discounts,); "row_entries" and
        "row_parameters", shape (rows,); and each result field, shape (rows, discounts).
    """
    # Parameters of each row as JSON text, the form a computation carries them in, shape (rows,)
    parameter_texts = [json.dumps(dict(parameters)) for _, parameters in rows]
    # Each row as a computation without its discount index, shape (rows,)
    row_keys = [(number, text) for (number, _), text in zip(rows, parameter_texts, strict=True)]
    # One computation per (row, discount index): rows × 100 computations
    computations: list[Computation] = [
        (number, text, index) for number, text in row_keys for index in range(len(DISCOUNTS))
    ]
    results = run_computations_in_parallel(
        computations, run_estimate_computation, output / f"{name}.jsonl", processes
    )
    arrays: dict[str, np.ndarray] = {
        "discounts": DISCOUNTS,
        # Schilling number of each row, shape (rows,)
        "row_entries": np.asarray([number for number, _ in rows]),
        # Parameters of each row as JSON text, shape (rows,)
        "row_parameters": np.asarray(parameter_texts),
    }
    for field in RESULT_FIELDS:
        # The field for each row and discount, shape (rows, discounts)
        arrays[field] = np.asarray(
            [
                [results[(number, text, index)][field] for index in range(len(DISCOUNTS))]
                for number, text in row_keys
            ]
        )
    np.savez(output / f"{name}.npz", allow_pickle=False, **arrays)
    return arrays


def list_sweep_rows() -> list[tuple[int, dict[str, float]]]:
    """List every (Schilling number, parameters) of PARAMETER_SWEEPS, entry by entry.

    Returns:
        The 142 rows, each entry's values in the order of PARAMETER_SWEEPS.
    """
    return [
        (number, {**fixed, varied: value})
        for number, (fixed, varied, values) in PARAMETER_SWEEPS.items()
        for value in values
    ]


def main() -> None:
    """Compute the 142 rows × 100 discounts and write complete_bernstein_parameter_sweeps.npz."""
    parser = argparse.ArgumentParser(description="Compute the parameter sweeps of Table 8.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    compute_estimate_curves(list_sweep_rows(), arguments.output, NAME, arguments.processes)


if __name__ == "__main__":
    main()
