"""Compute the estimates of τ⋆(λ) for the entries of Figures 6–8 (Appendix G) on this machine.

For each of the 43 entries of Tables 3–6 at its default parameters and each of 100
discounts λ from 0.01 to 0.99, estimate τ⋆(λ) with the search of Algorithm 2 and the search
settings listed in Appendix G. Each (entry, discount) is one computation.

Usage:
    python experiments/complete_bernstein_entries.py --output <folder> [--processes N]
    python experiments/complete_bernstein_parameter_sweeps.py --output <folder> [--processes N]
    python experiments/complete_bernstein_parameter_crossings.py --output <folder> [--processes N]
    python paper_figures/complete_bernstein_figures.py --archives <folder> --output figures

Running time: about 5 CPU-hours (estimate).
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from parallel import Computation, ComputationResult, count_usable_cpus, run_computations_in_parallel

from mmd_contraction import complete_bernstein

DISCOUNTS = np.linspace(0.01, 0.99, 100)  # λ
SEARCH_SETTINGS = complete_bernstein.SupremumSearchSettings(  # Table 7
    half_width=80.0,
    grid_size=8500,
    refined_point_count=12,
    refinement_half_width=120.0,
    refinement_tolerance=1e-10,
    decimal_digits=80,
)
RESULT_FIELDS = (  # what the archives keep of each search, one array per field
    "estimate",
    "best_log_argument",
    "at_window_edge",
)
ARCHIVE_NAME = "complete_bernstein_entries.npz"


def search_estimate(
    schilling_number: int, parameters: Mapping[str, float], discount_index: int
) -> ComputationResult:
    """Estimate τ⋆(λ) of one entry at one discount, with the result fields the archive keeps.

    Args:
        schilling_number: The entry's number in Schilling, Song and Vondraček's table.
        parameters: The entry's parameters.
        discount_index: The position of λ in DISCOUNTS.

    Returns:
        Each of RESULT_FIELDS → its value.
    """
    result = complete_bernstein.find_ratio_supremum(
        complete_bernstein.ENTRIES[schilling_number],
        parameters,
        float(DISCOUNTS[discount_index]),
        SEARCH_SETTINGS,
    )
    return {field: getattr(result, field) for field in RESULT_FIELDS}


def run_entry_computation(computation: Computation) -> ComputationResult:
    """One computation: (Schilling number, discount index), at the entry's default parameters."""
    schilling_number, discount_index = int(computation[0]), int(computation[1])
    entry = complete_bernstein.ENTRIES[schilling_number]
    return search_estimate(schilling_number, entry.default_parameters, discount_index)


def main() -> None:
    """Compute the 43 entries × 100 discounts and write complete_bernstein_entries.npz."""
    parser = argparse.ArgumentParser(
        description="Compute the estimates of τ⋆(λ) for the 43 entries."
    )
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # Schilling numbers of the 43 entries, in increasing order
    numbers = sorted(complete_bernstein.ENTRIES)
    # One computation per (entry, discount index): 43 × 100 computations
    computations: list[Computation] = [
        (number, index) for number in numbers for index in range(len(DISCOUNTS))
    ]
    results = run_computations_in_parallel(
        computations,
        run_entry_computation,
        arguments.output / "complete_bernstein_entries.jsonl",
        arguments.processes,
    )
    arrays: dict[str, np.ndarray] = {"discounts": DISCOUNTS, "entries": np.asarray(numbers)}
    for number in numbers:
        for field in RESULT_FIELDS:
            # The field at each discount for this entry, shape (discounts,)
            arrays[f"entry_{number}__{field}"] = np.asarray(
                [results[(number, index)][field] for index in range(len(DISCOUNTS))]
            )
    np.savez(arguments.output / ARCHIVE_NAME, allow_pickle=False, **arrays)


if __name__ == "__main__":
    main()
