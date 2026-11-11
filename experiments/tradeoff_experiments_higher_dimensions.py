"""Compute the tradeoff experiments in higher dimensions of Figure 18 (Appendix H.3).

For the cube checkerboard and the Gaussian mean shift in dimensions 2, 3, 4, 8, 16, 32 and 64,
and for each power-distance kernel −‖x − y‖^α with 50 exponents α from 0.2 to 2: the power of
the permutation test (1000 datasets, 1000 permutations, level 0.05, 100 points per sample).
The contraction factor of these kernels is exactly γ^α (Proposition 4.1), so the heatmap of
Algorithm 4 is built from it at the end. Each (pair, dimension, exponent) is one computation.

Usage:
    python experiments/tradeoff_experiments_higher_dimensions.py --output <folder> \
        [--processes N]
    python paper_figures/dimension_tradeoff_figure.py \
        --archive <folder>/dimension_tradeoff.npz --output figures

Running time: about 11 CPU-hours (36 minutes on 32 CPUs).
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from parallel import (
    Computation,
    ComputationResult,
    count_usable_cpus,
    run_computations_in_parallel,
)

from mmd_contraction.kernels import PowerDistanceKernel
from mmd_contraction.permutation_test import TwoSampleProblem, estimate_test_power
from mmd_contraction.tradeoff import build_tradeoff_heatmap
from mmd_contraction.tradeoff_tasks import GaussianMeanShiftPair, ParityCheckerboardPair

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

PAIR_NAMES = ("parity_checkerboard", "gaussian_mean_shift")  # the figure's row order
DIMENSIONS = (2, 3, 4, 8, 16, 32, 64)
EXPONENTS = np.linspace(0.2, 2.0, 50)  # α in −‖x − y‖^α
DISCOUNTS = np.linspace(0.01, 1.0, 100)  # γ
BUDGETS = np.linspace(0.01, 1.0, 100)  # τ̄
POWER_SEED_BASE = 20260800  # + index of (pair, dimension), the same for every exponent
SAMPLE_SIZE = 100
CHECKERBOARD_RADIUS = 1.0
MEAN_SEPARATION = 0.8
DATASET_COUNT = 1000
PERMUTATION_COUNT = 1000
LEVEL = 0.05


def build_pair(pair_name: str, dimension: int) -> TwoSampleProblem:
    """The two distributions of one row of the figure in one dimension, 100 points each.

    Args:
        pair_name: "parity_checkerboard" or "gaussian_mean_shift".
        dimension: d.

    Returns:
        The pair, with its sample sizes.
    """
    if pair_name == "parity_checkerboard":
        return ParityCheckerboardPair(
            dimension=dimension,
            first_sample_size=SAMPLE_SIZE,
            second_sample_size=SAMPLE_SIZE,
            radius=CHECKERBOARD_RADIUS,
        )
    return GaussianMeanShiftPair(
        dimension=dimension,
        mean_separation=MEAN_SEPARATION,
        first_sample_size=SAMPLE_SIZE,
        second_sample_size=SAMPLE_SIZE,
    )


def estimate_power(pair_name: str, dimension: int, exponent_index: int) -> float:
    """The power of the permutation test of one power-distance kernel on one pair.

    Args:
        pair_name: "parity_checkerboard" or "gaussian_mean_shift".
        dimension: d.
        exponent_index: The position of α in EXPONENTS.

    Returns:
        The fraction of the DATASET_COUNT datasets on which the test rejects at LEVEL.
    """
    # Index of (pair, dimension), in the order pair first, then dimension
    pair_dimension_index = PAIR_NAMES.index(pair_name) * len(DIMENSIONS) + DIMENSIONS.index(
        dimension
    )
    power = estimate_test_power(
        jax.random.key(POWER_SEED_BASE + pair_dimension_index),
        PowerDistanceKernel(exponent=float(EXPONENTS[exponent_index])),
        build_pair(pair_name, dimension),
        dataset_count=DATASET_COUNT,
        permutation_count=PERMUTATION_COUNT,
        level=LEVEL,
        datasets_per_batch=10,
    )
    return float(power)


def run_power_computation(computation: Computation) -> ComputationResult:
    """One computation: (pair name, dimension, exponent index)."""
    pair_name, dimension, exponent_index = (
        str(computation[0]),
        int(computation[1]),
        int(computation[2]),
    )
    return {"power": estimate_power(pair_name, dimension, exponent_index)}


def main() -> None:
    """Compute every power and write dimension_tradeoff.npz with the heatmaps."""
    parser = argparse.ArgumentParser(description="Compute the tradeoff in higher dimensions.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # One computation per (pair, dimension, exponent): 2 × 7 × 50 computations
    computations: list[Computation] = [
        (pair_name, dimension, index)
        for pair_name in PAIR_NAMES
        for dimension in DIMENSIONS
        for index in range(len(EXPONENTS))
    ]
    results = run_computations_in_parallel(
        computations,
        run_power_computation,
        arguments.output / "tradeoff_experiments_higher_dimensions.jsonl",
        arguments.processes,
    )
    # Contraction factor γ^α of every kernel at every discount, shape (exponents, discounts)
    thresholds = DISCOUNTS[None, :] ** EXPONENTS[:, None]
    arrays: dict[str, np.ndarray] = {
        "patterns": np.asarray(PAIR_NAMES),
        "dimensions": np.asarray(DIMENSIONS),
        "exponents": EXPONENTS,
        "discounts": DISCOUNTS,
        "budgets": BUDGETS,
    }
    for pair_name in PAIR_NAMES:
        # Powers, one row per dimension, shape (dimensions, exponents)
        powers = np.asarray(
            [
                [results[(pair_name, d, i)]["power"] for i in range(len(EXPONENTS))]
                for d in DIMENSIONS
            ]
        )
        # Heatmaps of Algorithm 4, one per dimension, shape (dimensions, budgets, discounts)
        heatmaps = np.stack(
            [
                np.asarray(
                    build_tradeoff_heatmap(
                        jnp.asarray(thresholds), jnp.asarray(row), jnp.asarray(BUDGETS)
                    )
                )
                for row in powers
            ]
        )
        arrays[f"{pair_name}__powers"] = powers
        arrays[f"{pair_name}__heatmaps"] = heatmaps
    np.savez(arguments.output / "dimension_tradeoff.npz", allow_pickle=False, **arrays)


if __name__ == "__main__":
    main()
