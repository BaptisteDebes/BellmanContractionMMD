"""Compute the tradeoff experiments of Figures 3, 16 and 17 (Appendix H) on this machine.

For each of the 4 kernel families of Table 11, each kernel of the family's grid and each of
the 7 tasks of Table 12: the contraction thresholds on the task's ball at 100 discounts (one
support of 512 points per discount, Algorithm 1) and the power of the permutation test
(1000 datasets, 1000 permutations, level 0.05). The heatmap of Algorithm 4 is built from
them at the end. Figure 16 is one permutation test (2000 permutations) on one checkerboard
dataset. Each threshold row, each power and the permutation example is one computation.

Usage:
    python experiments/tradeoff_experiments.py --output <folder> [--processes N]
    python paper_figures/tradeoff_figures.py --archives <folder> --output figures

Running time: about 46 CPU-hours.
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

from mmd_contraction import tradeoff_tasks
from mmd_contraction.finite_support import draw_contraction_thresholds
from mmd_contraction.kernels import (
    GaussianRBFKernel,
    GeneralizedMultiquadricKernel,
    LogarithmicKernel,
    PowerDistanceKernel,
    RadialKernel,
)
from mmd_contraction.permutation_test import estimate_test_power, run_permutation_test
from mmd_contraction.tradeoff import build_tradeoff_heatmap

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

FAMILY_GRIDS = {  # family → its parameter grid (Table 11)
    "rbf": np.geomspace(1e-3, 1e3, 100),  # ℓ in exp(−‖x − y‖²/ℓ²)
    "log": np.geomspace(1e-3, 1e3, 100),  # a in −log(1 + ‖x − y‖²/a)
    "gmq": np.linspace(0.05, 1.0, 50),  # β in −(1 + ‖x − y‖²)^β
    "power_distance": np.linspace(0.2, 2.0, 50),  # α in −‖x − y‖^α
}
TASK_NAMES = (  # Table 12, in its order
    "MEAN_SHIFT_1D",
    "SAME_MEAN_DIFFERENT_STD_1D",
    "BIMODAL_VS_UNIMODAL_1D",
    "TWO_GAUSSIAN_LOCATION",
    "COVARIANCE_SWAP",
    "BIMODAL_VS_UNIMODAL",
    "CHECKERBOARD",
)
DISCOUNTS = np.linspace(0.01, 1.0, 100)  # γ
BUDGETS = np.linspace(0.01, 1.0, 100)  # τ̄
THRESHOLD_SEED_BASE = 20260550  # + task index, split into one key per kernel of the grid
POWER_SEED_BASE = 20260650  # + task index, the same for every kernel and family
EXAMPLE_SEEDS = (20260434, 20260435, 20260436, 20260430, 20260431, 20260433, 20260432)
SUPPORT_SIZE = 512
RELATIVE_EIGENVALUE_TOLERANCE = 1e-9
DIRECTION_NORM_FLOOR = 1e-12
DATASET_COUNT = 1000
PERMUTATION_COUNT = 1000
LEVEL = 0.05
PERMUTATION_EXAMPLE_SEED = 20260321
PERMUTATION_EXAMPLE_SAMPLE_SIZE = 70  # points per sample, not moved onto the ball
PERMUTATION_EXAMPLE_LENGTHSCALE = 1.05
PERMUTATION_EXAMPLE_PERMUTATION_COUNT = 2000


def build_kernel(family: str, value: float) -> RadialKernel:
    """The kernel of a family at one value of its parameter.

    Args:
        family: "rbf", "log", "gmq" or "power_distance".
        value: ℓ, a, β or α.

    Returns:
        The kernel.
    """
    if family == "rbf":
        return GaussianRBFKernel(lengthscale=value)
    if family == "log":
        return LogarithmicKernel(scale=value)
    if family == "gmq":
        return GeneralizedMultiquadricKernel(scale=1.0, exponent=value)
    return PowerDistanceKernel(exponent=value)


def compute_thresholds(family: str, task_index: int, parameter_index: int) -> list[float]:
    """τ⋆ of one kernel on one task's ball at every discount, one support per discount.

    Args:
        family: The kernel family.
        task_index: The position of the task in TASK_NAMES.
        parameter_index: The position of the kernel in the family's grid.

    Returns:
        τ⋆ at each discount, (discounts,).
    """
    task = getattr(tradeoff_tasks, TASK_NAMES[task_index])
    grid = FAMILY_GRIDS[family]
    # One key per kernel of the grid, shape (parameters,)
    parameter_keys = jax.random.split(jax.random.key(THRESHOLD_SEED_BASE + task_index), len(grid))
    # τ⋆ at each discount for the one support, shape (discounts, 1)
    thresholds = draw_contraction_thresholds(
        parameter_keys[parameter_index],
        build_kernel(family, float(grid[parameter_index])),
        jnp.asarray(DISCOUNTS),
        radius=task.radius,
        dimension=len(task.first_mixture.means[0]),
        support_size=SUPPORT_SIZE,
        support_count=1,
        relative_eigenvalue_tolerance=RELATIVE_EIGENVALUE_TOLERANCE,
        supports_per_batch=1,
        discounts_per_batch=10,
        direction_norm_floor=DIRECTION_NORM_FLOOR,
    )
    return np.asarray(thresholds)[:, 0].tolist()


def estimate_power(family: str, task_index: int, parameter_index: int) -> float:
    """The power of the permutation test of one kernel on one task.

    Args:
        family: The kernel family.
        task_index: The position of the task in TASK_NAMES.
        parameter_index: The position of the kernel in the family's grid.

    Returns:
        The fraction of the DATASET_COUNT datasets on which the test rejects at LEVEL.
    """
    power = estimate_test_power(
        jax.random.key(POWER_SEED_BASE + task_index),
        build_kernel(family, float(FAMILY_GRIDS[family][parameter_index])),
        getattr(tradeoff_tasks, TASK_NAMES[task_index]),
        dataset_count=DATASET_COUNT,
        permutation_count=PERMUTATION_COUNT,
        level=LEVEL,
        datasets_per_batch=10,
    )
    return float(power)


def run_permutation_example() -> ComputationResult:
    """The permutation test of Figure 16: RBF ℓ = 1.05 on one checkerboard dataset.

    Returns:
        The two samples, the statistic, the statistics of the permutations and the p-value.
    """
    task = tradeoff_tasks.CHECKERBOARD
    first_key, second_key, shuffling_key = jax.random.split(
        jax.random.key(PERMUTATION_EXAMPLE_SEED), 3
    )
    # The two samples, each (70, 2), drawn without moving them onto the ball
    first = task.first_mixture.draw_points(first_key, PERMUTATION_EXAMPLE_SAMPLE_SIZE)
    second = task.second_mixture.draw_points(second_key, PERMUTATION_EXAMPLE_SAMPLE_SIZE)
    result = run_permutation_test(
        shuffling_key,
        GaussianRBFKernel(lengthscale=PERMUTATION_EXAMPLE_LENGTHSCALE),
        first,
        second,
        permutation_count=PERMUTATION_EXAMPLE_PERMUTATION_COUNT,
    )
    return {
        "first": np.asarray(first).tolist(),
        "second": np.asarray(second).tolist(),
        "statistic": float(result.statistic),
        "null_statistics": np.asarray(result.null_statistics).tolist(),
        "p_value": float(result.p_value),
    }


def run_tradeoff_computation(computation: Computation) -> ComputationResult:
    """One computation: ("thresholds" or "power", family, task index, parameter index), or
    ("permutation_example",).

    Each computation compiles programs for its own kernel and task; they are cleared after it
    to keep the memory of a process bounded.
    """
    kind = str(computation[0])
    if kind == "permutation_example":
        result = run_permutation_example()
    else:
        family, task_index, parameter_index = (
            str(computation[1]),
            int(computation[2]),
            int(computation[3]),
        )
        if kind == "thresholds":
            result = {"thresholds": compute_thresholds(family, task_index, parameter_index)}
        else:
            result = {"power": estimate_power(family, task_index, parameter_index)}
    jax.clear_caches()
    return result


def main() -> None:
    """Compute everything and write tradeoff__<family>.npz and figure_14_permutation_null.npz."""
    parser = argparse.ArgumentParser(description="Compute the tradeoff experiments.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # One computation per (kind, family, task, kernel), and the permutation example
    computations: list[Computation] = [
        (kind, family, task_index, parameter_index)
        for family, grid in FAMILY_GRIDS.items()
        for parameter_index in range(len(grid))
        for task_index in range(len(TASK_NAMES))
        for kind in ("power", "thresholds")
    ]
    computations.append(("permutation_example",))
    results = run_computations_in_parallel(
        computations,
        run_tradeoff_computation,
        arguments.output / "tradeoff_experiments.jsonl",
        arguments.processes,
    )
    for family, grid in FAMILY_GRIDS.items():
        arrays: dict[str, np.ndarray] = {
            "task_order": np.asarray(TASK_NAMES),
            "parameter_values": grid,
            "discounts": DISCOUNTS,
            "budgets": BUDGETS,
        }
        for task_index, task_name in enumerate(TASK_NAMES):
            task = getattr(tradeoff_tasks, task_name)
            # τ⋆ for each kernel and discount, shape (parameters, discounts)
            thresholds = np.asarray(
                [
                    results[("thresholds", family, task_index, i)]["thresholds"]
                    for i in range(len(grid))
                ]
            )
            # Power of each kernel, shape (parameters,)
            powers = np.asarray(
                [results[("power", family, task_index, i)]["power"] for i in range(len(grid))]
            )
            # Best power under τ⋆ ≤ τ̄ for each budget and discount, shape (budgets, discounts)
            heatmap = build_tradeoff_heatmap(
                jnp.asarray(thresholds), jnp.asarray(powers), jnp.asarray(BUDGETS)
            )
            # One dataset of the task, for the left column of the figures
            first, second = task.draw_samples(jax.random.key(EXAMPLE_SEEDS[task_index]))
            prefix = task_name.lower()
            arrays[f"{prefix}__thresholds"] = thresholds
            arrays[f"{prefix}__powers"] = powers
            arrays[f"{prefix}__heatmap"] = np.asarray(heatmap)
            arrays[f"{prefix}__example_first_sample"] = np.asarray(first)
            arrays[f"{prefix}__example_second_sample"] = np.asarray(second)
            arrays[f"{prefix}__radius"] = np.asarray(task.radius)
        np.savez(arguments.output / f"tradeoff__{family}.npz", allow_pickle=False, **arrays)
    example = results[("permutation_example",)]
    np.savez(
        arguments.output / "figure_14_permutation_null.npz",
        allow_pickle=False,
        first_sample=np.asarray(example["first"]),
        second_sample=np.asarray(example["second"]),
        statistic=np.asarray(example["statistic"]),
        null_statistics=np.asarray(example["null_statistics"]),
        p_value=np.asarray(example["p_value"]),
        lengthscale=np.asarray(PERMUTATION_EXAMPLE_LENGTHSCALE),
        level=np.asarray(LEVEL),
    )


if __name__ == "__main__":
    main()
