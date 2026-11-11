"""Compute the contraction thresholds on a ball of Figure 2 on this machine.

For the RBF kernel exp(−‖x − y‖²/ℓ²) and the logarithmic kernel −log(1 + ‖x − y‖²/ℓ²), at 81
ratios R/ℓ from 0.05 to 2 (R = 1) and 81 discounts γ from 0 to 1, draw 10 random supports of
512 points in the ball of radius 1 and keep the largest contraction threshold (Algorithm 1).
Each (kernel, ratio) is one computation.

Usage:
    python experiments/bounded_support_heatmaps.py --output <folder> [--processes N]
    python paper_figures/bounded_support_heatmaps.py --archives <folder> --output figures

Running time: about 7 CPU-hours (12 minutes on 32 CPUs).
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

from mmd_contraction.finite_support import draw_contraction_thresholds
from mmd_contraction.kernels import GaussianRBFKernel, LogarithmicKernel, RadialKernel

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

KERNEL_NAMES = ("rbf", "log")
RATIOS = np.linspace(0.05, 2.0, 81)  # R/ℓ, with R = 1
DISCOUNTS = np.linspace(0.0, 1.0, 81)  # γ
RADIUS = 1.0
SUPPORT_SIZE = 512
SUPPORT_COUNT = 10
SEED = 321  # split into one key per ratio, the same for both kernels
RELATIVE_EIGENVALUE_TOLERANCE = 1e-9
DIRECTION_NORM_FLOOR = 1e-12
REJECTION_TOLERANCE = 1e-6  # stored for the drawing: thresholds above 1 + this are hatched


def build_kernel(kernel_name: str, ratio: float) -> RadialKernel:
    """exp(−‖x − y‖²/ℓ²) or −log(1 + ‖x − y‖²/ℓ²), with ℓ = R/ratio and R = 1.

    Args:
        kernel_name: "rbf" or "log".
        ratio: R/ℓ.

    Returns:
        The kernel.
    """
    # Lengthscale ℓ = R/ratio
    lengthscale = RADIUS / ratio
    if kernel_name == "rbf":
        return GaussianRBFKernel(lengthscale=lengthscale)
    # −log(1 + ‖x − y‖²/a) with a = ℓ²
    return LogarithmicKernel(scale=lengthscale**2)


def run_heatmap_row(computation: Computation) -> ComputationResult:
    """One row of a heatmap: (kernel name, ratio index) → the largest threshold at each γ.

    Args:
        computation: (kernel name, ratio index).

    Returns:
        {"largest_thresholds": max over the 10 supports of τ⋆ at each discount, (discounts,)}.
    """
    kernel_name, ratio_index = str(computation[0]), int(computation[1])
    # One key per ratio, shape (ratios,)
    ratio_keys = jax.random.split(jax.random.key(SEED), len(RATIOS))
    # τ⋆ of every support at every discount, shape (discounts, supports)
    thresholds = draw_contraction_thresholds(
        ratio_keys[ratio_index],
        build_kernel(kernel_name, float(RATIOS[ratio_index])),
        jnp.asarray(DISCOUNTS),
        radius=RADIUS,
        dimension=1,
        support_size=SUPPORT_SIZE,
        support_count=SUPPORT_COUNT,
        relative_eigenvalue_tolerance=RELATIVE_EIGENVALUE_TOLERANCE,
        supports_per_batch=10,
        discounts_per_batch=9,
        direction_norm_floor=DIRECTION_NORM_FLOOR,
    )
    # Largest τ⋆ over the supports at each discount, shape (discounts,)
    return {"largest_thresholds": np.asarray(thresholds).max(axis=1).tolist()}


def main() -> None:
    """Compute both heatmaps and write <kernel>_fixed_support_heatmap_data.npz for each."""
    parser = argparse.ArgumentParser(description="Compute the heatmaps of Figure 2.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # One computation per (kernel, ratio index): 2 × 81 computations
    computations: list[Computation] = [
        (kernel_name, index) for kernel_name in KERNEL_NAMES for index in range(len(RATIOS))
    ]
    results = run_computations_in_parallel(
        computations,
        run_heatmap_row,
        arguments.output / "bounded_support_heatmaps.jsonl",
        arguments.processes,
    )
    for kernel_name in KERNEL_NAMES:
        # Largest τ⋆ for each ratio and discount, shape (ratios, discounts)
        grid = np.asarray(
            [results[(kernel_name, index)]["largest_thresholds"] for index in range(len(RATIOS))]
        )
        # The names the drawing script reads: λ for γ, N for the support size, J for the count
        np.savez(
            arguments.output / f"{kernel_name}_fixed_support_heatmap_data.npz",
            allow_pickle=False,
            lambda_values=DISCOUNTS,
            ratio_values=RATIOS,
            tau_grid=grid,
            rejection_tol=REJECTION_TOLERANCE,
            eig_tol=RELATIVE_EIGENVALUE_TOLERANCE,
            R=RADIUS,
            N=SUPPORT_SIZE,
            J=SUPPORT_COUNT,
            key_seed=SEED,
        )


if __name__ == "__main__":
    main()
