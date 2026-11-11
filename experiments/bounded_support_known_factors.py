"""Compute the estimates on balls of Figure 4 (Appendix D.4) on this machine.

For six kernels whose smallest contraction factor across dimensions is known, the generalized
multiquadric −(1 + ‖x − y‖²)^β with β = 0.25, 0.5, 0.75 and the power distance −‖x − y‖^α with
α = 0.5, 1, 1.5, and for six radii R from 0.1 to 10⁴, draw one random support of 256 points in
the ball B(0, R) at each of 80 discounts λ from 0 to 0.99, in dimension 1, and compute its
contraction threshold (Algorithm 1). Each (kernel, radius) is one computation.

Usage:
    python experiments/bounded_support_known_factors.py --output <folder> [--processes N]
    python paper_figures/bounded_support_known_factors_figure.py \
        --archive <folder>/bounded_support_known_factors.npz --output figures

Running time: about 2 CPU-minutes (31 seconds on 4 CPUs).
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from parallel import (
    Computation,
    ComputationResult,
    count_usable_cpus,
    run_computations_in_parallel,
)

from mmd_contraction.finite_support import draw_contraction_thresholds
from mmd_contraction.kernels import GeneralizedMultiquadricKernel, PowerDistanceKernel, RadialKernel

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

KERNELS = (  # (family, exponent), in the figure's order: β for "gmq", α for "power_distance"
    ("gmq", 0.25),
    ("gmq", 0.50),
    ("gmq", 0.75),
    ("power_distance", 0.50),
    ("power_distance", 1.00),
    ("power_distance", 1.50),
)
RADII = np.logspace(-1.0, 4.0, 6)  # R
DISCOUNTS = np.linspace(0.0, 0.99, 80)  # λ
SUPPORT_SIZE = 256
SUPPORT_COUNT = 1
SEED = 123  # folded in with the kernel index, then with the radius index
RELATIVE_EIGENVALUE_TOLERANCE = 1e-9
DIRECTION_NORM_FLOOR = 1e-12
ARCHIVE_NAME = "bounded_support_known_factors.npz"


def build_kernel(family: str, exponent: float) -> RadialKernel:
    """Build the kernel −(1 + ‖x − y‖²)^β (h = 1) for "gmq", or −‖x − y‖^α for "power_distance".

    Args:
        family: "gmq" or "power_distance".
        exponent: β or α.

    Returns:
        The kernel.
    """
    if family == "gmq":
        return GeneralizedMultiquadricKernel(scale=1.0, exponent=exponent)
    return PowerDistanceKernel(exponent=exponent)


def run_estimate_curve(computation: Computation) -> ComputationResult:
    """Compute one curve of Figure 4: (kernel index, radius index) → τ̂⋆ at each discount.

    Args:
        computation: (kernel index, radius index).

    Returns:
        {"thresholds": τ̂⋆ of the support drawn at each discount, (discounts,)}.
    """
    kernel_index, radius_index = int(computation[0]), int(computation[1])
    family, exponent = KERNELS[kernel_index]
    # Key of this curve: the seed folded in with the kernel index, then with the radius index
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(SEED), kernel_index), radius_index)
    # τ⋆ of the one support at every discount, shape (discounts, 1)
    thresholds = draw_contraction_thresholds(
        key,
        build_kernel(family, exponent),
        jnp.asarray(DISCOUNTS),
        radius=float(RADII[radius_index]),
        dimension=1,
        support_size=SUPPORT_SIZE,
        support_count=SUPPORT_COUNT,
        relative_eigenvalue_tolerance=RELATIVE_EIGENVALUE_TOLERANCE,
        supports_per_batch=1,
        discounts_per_batch=8,
        direction_norm_floor=DIRECTION_NORM_FLOOR,
    )
    return {"thresholds": np.asarray(thresholds)[:, 0].tolist()}


def main() -> None:
    """Compute the 6 × 6 curves and write bounded_support_known_factors.npz."""
    parser = argparse.ArgumentParser(description="Compute the estimates on balls of Figure 4.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # One computation per (kernel index, radius index): 6 × 6 computations
    computations: list[Computation] = [
        (kernel_index, radius_index)
        for kernel_index in range(len(KERNELS))
        for radius_index in range(len(RADII))
    ]
    results = run_computations_in_parallel(
        computations,
        run_estimate_curve,
        arguments.output / "bounded_support_known_factors.jsonl",
        arguments.processes,
    )
    # τ̂⋆ for each kernel, radius and discount, shape (kernels, radii, discounts)
    thresholds: Float[np.ndarray, "kernels radii discounts"] = np.asarray(
        [
            [
                results[(kernel_index, radius_index)]["thresholds"]
                for radius_index in range(len(RADII))
            ]
            for kernel_index in range(len(KERNELS))
        ]
    )
    np.savez(
        arguments.output / ARCHIVE_NAME,
        allow_pickle=False,
        families=np.asarray([family for family, _ in KERNELS]),
        exponents=np.asarray([exponent for _, exponent in KERNELS]),
        radii=RADII,
        discounts=DISCOUNTS,
        thresholds=thresholds,
        support_size=SUPPORT_SIZE,
        support_count=SUPPORT_COUNT,
        relative_eigenvalue_tolerance=RELATIVE_EIGENVALUE_TOLERANCE,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
