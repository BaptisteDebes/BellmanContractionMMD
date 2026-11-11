"""Compute Table 9 (Appendix G.2), the contraction of truncated chains on a ball, on this machine.

The two chains are those of Figures 14 and 15: the chained RBF kernel with γ = 0.95, a = c = 1
and τ = 0.99275 or 0.911525. For each chain and each number of terms K of the table's grid (1
to 40, then 50, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000 and 1500), and for the full
chain with 2500 terms, compute the contraction threshold at γ of 10 sets of 512 points drawn in
the ball of radius 1, in dimension 1 (Algorithm 1). The sets are the same for every K and both
chains. Each (chain, K) is one computation. Then, for each chain, compute the share R_K^(N)(t)
of the omitted terms with N = 5000, for every K, at 400 squared distances t up to 4, and print
the table's four rows.

Usage:
    python experiments/truncated_chain_contraction.py --output <folder> [--processes N]

Running time: about 0.2 CPU-hours (3 minutes on 4 CPUs).
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Int
from parallel import (
    Computation,
    ComputationResult,
    count_usable_cpus,
    run_computations_in_parallel,
)

from mmd_contraction.finite_support import draw_contraction_thresholds
from mmd_contraction.kernels import TruncatedChainedRBFKernel

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

DISCOUNT = 0.95  # γ, of the chains and of the thresholds
TARGETS = (0.99275, 0.911525)  # τ of the two chains, in the table's column order
SCALE = 1.0  # a
WEIGHT = 1.0  # c
TESTED_TERM_COUNTS = (*range(1, 41), 50, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000, 1500)
FULL_CHAIN_TERM_COUNT = 2500
SHARE_LAST_INDEX = 5000  # N of R_K^(N)(t)
RADIUS = 1.0
DIMENSION = 1
SUPPORT_SIZE = 512
SUPPORT_COUNT = 10
SEED = 321  # the same key for every computation, so the same sets of points
RELATIVE_EIGENVALUE_TOLERANCE = 1e-9
DIRECTION_NORM_FLOOR = 1e-12
LEVEL = 1e-3  # the 10⁻³ of the table's last two rows
ARCHIVE_NAME = "truncated_chain_contraction.json"


def build_chain(target: float, term_count: int) -> TruncatedChainedRBFKernel:
    """Build the chain of the table with target factor τ, truncated to its first K terms.

    Args:
        target: τ.
        term_count: K.

    Returns:
        The truncated chained RBF kernel with γ = 0.95 and a = c = 1.
    """
    return TruncatedChainedRBFKernel(
        discount=DISCOUNT,
        target_contraction_factor=target,
        scale=SCALE,
        weight=WEIGHT,
        term_count=term_count,
    )


def run_threshold_computation(computation: Computation) -> ComputationResult:
    """Compute the thresholds of one truncated chain: (chain index, K) → its 10 thresholds.

    Args:
        computation: (chain index, number of terms K).

    Returns:
        {"thresholds": the contraction threshold at γ of each of the 10 sets of points, (10,)}.
    """
    chain_index, term_count = int(computation[0]), int(computation[1])
    # Thresholds at γ of the 10 sets of points, shape (1, 10)
    thresholds = draw_contraction_thresholds(
        jax.random.key(SEED),
        build_chain(TARGETS[chain_index], term_count),
        jnp.asarray([DISCOUNT]),
        radius=RADIUS,
        dimension=DIMENSION,
        support_size=SUPPORT_SIZE,
        support_count=SUPPORT_COUNT,
        relative_eigenvalue_tolerance=RELATIVE_EIGENVALUE_TOLERANCE,
        supports_per_batch=SUPPORT_COUNT,
        discounts_per_batch=1,
        direction_norm_floor=DIRECTION_NORM_FLOOR,
    )
    return {"thresholds": np.asarray(thresholds)[0].tolist()}


def find_first_tested_term_count(satisfied: Bool[np.ndarray, " counts"]) -> int:
    """Find the first number of terms of the table's grid where a condition holds.

    Args:
        satisfied: Whether the condition holds at each K of TESTED_TERM_COUNTS: shape (counts,).

    Returns:
        The smallest tested K where it holds.

    Raises:
        ValueError: If it holds at no tested K.
    """
    # Positions in the grid where the condition holds
    positions = np.flatnonzero(satisfied)
    if positions.size == 0:
        raise ValueError("the condition holds at no tested number of terms")
    return TESTED_TERM_COUNTS[int(positions[0])]


def find_first_term_count_reaching_level(
    largest_shares: Float[np.ndarray, " indices"], level: float
) -> int:
    """Find the smallest K whose largest share of the omitted terms is at most the level.

    Args:
        largest_shares: max over t of R_K^(N)(t), for K = 0, …, N: shape (N + 1,).
        level: The level, 10⁻³ in the table.

    Returns:
        The smallest K with max_t R_K^(N)(t) ≤ level.

    Raises:
        ValueError: If no K ≤ N reaches the level.
    """
    # Truncation indices K whose largest share is at most the level
    reached: Int[np.ndarray, " reached"] = np.flatnonzero(largest_shares <= level)
    if reached.size == 0:
        raise ValueError("no truncation index up to N reaches the level")
    return int(reached[0])


@dataclass(frozen=True)
class TableColumn:
    """One column of Table 9: one chain, its four rows, and the values they are read from."""

    target: float
    """τ of the chain."""
    full_chain_estimate: float
    """Row 1: the largest threshold of the full chain over the 10 sets of points."""
    term_count_passing_the_test: int
    """Row 2: the first tested K whose largest threshold is below 1."""
    term_count_within_level_of_full_chain: int
    """Row 3: the first tested K whose largest threshold is within 10⁻³ of row 1."""
    term_count_with_shares_at_most_level: int
    """Row 4: the smallest K with max_t R_K^(N)(t) ≤ 10⁻³."""
    thresholds: list[list[float]]
    """The 10 thresholds of each tested K, then of the full chain: (53, 10)."""
    largest_shares: list[float]
    """max_t R_K^(N)(t) for K = 0, …, N: (N + 1,)."""


def compute_table_column(
    target: float,
    thresholds: Float[np.ndarray, "counts supports"],
    squared_distances: Float[jax.Array, " distances"],
) -> TableColumn:
    """Compute one column of Table 9 from the chain's thresholds and its shares.

    Args:
        target: τ of the chain.
        thresholds: The 10 thresholds of each tested K, then of the full chain: shape (53, 10).
        squared_distances: Squared distances t at which the shares are measured: shape (T,).

    Returns:
        The column, with the values its rows are read from.
    """
    # Estimate for each tested K, then for the full chain: the largest of the 10 thresholds,
    # shape (53,)
    estimates: Float[np.ndarray, " counts"] = thresholds.max(axis=1)
    full_chain_estimate = float(estimates[-1])
    # Largest share max_t R_K^(N)(t) for K = 0, …, N, shape (N + 1,)
    largest_shares: Float[np.ndarray, " indices"] = np.asarray(
        build_chain(target, SHARE_LAST_INDEX + 1).compute_omitted_shares(squared_distances)
    ).max(axis=1)
    return TableColumn(
        target=target,
        full_chain_estimate=full_chain_estimate,
        term_count_passing_the_test=find_first_tested_term_count(estimates[:-1] < 1.0),
        term_count_within_level_of_full_chain=find_first_tested_term_count(
            np.abs(estimates[:-1] - full_chain_estimate) <= LEVEL
        ),
        term_count_with_shares_at_most_level=find_first_term_count_reaching_level(
            largest_shares, LEVEL
        ),
        thresholds=thresholds.tolist(),
        largest_shares=largest_shares.tolist(),
    )


def main() -> None:
    """Compute the thresholds and shares, write truncated_chain_contraction.json, print Table 9."""
    parser = argparse.ArgumentParser(description="Compute Table 9, truncated chains on a ball.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    parser.add_argument(
        "--processes", type=int, default=count_usable_cpus(), help="default: all usable CPUs"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    # The tested K, then the full chain: 53 numbers of terms
    term_counts = (*TESTED_TERM_COUNTS, FULL_CHAIN_TERM_COUNT)
    # One computation per (chain index, K): 2 × 53 computations
    computations: list[Computation] = [
        (chain_index, term_count)
        for chain_index in range(len(TARGETS))
        for term_count in term_counts
    ]
    results = run_computations_in_parallel(
        computations,
        run_threshold_computation,
        arguments.output / "truncated_chain_contraction.jsonl",
        arguments.processes,
    )
    # Squared distances t at which the shares are measured: 400 geometric values from
    # 10⁻¹² (2R)² to (2R)², the squared diameter of the ball, shape (400,)
    squared_distances = jnp.geomspace(1e-12 * (2.0 * RADIUS) ** 2, (2.0 * RADIUS) ** 2, 400)
    columns = [
        compute_table_column(
            target,
            np.asarray(
                [results[(chain_index, term_count)]["thresholds"] for term_count in term_counts]
            ),
            squared_distances,
        )
        for chain_index, target in enumerate(TARGETS)
    ]
    settings = {
        "discount": DISCOUNT,
        "scale": SCALE,
        "weight": WEIGHT,
        "tested_term_counts": list(TESTED_TERM_COUNTS),
        "full_chain_term_count": FULL_CHAIN_TERM_COUNT,
        "share_last_index": SHARE_LAST_INDEX,
        "radius": RADIUS,
        "dimension": DIMENSION,
        "support_size": SUPPORT_SIZE,
        "support_count": SUPPORT_COUNT,
        "seed": SEED,
        "relative_eigenvalue_tolerance": RELATIVE_EIGENVALUE_TOLERANCE,
        "direction_norm_floor": DIRECTION_NORM_FLOOR,
        "level": LEVEL,
    }
    (arguments.output / ARCHIVE_NAME).write_text(
        json.dumps(
            {"settings": settings, "columns": [asdict(column) for column in columns]}, indent=1
        )
        + "\n"
    )
    print("Table 9, columns τ = " + ", ".join(str(column.target) for column in columns))
    print(
        "Estimated smallest contraction factor of the full chain: "
        + ", ".join(f"{column.full_chain_estimate:.3f}" for column in columns)
    )
    print(
        "Terms needed to pass the test: "
        + ", ".join(str(column.term_count_passing_the_test) for column in columns)
    )
    print(
        "Terms needed to be within 1e-3 of the full chain: "
        + ", ".join(str(column.term_count_within_level_of_full_chain) for column in columns)
    )
    print(
        "Terms needed for R_K^(N)(t) <= 1e-3 on the ball: "
        + ", ".join(str(column.term_count_with_shares_at_most_level) for column in columns)
    )


if __name__ == "__main__":
    main()
