"""Permutation test of P = Q with the biased MMD² statistic, and its power (Appendix H.1).

For samples X_1, …, X_n from P and Y_1, …, Y_m from Q, pooled into Z_1, …, Z_N with
N = n + m, the statistic is T = ℓᵀKℓ, where K_ij = k(Z_i, Z_j) and ℓ_i = 1/n for the X's and
−1/m for the Y's: the biased estimate of MMD²_k(P, Q). Shuffling the labels gives the null
statistics T_π, and the test rejects P = Q when p̂ = (1 + #{T_π ≥ T}) / (B_perm + 1) ≤ α.
Its power is the probability of rejecting, estimated on fresh datasets (Algorithm 3).
"""

from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float64, PRNGKeyArray

from mmd_contraction.distances import compute_squared_distances
from mmd_contraction.kernels import RadialKernel
from mmd_contraction.precision import raise_unless_float64_enabled


@runtime_checkable
class TwoSampleProblem(Protocol):
    """A pair of distributions P, Q with the sample sizes n, m drawn from them.

    Implementations are hashable records of plain numbers (`jax.tree_util.register_static`),
    so that compiled functions can receive them, as they receive kernels.
    """

    @property
    def first_sample_size(self) -> int:
        """n, the number of points drawn from P."""
        ...

    @property
    def second_sample_size(self) -> int:
        """m, the number of points drawn from Q."""
        ...

    def draw_samples(
        self, key: PRNGKeyArray
    ) -> tuple[Float64[Array, "first dimension"], Float64[Array, "second dimension"]]:
        """Draw X_1, …, X_n from P and Y_1, …, Y_m from Q.

        Args:
            key: Random key for both samples.

        Returns:
            X_1, …, X_n: shape (n, d), and Y_1, …, Y_m: shape (m, d).
        """
        ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PermutationTestResult:
    """The statistic, its null statistics and the p-value of one permutation test."""

    statistic: Float64[Array, ""]
    """T = ℓᵀKℓ, the biased estimate of MMD²."""
    null_statistics: Float64[Array, " permutations"]
    """T_π for each of the B_perm shuffles of the labels, computed as T + (T_π − T)."""
    p_value: Float64[Array, ""]
    """p̂ = (1 + #{T_π ≥ T}) / (B_perm + 1), with T_π ≥ T read from the sign of T_π − T."""


def run_permutation_test(
    key: PRNGKeyArray,
    kernel: RadialKernel,
    first_sample: Float64[Array, "first dimension"],
    second_sample: Float64[Array, "second dimension"],
    permutation_count: int,
) -> PermutationTestResult:
    """Run the permutation test of P = Q on one pair of samples.

    Shuffling the labels of the pooled points is done by shuffling the entries of ℓ:
    ℓ_πᵀKℓ_π is the statistic of the pooled points relabelled by π.

    Whether T_π ≥ T is decided from the difference T_π − T = (ℓ_π − ℓ)ᵀK(ℓ_π + ℓ),
    computed without the diagonal of K, which adds the same k(0)(1/n + 1/m) to every
    statistic. A pair of points whose product ℓ_iℓ_j the shuffle leaves unchanged then adds
    exactly 0, so a shuffle that changes nothing gives exactly T_π − T = 0, even when the
    kernel values between points are far below the rounding error of T.

    Args:
        key: Random key for the B_perm shuffles.
        kernel: The radial kernel k.
        first_sample: X_1, …, X_n: shape (n, d).
        second_sample: Y_1, …, Y_m: shape (m, d).
        permutation_count: B_perm, the number of shuffles.

    Returns:
        T, the B_perm values T_π and p̂.

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    return _run_permutation_test_compiled(
        key, kernel, first_sample, second_sample, permutation_count=permutation_count
    )


@partial(jax.jit, static_argnames=("permutation_count",))
def _run_permutation_test_compiled(
    key: PRNGKeyArray,
    kernel: RadialKernel,
    first_sample: Float64[Array, "first dimension"],
    second_sample: Float64[Array, "second dimension"],
    *,
    permutation_count: int,
) -> PermutationTestResult:
    """Compute `run_permutation_test`, compiled once per kernel, sizes and B_perm.

    Args and returns: see `run_permutation_test`.
    """
    # Sample sizes n and m
    first_size, second_size = first_sample.shape[0], second_sample.shape[0]
    # Pooled points Z_1, …, Z_N, the X's then the Y's, shape (N, d)
    pooled = jnp.concatenate([first_sample, second_sample])
    # Gram matrix K, K_ij = k(Z_i, Z_j), shape (N, N)
    gram = kernel.evaluate_on_squared_distances(compute_squared_distances(pooled))
    # Sample weights ℓ: 1/n for the X's, −1/m for the Y's, shape (N,)
    sample_weights: Float64[Array, " pooled"] = jnp.concatenate(
        [
            jnp.full((first_size,), 1.0 / first_size, dtype=jnp.float64),
            jnp.full((second_size,), -1.0 / second_size, dtype=jnp.float64),
        ]
    )
    # Statistic T = ℓᵀKℓ, the biased estimate of MMD², shape ()
    statistic = sample_weights @ gram @ sample_weights
    # One key per shuffle, shape (B_perm,)
    permutation_keys = jax.random.split(key, permutation_count)
    # Shuffled weights ℓ_π: the entries of ℓ in shuffled order, one row per shuffle,
    # shape (B_perm, N)
    shuffled_weights = jax.vmap(
        lambda permutation_key: jax.random.permutation(permutation_key, sample_weights)
    )(permutation_keys)
    # Gram matrix without its diagonal, which adds k(0)(1/n + 1/m) to T and to every T_π,
    # shape (N, N)
    off_diagonal_gram = jnp.fill_diagonal(gram, 0.0, inplace=False)
    # Differences T_π − T = (ℓ_π − ℓ)ᵀK(ℓ_π + ℓ), computed as one sum, shape (B_perm,)
    differences = jnp.sum(
        ((shuffled_weights - sample_weights) @ off_diagonal_gram)
        * (shuffled_weights + sample_weights),
        axis=1,
    )
    # Null statistics T_π = T + (T_π − T), shape (B_perm,)
    null_statistics = statistic + differences
    # Number of shuffles with T_π ≥ T, read from the sign of the differences, shape ()
    exceed_count = jnp.sum(differences >= 0.0)
    # p-value p̂ = (1 + #{T_π ≥ T}) / (B_perm + 1), shape ()
    p_value = (1.0 + exceed_count) / (permutation_count + 1.0)
    return PermutationTestResult(statistic, null_statistics, p_value)


def estimate_test_power(
    key: PRNGKeyArray,
    kernel: RadialKernel,
    problem: TwoSampleProblem,
    dataset_count: int,
    permutation_count: int,
    level: float,
    datasets_per_batch: int,
) -> Float64[Array, ""]:
    """Estimate the power of the level-α permutation test against (P, Q) (Algorithm 3).

    The key is split into one key per dataset, and each of those into a key for the samples
    and a key for the shuffles. The same key therefore gives the same datasets and the same
    shuffles whatever the kernel, so kernels estimated with one key are compared on the
    same data.

    Args:
        key: Random key for all datasets and shuffles.
        kernel: The radial kernel k.
        problem: The pair (P, Q) and the sample sizes n, m.
        dataset_count: B, the number of fresh datasets.
        permutation_count: B_perm, the number of shuffles per dataset.
        level: α, the level of the test.
        datasets_per_batch: How many datasets are computed together; changes memory use,
            not the datasets drawn.

    Returns:
        The fraction of the B datasets on which p̂ ≤ α: shape ().

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    return _estimate_test_power_compiled(
        key,
        kernel,
        problem,
        jnp.asarray(level, dtype=jnp.float64),
        dataset_count=dataset_count,
        permutation_count=permutation_count,
        datasets_per_batch=datasets_per_batch,
    )


@partial(jax.jit, static_argnames=("dataset_count", "permutation_count", "datasets_per_batch"))
def _estimate_test_power_compiled(
    key: PRNGKeyArray,
    kernel: RadialKernel,
    problem: TwoSampleProblem,
    level: Float64[Array, ""],
    *,
    dataset_count: int,
    permutation_count: int,
    datasets_per_batch: int,
) -> Float64[Array, ""]:
    """Compute `estimate_test_power`, compiled once per kernel, problem and settings.

    Args and returns: see `estimate_test_power`; `level` is an array here.
    """
    # One key per dataset, shape (B,)
    dataset_keys = jax.random.split(key, dataset_count)

    def test_one_dataset(dataset_key: PRNGKeyArray) -> Bool[Array, ""]:
        """Draw one dataset and test it.

        Args:
            dataset_key: This dataset's key.

        Returns:
            Whether the test rejects P = Q on it: shape ().
        """
        # Key for the samples and key for the shuffles
        sampling_key, shuffling_key = jax.random.split(dataset_key)
        # Samples X_1, …, X_n and Y_1, …, Y_m, shapes (n, d) and (m, d)
        first_sample, second_sample = problem.draw_samples(sampling_key)
        # Permutation test on this dataset
        result = _run_permutation_test_compiled(
            shuffling_key, kernel, first_sample, second_sample, permutation_count=permutation_count
        )
        # Rejection of P = Q when p̂ ≤ α, shape ()
        return result.p_value <= level

    # Rejection flag of each dataset, datasets_per_batch datasets at a time, shape (B,)
    rejections = jax.lax.map(test_one_dataset, dataset_keys, batch_size=datasets_per_batch)
    # Power estimate: the fraction of rejections, shape ()
    return jnp.mean(rejections.astype(jnp.float64))
