"""Contraction thresholds of finite supports: Proposition 5.2 and Algorithm 1 of the paper.

For a finite support S = {z_1, …, z_N}, a radial kernel k and a discount γ, the contraction
threshold τ⋆_S(γ) is the smallest τ for which τA − B is positive semidefinite, where A and
B are the Gram matrices of S and of γS restricted to zero-sum weights. Algorithm 1 draws
random supports in a ball and computes this threshold for each.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64, Key, PRNGKeyArray

from mmd_contraction.distances import compute_squared_distances
from mmd_contraction.kernels import RadialKernel
from mmd_contraction.precision import raise_unless_float64_enabled


def build_zero_sum_basis(support_size: int) -> Float64[Array, "points basis"]:
    """Build an orthonormal basis Q of the zero-sum weights 1_N^⊥ (Proposition 5.2).

    Args:
        support_size: Number of support points N.

    Returns:
        Q, whose orthonormal columns span {w ∈ R^N : 1ᵀw = 0}: shape (N, N − 1).

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    # Vector 1_N, shape (N, 1)
    ones = np.ones((support_size, 1), dtype=np.float64)
    # Centering projector H_N = I − 11ᵀ/N, whose columns span 1_N^⊥, shape (N, N)
    centering = np.eye(support_size, dtype=np.float64) - (ones @ ones.T) / float(support_size)
    # Drop one redundant column; the others still span 1_N^⊥, shape (N, N − 1)
    spanning_columns = centering[:, 1:]
    # Orthonormal columns with the same span, Q, shape (N, N − 1)
    basis, _ = np.linalg.qr(spanning_columns)
    # Q as a JAX array, shape (N, N − 1)
    return jnp.asarray(basis, dtype=jnp.float64)


def compute_contraction_threshold(
    support: Float64[Array, "points dimension"],
    kernel: RadialKernel,
    discount: Float64[Array, ""],
    zero_sum_basis: Float64[Array, "points basis"],
    relative_eigenvalue_tolerance: float,
) -> Float64[Array, ""]:
    """Compute the contraction threshold τ̂⋆_S(γ) of one finite support (Proposition 5.2).

    With K_ij = k(z_i, z_j), (K_γ)_ij = k(γz_i, γz_j), A = QᵀKQ and B = QᵀK_γQ, the
    threshold is the largest generalized eigenvalue of (B, A). As in Algorithm 1, A is
    eigendecomposed as P diag(μ) Pᵀ, only the directions with μ_i > ε max_i μ_i are kept,
    and the result is the largest eigenvalue of C_ε = Δ⁻¹PᵀBPΔ⁻¹, where Δ⁻¹ is diagonal
    with μ_i^(−1/2) on the kept directions and 0 elsewhere.

    Args:
        support: The support points z_1, …, z_N: shape (N, d).
        kernel: The radial kernel k.
        discount: γ, the factor of the scaling map z ↦ γz: shape ().
        zero_sum_basis: Q from `build_zero_sum_basis(N)`: shape (N, N − 1).
        relative_eigenvalue_tolerance: ε. Directions of A with μ_i ≤ ε max_i μ_i are
            dropped, negative eigenvalues included.

    Returns:
        τ̂⋆_S(γ): shape (). It equals τ⋆_S(γ) when no direction is dropped.

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    # Points γz_1, …, γz_N, shape (N, d)
    contracted_support = discount * support
    # Gram matrix K, K_ij = k(z_i, z_j), shape (N, N)
    gram = kernel.evaluate_on_squared_distances(compute_squared_distances(support))
    # Gram matrix K_γ, (K_γ)_ij = k(γz_i, γz_j), shape (N, N)
    contracted_gram = kernel.evaluate_on_squared_distances(
        compute_squared_distances(contracted_support)
    )

    # Restricted Gram matrix on 1_N^⊥, A = QᵀKQ, shape (N − 1, N − 1)
    restricted_gram = zero_sum_basis.T @ gram @ zero_sum_basis
    # Restricted Gram matrix on 1_N^⊥, B = QᵀK_γQ, shape (N − 1, N − 1)
    restricted_contracted_gram = zero_sum_basis.T @ contracted_gram @ zero_sum_basis
    # Enforce exact symmetry of A
    restricted_gram = 0.5 * (restricted_gram + restricted_gram.T)
    # Enforce exact symmetry of B
    restricted_contracted_gram = 0.5 * (restricted_contracted_gram + restricted_contracted_gram.T)

    # Eigendecomposition A = P diag(μ) Pᵀ
    decomposition = jnp.linalg.eigh(restricted_gram)
    # Eigenvalues μ_i of A, shape (N − 1,)
    eigenvalues: Float64[Array, " basis"] = decomposition.eigenvalues
    # Eigenvectors of A, the columns of P, shape (N − 1, N − 1)
    eigenvectors: Float64[Array, "basis basis"] = decomposition.eigenvectors
    # Relative cutoff ε max_i μ_i, shape ()
    cutoff = relative_eigenvalue_tolerance * jnp.max(eigenvalues)
    # Kept directions of A, those with μ_i > ε max_i μ_i, shape (N − 1,)
    kept = eigenvalues > cutoff
    # Truncated inverse square roots: μ_i^(−1/2) if kept, 0 if dropped, shape (N − 1,)
    # (the inner where keeps dropped, possibly negative, μ_i out of the square root)
    inverse_square_roots = jnp.where(kept, 1.0 / jnp.sqrt(jnp.where(kept, eigenvalues, 1.0)), 0.0)
    # Truncated inverse-square-root factor P diag(inverse_square_roots), shape (N − 1, N − 1)
    whitening = eigenvectors * inverse_square_roots[None, :]

    # Reduced symmetric matrix C_ε = whiteningᵀ B whitening, shape (N − 1, N − 1)
    reduced = whitening.T @ restricted_contracted_gram @ whitening
    # Enforce exact symmetry of C_ε
    reduced = 0.5 * (reduced + reduced.T)
    # Full symmetric eigensolver, then keep the largest eigenvalue: τ̂⋆_S(γ), shape ()
    return jnp.max(jnp.linalg.eigvalsh(reduced))


def draw_uniform_ball_support(
    key: PRNGKeyArray,
    dimension: int,
    support_size: int,
    radius: Float64[Array, ""],
    direction_norm_floor: float,
) -> Float64[Array, "points dimension"]:
    """Draw N points independently and uniformly in the ball B(0, R) of R^d.

    Each point is a uniform direction, a normalized standard Gaussian vector, times the
    radius R u^(1/d) with u uniform on [0, 1).

    Args:
        key: Random key; split into one key for the directions and one for the radii.
        dimension: d.
        support_size: N.
        radius: R: shape ().
        direction_norm_floor: Added to each Gaussian vector's norm before dividing, so
            that a zero vector cannot cause a division by zero.

    Returns:
        The points z_1, …, z_N: shape (N, d).

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    # Key for the directions and key for the radii
    direction_key, radius_key = jax.random.split(key, 2)
    # Standard Gaussian vectors, shape (N, d)
    gaussian = jax.random.normal(direction_key, shape=(support_size, dimension), dtype=jnp.float64)
    # Their norms, plus the floor that keeps the division finite, shape (N, 1)
    norms = jnp.linalg.norm(gaussian, axis=-1, keepdims=True) + direction_norm_floor
    # Uniform directions on the unit sphere, shape (N, d)
    directions = gaussian / norms
    # Uniform draws u in [0, 1), shape (N, 1)
    uniforms = jax.random.uniform(radius_key, shape=(support_size, 1), dtype=jnp.float64)
    # Exponent 1/d, shape ()
    exponent = jnp.asarray(1.0 / dimension, dtype=jnp.float64)
    # Radii R u^(1/d), which make the points uniform in the ball, shape (N, 1)
    radii = radius * jnp.power(uniforms, exponent)
    # Support points z_1, …, z_N, shape (N, d)
    return radii * directions


def draw_contraction_thresholds(
    key: PRNGKeyArray,
    kernel: RadialKernel,
    discounts: Float64[Array, " discounts"],
    radius: float,
    dimension: int,
    support_size: int,
    support_count: int,
    relative_eigenvalue_tolerance: float,
    supports_per_batch: int,
    discounts_per_batch: int,
    direction_norm_floor: float,
) -> Float64[Array, "discounts supports"]:
    """Draw J random supports for each discount and compute every threshold (Algorithm 1).

    The key is split into one key per discount and per support, `split(key, (L, J))`;
    support j of discount l is drawn by `draw_uniform_ball_support` with key [l, j] and its
    threshold computed by `compute_contraction_threshold`. Each discount therefore has its
    own J supports.
    Taking the maximum over the last axis gives the sampled worst case max_j τ̂⋆_{S_j}(γ).

    Args:
        key: Random key for all supports.
        kernel: The radial kernel k.
        discounts: The discounts γ_1, …, γ_L: shape (L,).
        radius: R, the radius of the ball the supports are drawn in.
        dimension: d.
        support_size: N, the number of points per support.
        support_count: J, the number of supports per discount.
        relative_eigenvalue_tolerance: ε of `compute_contraction_threshold`.
        supports_per_batch: How many supports are computed together; changes memory use,
            not the supports drawn.
        discounts_per_batch: How many discounts are computed together; likewise.
        direction_norm_floor: See `draw_uniform_ball_support`.

    Returns:
        τ̂⋆_{S_j}(γ_l) at row l, column j: shape (L, J).

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    return _draw_contraction_thresholds_compiled(
        key,
        kernel,
        discounts,
        jnp.asarray(radius, dtype=jnp.float64),
        dimension=dimension,
        support_size=support_size,
        support_count=support_count,
        relative_eigenvalue_tolerance=relative_eigenvalue_tolerance,
        supports_per_batch=supports_per_batch,
        discounts_per_batch=discounts_per_batch,
        direction_norm_floor=direction_norm_floor,
    )


@partial(
    jax.jit,
    static_argnames=(
        "dimension",
        "support_size",
        "support_count",
        "relative_eigenvalue_tolerance",
        "supports_per_batch",
        "discounts_per_batch",
        "direction_norm_floor",
    ),
)
def _draw_contraction_thresholds_compiled(
    key: PRNGKeyArray,
    kernel: RadialKernel,
    discounts: Float64[Array, " discounts"],
    radius: Float64[Array, ""],
    *,
    dimension: int,
    support_size: int,
    support_count: int,
    relative_eigenvalue_tolerance: float,
    supports_per_batch: int,
    discounts_per_batch: int,
    direction_norm_floor: float,
) -> Float64[Array, "discounts supports"]:
    """Compute `draw_contraction_thresholds`, compiled once per kernel and settings.

    Args and returns: see `draw_contraction_thresholds`; `radius` is an array here.
    """
    # Basis Q of 1_N^⊥, built while compiling, shape (N, N − 1)
    zero_sum_basis = build_zero_sum_basis(support_size)
    # One key per discount and per support, shape (L, J)
    support_keys = jax.random.split(key, (discounts.shape[0], support_count))

    def compute_thresholds_for_one_discount(
        support_keys_and_discount: tuple[Key[Array, " supports"], Float64[Array, ""]],
    ) -> Float64[Array, " supports"]:
        """Draw the J supports of one discount and compute their thresholds.

        Args:
            support_keys_and_discount: This discount's J support keys, shape (J,), and
                the discount γ_l.

        Returns:
            τ̂⋆_{S_j}(γ_l) for j = 1, …, J: shape (J,).
        """
        # This discount's support keys, shape (J,), and the discount γ_l, shape ()
        discount_support_keys, discount = support_keys_and_discount

        def compute_threshold_for_one_support(support_key: PRNGKeyArray) -> Float64[Array, ""]:
            """Draw one support and compute its threshold.

            Args:
                support_key: This support's key.

            Returns:
                τ̂⋆_S(γ_l): shape ().
            """
            # Support points z_1, …, z_N, shape (N, d)
            support = draw_uniform_ball_support(
                support_key, dimension, support_size, radius, direction_norm_floor
            )
            # Threshold τ̂⋆_S(γ_l), shape ()
            return compute_contraction_threshold(
                support, kernel, discount, zero_sum_basis, relative_eigenvalue_tolerance
            )

        # Thresholds of the J supports, supports_per_batch at a time, shape (J,)
        return jax.lax.map(
            compute_threshold_for_one_support, discount_support_keys, batch_size=supports_per_batch
        )

    # Thresholds of all discounts, discounts_per_batch at a time, shape (L, J)
    return jax.lax.map(
        compute_thresholds_for_one_discount,
        (support_keys, discounts),
        batch_size=discounts_per_batch,
    )
