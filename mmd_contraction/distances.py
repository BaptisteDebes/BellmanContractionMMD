"""Squared distances between all pairs of points, the input of every radial kernel."""

import jax.numpy as jnp
from jaxtyping import Array, Float64


def compute_squared_distances(
    points: Float64[Array, "points dimension"],
) -> Float64[Array, "points points"]:
    """Compute the squared distances ‖z_i − z_j‖² between all pairs of points.

    Args:
        points: z_1, …, z_N: shape (N, d).

    Returns:
        ‖z_i − z_j‖² at row i, column j: shape (N, N); exactly 0 on the diagonal.
    """
    # Differences z_i − z_j for all pairs, shape (N, N, d)
    differences = points[:, None, :] - points[None, :, :]
    # Squared distances ‖z_i − z_j‖², shape (N, N)
    squared_distances = jnp.sum(differences * differences, axis=-1)
    # A point's distance to itself, set to exactly 0: when the points come from earlier
    # operations in the same compiled program, the compiler can round the two sides of
    # z_i − z_i differently, and a small kernel exponent would magnify the difference
    return squared_distances.at[jnp.diag_indices(points.shape[0])].set(0.0)
