"""The tradeoff heatmap of Algorithm 4: the best power a contraction budget allows."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float64

from mmd_contraction.precision import raise_unless_float64_enabled


def build_tradeoff_heatmap(
    thresholds: Float64[Array, "parameters discounts"],
    powers: Float64[Array, " parameters"],
    budgets: Float64[Array, " budgets"],
) -> Float64[Array, "budgets discounts"]:
    """Build the heatmap of Algorithm 4, step 3.

    The cell of budget τ̄ and discount γ_l is the highest power D̂_i among the kernel
    parameters i whose threshold τ̂_{i,l} is at most τ̄, or NaN when no parameter fits.

    Args:
        thresholds: τ̂_{i,l}, one row per kernel parameter, one column per discount:
            shape (parameters, discounts).
        powers: D̂_i, one per kernel parameter: shape (parameters,).
        budgets: τ̄, the contraction budgets: shape (budgets,).

    Returns:
        The best power for each budget and discount: shape (budgets, discounts).

    Raises:
        RuntimeError: If float64 is not enabled in JAX.
    """
    raise_unless_float64_enabled()
    # Fit of parameter i to budget τ̄ at discount γ_l, τ̂_{i,l} ≤ τ̄,
    # shape (budgets, parameters, discounts)
    fits: Bool[Array, "budgets parameters discounts"] = (
        thresholds[None, :, :] <= budgets[:, None, None]
    )
    # Allowed powers: D̂_i where parameter i fits, −∞ elsewhere,
    # shape (budgets, parameters, discounts)
    allowed_powers = jnp.where(fits, powers[None, :, None], -jnp.inf)
    # Best power max_i D̂_i over the parameters that fit, −∞ if none fits,
    # shape (budgets, discounts)
    best_powers = jnp.max(allowed_powers, axis=1)
    # Heatmap: the best power, NaN where no parameter fits, shape (budgets, discounts)
    return jnp.where(jnp.isneginf(best_powers), jnp.nan, best_powers)
