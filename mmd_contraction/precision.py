"""Check of the floating-point precision the package's functions need."""

import jax


def raise_unless_float64_enabled() -> None:
    """Check that JAX computes in float64, which the package's functions need.

    Raises:
        RuntimeError: If `jax_enable_x64` is off; JAX would then compute in float32.
    """
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "float64 is required: call jax.config.update('jax_enable_x64', True) first."
        )
