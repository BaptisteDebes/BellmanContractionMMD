"""Test setup: float64, and runtime shape checks on every annotated `mmd_contraction` function.

The jaxtyping import hook must be installed before `mmd_contraction` is imported. It wraps
every function, method and dataclass of the package with beartype, so the jaxtyping shape
and dtype annotations are checked on each call during the tests.
"""

import jax
from jaxtyping import install_import_hook

jax.config.update("jax_enable_x64", True)

_shape_check_hook = install_import_hook("mmd_contraction", "beartype.beartype")
