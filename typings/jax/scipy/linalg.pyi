"""``jax.scipy.linalg``, typed loosely: solvers return JAX arrays (typed ``Any``)."""

from typing import Any

def __getattr__(name: str) -> Any: ...
