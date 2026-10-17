"""``jax.scipy``, typed loosely: every submodule returns JAX values (typed ``Any``)."""

from typing import Any

def __getattr__(name: str) -> Any: ...
