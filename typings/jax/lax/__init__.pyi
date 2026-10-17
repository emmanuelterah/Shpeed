"""``jax.lax``, typed loosely: control-flow primitives return JAX values (typed ``Any``)."""

from typing import Any

def __getattr__(name: str) -> Any: ...
