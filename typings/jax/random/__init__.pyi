"""``jax.random``, typed loosely: keys and draws are JAX values (typed ``Any``)."""

from typing import Any

def __getattr__(name: str) -> Any: ...
