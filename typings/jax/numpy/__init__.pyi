"""``jax.numpy``, typed loosely: every call returns a JAX array (typed ``Any``)."""

from typing import Any

def __getattr__(name: str) -> Any: ...
