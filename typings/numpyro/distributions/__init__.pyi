"""NumPyro distributions, typed loosely: each is a callable returning a distribution."""

from typing import Any

def __getattr__(name: str) -> Any: ...
