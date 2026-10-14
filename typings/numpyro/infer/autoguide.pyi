"""NumPyro automatic guides, typed loosely."""

from typing import Any

def __getattr__(name: str) -> Any: ...
