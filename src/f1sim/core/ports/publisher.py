"""Publisher port (P8-03)."""

from collections.abc import Mapping
from typing import Any, Protocol


class Publisher(Protocol):
    def publish(self, record: Mapping[str, Any]) -> str:
        """Write a schema-valid prediction record atomically and return where it went."""
        ...
