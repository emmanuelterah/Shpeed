"""The Publisher port (P8-03): where prediction records go."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol


class Publisher(Protocol):
    """Writes prediction records somewhere readers can find them."""

    @abstractmethod
    def publish(self, record: Mapping[str, Any]) -> str:
        """Write one prediction record atomically.

        Args:
            record: A record valid against ``schemas/prediction-record.v1.schema.json``.

        Returns:
            Where the record was written, such as a file path.
        """
