"""The SourceAdapter port (P0-04, P1-02, P1-03): one external data source."""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from f1sim.core.domain.entities import SessionKey


class SourceAdapter(Protocol):
    """Fetches one source's data for a session as named canonical tables.

    Attributes:
        name: Short source name recorded in the sync ledger, such as ``"jolpica"``.
    """

    name: str

    @abstractmethod
    def sessions(self, season: int) -> Sequence[SessionKey]:
        """List the sessions of a season that this source can provide now.

        Args:
            season: Championship year.

        Returns:
            Finished sessions this source covers, in schedule order.

        Raises:
            DataError: If the source cannot list the season at all.
        """

    @abstractmethod
    def fetch(self, session: SessionKey) -> Mapping[str, pd.DataFrame]:
        """Fetch one session's data.

        Args:
            session: The session to fetch.

        Returns:
            Canonical tables keyed by table name, such as ``"laps"`` or ``"results"``.

        Raises:
            DataError: If the source has no usable data for the session.
        """
