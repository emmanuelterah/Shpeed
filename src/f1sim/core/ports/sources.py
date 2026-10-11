"""SourceAdapter port (P0-04, P1-02, P1-03)."""

from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from f1sim.core.domain.entities import SessionKey


class SourceAdapter(Protocol):
    """Fetches one source's data for a session as named tables."""

    name: str

    def sessions(self, season: int) -> Sequence[SessionKey]:
        """Every session this source has for ``season``."""
        ...

    def fetch(self, session: SessionKey) -> Mapping[str, pd.DataFrame]:
        """Tables for one session, keyed by table name (e.g. ``"laps"``, ``"stints"``)."""
        ...
