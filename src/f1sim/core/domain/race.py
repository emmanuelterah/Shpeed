"""Race entries and outcomes: who starts a race, and how each of them finished."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

DNF = 0
"""Outcome code for a driver who was not classified (retired, disqualified, did not start)."""


@dataclass(frozen=True, slots=True)
class RaceEntry:
    """One driver entered in a race, as known at the prediction cutoff.

    Attributes:
        driver_id: Jolpica driver ID.
        team_id: Jolpica constructor ID.
        grid: Expected starting position, from 1 (see ADR-0005 for how it is chosen).
    """

    driver_id: str
    team_id: str
    grid: int


def encode_outcome(
    entries: Sequence[RaceEntry], classified: Sequence[tuple[str, int]]
) -> NDArray[np.int_]:
    """Turn an official classification into per-entry finishing positions.

    Positions are re-ranked among the entries, so a classified driver who was not an entry
    does not leave a gap. Entries missing from the classification did not finish.

    Args:
        entries: The drivers that were forecast, in forecast order.
        classified: ``(driver_id, official_position)`` for every classified finisher.

    Returns:
        For each entry, its finishing position among entries (from 1), or ``DNF`` (0).
    """
    index = {e.driver_id: i for i, e in enumerate(entries)}
    finishers = sorted((pos, did) for did, pos in classified if did in index)
    outcome = np.full(len(entries), DNF, dtype=np.int64)
    for rank, (_, driver_id) in enumerate(finishers, start=1):
        outcome[index[driver_id]] = rank
    return outcome
