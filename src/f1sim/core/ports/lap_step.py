"""LapStep port (P5-08). The engine runs an ordered list of steps each lap (open/closed)."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from f1sim.core.domain.race_state import RaceState


@dataclass(frozen=True)
class LapContext:
    """Read-only facts about the race that steps may need."""

    total_laps: int
    pit_loss_s: float
    overtake_aid: str | None


class LapStep(Protocol):
    """One race mechanic.

    Mutates ``state`` in place for speed and must not take on other steps' concerns.
    """

    name: str

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None: ...
