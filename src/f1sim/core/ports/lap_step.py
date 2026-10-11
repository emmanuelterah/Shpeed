"""The LapStep port (P5-08).

The engine runs an ordered list of steps each lap, so a new race mechanic is a new step
rather than an edit to the engine (open/closed principle).
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from f1sim.core.domain.race_state import RaceState


@dataclass(frozen=True)
class LapContext:
    """Read-only facts about the race that steps may need.

    Attributes:
        total_laps: Race distance in laps.
        pit_loss_s: Time lost to a pit stop at this circuit, in seconds.
        overtake_aid: ``"drs"`` or ``"overtake_mode"``, or None for neither.
    """

    total_laps: int
    pit_loss_s: float
    overtake_aid: str | None


class LapStep(Protocol):
    """One race mechanic, applied once per lap.

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name: str

    @abstractmethod
    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Advance ``state`` by this mechanic, in place.

        Args:
            state: Parallel race state, mutated in place for speed.
            ctx: Facts about the race.
            rng: Seeded random generator shared by all steps.
        """
