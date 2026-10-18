"""Lap-time loss for following closely in dirty air (P5-05).

A car that starts the lap within ``window_s`` of the car ahead loses downforce in its wake and
so time: ``penalty_s`` at zero gap, tapering linearly to nothing at the edge of the window. It is
its own step, apart from the pass model, so the ablation runner (P6-09) can switch it off alone.
"""

import numpy as np

from f1sim.core.domain.race_state import RaceState, following_fraction
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "dirty_air"
DIRTY_AIR_S = 1.0
"""A car within this many seconds of the one ahead is in dirty air, seconds."""
DIRTY_AIR_PENALTY_S = 0.4
"""Time lost per lap at zero gap in dirty air, seconds; it tapers to zero at ``DIRTY_AIR_S``."""


class DirtyAirStep:
    """Charge each close follower its dirty-air time loss (P5-05).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(
        self, *, window_s: float = DIRTY_AIR_S, penalty_s: float = DIRTY_AIR_PENALTY_S
    ) -> None:
        """Create the step.

        Args:
            window_s: Gap within which dirty air is felt, seconds.
            penalty_s: Time lost at zero gap, seconds.

        Raises:
            InvariantError: If the window is not positive or the penalty is negative.
        """
        if window_s <= 0.0 or penalty_s < 0.0:
            raise InvariantError("the dirty-air window must be positive and the penalty >= 0")
        self.window_s = window_s
        self.penalty_s = penalty_s

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Add the dirty-air loss, measured on the gaps at the start of the lap.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused.
            rng: Seeded random generator; unused, the loss is deterministic.
        """
        del ctx, rng
        penalty = self.penalty_s * following_fraction(state, self.window_s)
        penalty = np.where(state.neutralised, 0.0, penalty)
        state.race_time_s = state.race_time_s + penalty
        state.lap_time_s = state.lap_time_s + penalty
