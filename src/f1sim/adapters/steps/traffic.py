"""Time lost defending a place (P6-08, a should-have).

Dirty air charges the follower; this step charges the defender. A car with another car within
``window_s`` behind it has to defend, which costs it a little lap time, so a battle slows both
cars and lets a third car close. Lapping is not modelled (cars are compared at equal lap counts),
so the backmarker half of the roadmap's "traffic and defending" is left for when the engine tracks
lap counts; this is the defending half. Like ``DirtyAirStep`` it is switched off under a
neutralisation and written so the ablation runner can turn it off on its own.
"""

import numpy as np

from f1sim.core.domain.race_state import (
    RaceState,
    ahead_in_order,
    following_fraction,
    lap_start_times,
)
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "traffic"
TRAFFIC_WINDOW_S = 1.5
"""A car with another within this many seconds behind it is defending, seconds."""
DEFEND_LOSS_S = 0.3
"""Time lost per lap while defending at zero gap, tapering to zero at the window edge."""


class TrafficStep:
    """Charge each car that is under threat a defending time loss (P6-08).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(
        self, *, window_s: float = TRAFFIC_WINDOW_S, loss_s: float = DEFEND_LOSS_S
    ) -> None:
        """Create the step.

        Args:
            window_s: Gap within which a follower makes the car ahead defend, seconds.
            loss_s: Time lost at zero gap, seconds.

        Raises:
            InvariantError: If the window is not positive or the loss is negative.
        """
        if window_s <= 0.0 or loss_s < 0.0:
            raise InvariantError("the traffic window must be positive and the loss >= 0")
        self.window_s = window_s
        self.loss_s = loss_s

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Add the defending loss, measured on the gaps at the start of the lap.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused.
            rng: Seeded random generator; unused, the loss is deterministic.
        """
        del ctx, rng
        threat = self.loss_s * following_fraction(state, self.window_s)
        penalty = np.zeros_like(threat)
        rows = np.arange(state.n_sims, dtype=np.int64).reshape(-1, 1)
        ahead = ahead_in_order(lap_start_times(state))
        defender = np.where(ahead >= 0, ahead, 0)
        np.add.at(penalty, (rows, defender), threat)
        penalty = np.where(state.neutralised, 0.0, penalty)
        state.race_time_s = state.race_time_s + penalty
        state.lap_time_s = state.lap_time_s + penalty
