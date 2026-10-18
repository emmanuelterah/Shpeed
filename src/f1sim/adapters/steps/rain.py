"""Rain regime (P6-07, a should-have, cut first if time slips).

The rain probability is an input at prediction time, not a fitted hazard: each run is drawn dry or
wet, and a wet run's laps carry ``WET_EXTRA_S`` more time (plus noise), so the forecast is a mix of
dry and wet races and the spread already reflects the chance of a downpour. It is deliberately
simple: one extra-seconds figure for every car, no intermediate versus full-wet distinction, no
drying line. That is enough to move the forecast when the rain chance is high while keeping the
model honest about how little the store says about grip.
"""

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "rain"
WET_EXTRA_S = 12.0
"""Extra lap time for a wet run, seconds; a first, deliberately coarse figure."""
WET_NOISE_S = 1.5
"""Extra lap-to-lap variability in the wet, seconds."""


class RainStep:
    """Add the wet-run time loss to each lap (P6-07).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(
        self,
        wet: NDArray[np.bool_],
        *,
        extra_s: float = WET_EXTRA_S,
        noise_s: float = WET_NOISE_S,
    ) -> None:
        """Create the step.

        Args:
            wet: Which simulations are wet, of shape ``(n_sims,)``.
            extra_s: Extra lap time in a wet run, seconds.
            noise_s: Extra lap-to-lap variability in the wet, seconds.

        Raises:
            InvariantError: If ``wet`` is not 1-D or the figures are negative.
        """
        if wet.ndim != 1:
            raise InvariantError("wet must be a 1-D array of (n_sims,)")
        if extra_s < 0.0 or noise_s < 0.0:
            raise InvariantError("wet time figures must be non-negative")
        self.wet = wet
        self.extra_s = extra_s
        self.noise_s = noise_s

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Add the wet-run loss to every car in a wet simulation.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused.
            rng: Seeded random generator.

        Raises:
            InvariantError: If ``wet`` does not have one value per simulation.
        """
        del ctx
        if self.wet.shape != (state.n_sims,):
            raise InvariantError("wet must have one value per simulation")
        noise = rng.normal(0.0, self.noise_s, size=state.race_time_s.shape)
        added = np.where(self.wet[:, np.newaxis], self.extra_s + noise, 0.0)
        added = np.where(state.running, added, 0.0)
        state.race_time_s = state.race_time_s + added
        state.lap_time_s = state.lap_time_s + added
        update_order(state)
