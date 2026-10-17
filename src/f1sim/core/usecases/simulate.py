"""Run the engine for one event and aggregate the runs into a PositionMatrix (P5-12).

The engine is deliberately split from the model: this use case knows only the ``LapStep`` port, so
any set of mechanics can be simulated and the same aggregation runs for all of them.
"""

from collections.abc import Sequence

import numpy as np

from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race_state import RaceState, finishing_orders
from f1sim.core.engine.engine import run_race
from f1sim.core.ports.lap_step import LapContext, LapStep

SMOOTHING = 1e-3
"""Weight of the uniform forecast mixed into the matrix, matching the baseline predictors."""


def simulate_race(
    drivers: Sequence[str],
    state: RaceState,
    steps: Sequence[LapStep],
    ctx: LapContext,
    rng: np.random.Generator,
) -> PositionMatrix:
    """Run every parallel race to the flag and count the finishing positions.

    Args:
        drivers: Driver IDs, one per car, in ``state`` column order.
        state: The starting state; it is mutated in place.
        steps: The race mechanics to apply each lap, in order.
        ctx: Facts about the race, including its length.
        rng: Seeded random generator shared by every step.

    Returns:
        The finishing-position forecast, mixed with ``SMOOTHING`` of the uniform forecast.
    """
    final = run_race(state, steps, ctx, rng)
    orders = finishing_orders(final)
    return PositionMatrix.from_finishing_orders(tuple(drivers), orders).smoothed(SMOOTHING)
