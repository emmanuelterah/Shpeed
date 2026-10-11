"""Runs many races in parallel, lap by lap, through an ordered pipeline of LapSteps (P5-02)."""

from collections.abc import Sequence

import numpy as np

from f1sim.core.domain.race_state import RaceState
from f1sim.core.ports.lap_step import LapContext, LapStep


def run_race(
    state: RaceState, steps: Sequence[LapStep], ctx: LapContext, rng: np.random.Generator
) -> RaceState:
    """Advance ``state`` to the chequered flag.

    Adding a mechanic means adding a step, not editing this function.

    Args:
        state: Starting state; it is mutated in place.
        steps: Mechanics to apply each lap, in order.
        ctx: Facts about the race, including its length.
        rng: Seeded random generator passed to every step.

    Returns:
        The same ``state`` object, at the end of the race.
    """
    while state.lap < ctx.total_laps:
        state.lap += 1
        for step in steps:
            step.apply(state, ctx, rng)
    return state
