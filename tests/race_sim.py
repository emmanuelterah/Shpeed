"""A synthetic race wired through every Phase 5 step, for engine-level tests.

The pace, tyres, plans and pit losses are drawn from a seeded generator, so a test can build
the same race twice (reproducibility, P5-09), run it with an invariant checker appended (property
tests, P5-10) or time it at full size (performance, P5-11) without fitting any model.
"""

from dataclasses import dataclass

import numpy as np

from f1sim.adapters.models.overtaking import default_overtaking
from f1sim.adapters.models.tyres import AGE_GRID, SampledLoss
from f1sim.adapters.steps.dirty_air import DirtyAirStep
from f1sim.adapters.steps.lap_time import LapTimeStep
from f1sim.adapters.steps.overtake import OvertakeStep
from f1sim.adapters.steps.pit_stop import PitStopStep
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.race_state import RaceState
from f1sim.core.engine.grid import FULL_FUEL_KG, starting_state
from f1sim.core.ports.lap_step import LapContext, LapStep

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WEAR = (0.09, 0.05, 0.03)
OFFSETS = (-0.6, 0.0, 0.4)
BASE_LAP_S = 90.0
PIT_LOSS_S = 21.0


@dataclass
class SimRace:
    """Everything ``simulate_race`` needs for one synthetic race.

    Attributes:
        drivers: Driver IDs in column order.
        state: The starting state.
        steps: Lap-time, dirty-air, pit-stop and overtake steps, in engine order.
        ctx: The race facts.
    """

    drivers: tuple[str, ...]
    state: RaceState
    steps: list[LapStep]
    ctx: LapContext


def build_race(  # pylint: disable=too-many-locals
    *,
    n_sims: int,
    n_cars: int = 22,
    laps: int = 57,
    seed: int = 0,
    p_dnf: float = 0.08,
    aid: str | None = "drs",
    pace_spread_s: float = 2.0,
) -> SimRace:
    """Build a seeded synthetic race with one or two stops per car."""
    rng = np.random.default_rng(seed)
    entries = [
        RaceEntry(f"d{i}", f"t{i // 2}", i + 1, pit_lane=i == n_cars - 1) for i in range(n_cars)
    ]
    spread = np.linspace(0.0, pace_spread_s, n_cars)
    pace = BASE_LAP_S + spread + rng.normal(0.0, 0.15 * pace_spread_s, (n_sims, n_cars))
    values = np.stack(
        [
            np.tile(rate * AGE_GRID, (n_sims, 1)) + rng.normal(0.0, 0.01, (n_sims, len(AGE_GRID)))
            for rate in WEAR
        ]
    )
    loss = SampledLoss(COMPOUNDS, np.clip(values, 0.0, None))
    stops = np.zeros((n_sims, n_cars, 3), dtype=np.int64)
    first = rng.integers(max(1, laps // 4), max(2, laps // 2), (n_sims, n_cars))
    stops[:, :, 0] = first
    second = rng.random((n_sims, n_cars)) < 0.4
    stops[:, :, 1] = np.where(second, np.minimum(first + laps // 4, laps - 1), 0)
    compounds = rng.integers(0, 3, (n_sims, n_cars, 3))
    pit_loss = PIT_LOSS_S + rng.exponential(0.5, (n_sims, n_cars, 3))
    steps: list[LapStep] = [
        LapTimeStep(
            pace,
            loss,
            fuel_s_per_lap=0.03,
            burn_kg_per_lap=FULL_FUEL_KG / laps,
            reference_fuel_kg=FULL_FUEL_KG,
            compound_offset_s=np.asarray(OFFSETS),
        ),
        DirtyAirStep(),
        PitStopStep(stops, compounds, pit_loss),
        OvertakeStep(default_overtaking(), circuit="bahrain"),
    ]
    state = starting_state(
        entries,
        n_sims=n_sims,
        start_compounds=rng.integers(0, 3, (n_sims, n_cars)),
        retired=rng.random((n_sims, n_cars)) < p_dnf,
    )
    return SimRace(
        tuple(e.driver_id for e in entries), state, steps, LapContext(laps, PIT_LOSS_S, aid)
    )
