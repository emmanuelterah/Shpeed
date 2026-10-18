"""Safety car, VSC and red flag (P6-01, P6-02).

A neutralisation is drawn from the circuit's fitted hazard and lasts a sampled number of laps:

- **Safety car (level 2)** bunches the field to ``SC_GAP_S`` a car and makes stops cheaper
  (``PitStopStep`` reads the level through ``RaceState.neutralised``). No passing happens while
  it is out.
- **VSC (level 1)** cheapens stops by less and, unlike a safety car, does not bunch the field,
  because the gaps are preserved under a virtual safety car.
- **Red flag (level 3, P6-02)** lasts the lap it is shown: cars take a free tyre change and line
  up for a standing restart at ``RESTART_GAP_S``, then the shared lap-1 launch
  (:func:`apply_launch`) scrubs the field again.

A safety car or VSC lasts ``duration_laps`` laps; the level is written onto ``RaceState.safety_car``
each lap, so ``DirtyAirStep``, ``OvertakeStep`` and ``PitStopStep`` see it without any wiring.
"""

import numpy as np
from numpy.typing import NDArray

from f1sim.adapters.models.hazards import SafetyCarHazard
from f1sim.adapters.steps.lap1 import Launch
from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "safety_car"
NO_NEUTRALISATION = 0
VSC_LEVEL = 1
SAFETY_CAR_LEVEL = 2
RED_FLAG_LEVEL = 3
SC_GAP_S = 1.0
"""Gap between consecutive cars once the safety car has bunched the field, seconds."""
RESTART_GAP_S = 0.5
"""Gap between consecutive cars on a red-flag standing restart, seconds."""
SC_SHARE = 0.6
"""Share of neutralisations whose incident is serious enough for a full safety car, not a VSC."""


class SafetyCarStep:
    """Draw, run and end neutralisations lap by lap (P6-01, P6-02).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(  # pylint: disable=too-many-arguments
        self,
        hazard: SafetyCarHazard,
        *,
        circuit: str | None = None,
        total_laps: int,
        launch: Launch | None = None,
        sc_share: float = SC_SHARE,
        sc_gap_s: float = SC_GAP_S,
        restart_gap_s: float = RESTART_GAP_S,
    ) -> None:
        """Create the step.

        Args:
            hazard: The fitted safety-car, red-flag and duration hazards.
            circuit: Circuit ID, for its per-lap deployment rate.
            total_laps: Race distance in laps, to turn the red-flag rate per race into per lap.
            launch: The shared launch fixture, for the red-flag standing restart (P6-02).
            sc_share: Share of neutralisations that are full safety cars rather than VSCs.
            sc_gap_s: Gap between cars once bunched, seconds.
            restart_gap_s: Gap between cars on a standing restart, seconds.

        Raises:
            InvariantError: If the race has no laps or a share is not a probability.
        """
        if total_laps < 1:
            raise InvariantError("a race must have at least one lap")
        if not 0.0 <= sc_share <= 1.0:
            raise InvariantError("sc_share must be a probability")
        self.hazard = hazard
        self.deploy_rate = hazard.deploy_rate(circuit)
        self.red_per_lap = hazard.red_flag_per_race / total_laps
        self.duration_laps = max(1, round(hazard.duration_laps))
        self.launch = launch
        self.sc_share = sc_share
        self.sc_gap_s = sc_gap_s
        self.restart_gap_s = restart_gap_s
        self._level: NDArray[np.int_] | None = None
        self._remaining: NDArray[np.int_] | None = None

    def apply(  # pylint: disable=too-many-locals
        self, state: RaceState, ctx: LapContext, rng: np.random.Generator
    ) -> None:
        """Advance the neutralisation state and bunch or restart the field.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused, the race distance was passed at construction.
            rng: Seeded random generator.
        """
        del ctx
        n_sims = state.n_sims
        if self._level is None or self._level.shape != (n_sims,):
            self._level = np.zeros(n_sims, dtype=np.int64)
            self._remaining = np.zeros(n_sims, dtype=np.int64)
        remaining = np.asarray(self._remaining, dtype=np.int64)
        level = np.asarray(self._level, dtype=np.int64)
        level = np.where(remaining <= 0, NO_NEUTRALISATION, level)
        active = level > NO_NEUTRALISATION
        red = (~active) & (rng.random(n_sims) < self.red_per_lap)
        deploying = (~active) & ~red & (rng.random(n_sims) < self.deploy_rate)
        full = deploying & (rng.random(n_sims) < self.sc_share)
        duration = rng.integers(1, 2 * self.duration_laps + 1, n_sims)
        new_remaining = np.where(deploying, duration - 1, np.maximum(remaining - 1, 0))
        new_remaining = np.where(red, 0, new_remaining)
        new_level = np.where(
            red,
            RED_FLAG_LEVEL,
            np.where(deploying, np.where(full, SAFETY_CAR_LEVEL, VSC_LEVEL), level),
        )
        self._level = new_level
        self._remaining = new_remaining
        state.safety_car = new_level
        bunch = new_level == SAFETY_CAR_LEVEL
        if bunch.any():
            self._bunch(state, bunch, self.sc_gap_s)
        restart = new_level == RED_FLAG_LEVEL
        if restart.any():
            state.tyre_age = np.where(restart[:, np.newaxis], 0, state.tyre_age)
            self._bunch(state, restart, self.restart_gap_s)
            if self.launch is not None:
                self.launch(state, rng, mask=restart)
        update_order(state)

    def _bunch(self, state: RaceState, sims: NDArray[np.bool_], gap_s: float) -> None:
        """Line the running cars up ``gap_s`` apart behind the leader, for the given sims."""
        times = state.race_time_s
        running = state.running
        finite = np.where(running, times, np.inf)
        order = np.argsort(finite, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(state.n_sims, dtype=np.int64).reshape(-1, 1)
        ranks[rows, order] = np.arange(1, state.n_cars + 1)
        leader = np.min(finite, axis=1, keepdims=True)
        lead = np.where(np.isfinite(leader), leader, 0.0)
        target = lead + (ranks - 1).astype(np.float64) * gap_s
        mask = sims[:, np.newaxis] & running
        delta = np.where(mask, target - times, 0.0)
        state.race_time_s = times + delta
        state.lap_time_s = state.lap_time_s + delta


def level_of(state: RaceState) -> NDArray[np.int_]:
    """The neutralisation level per car, of shape ``(n_sims, n_cars)``, zero when unset.

    Args:
        state: The race state.

    Returns:
        The broadcast level array.
    """
    if state.safety_car is None:
        return np.zeros(state.race_time_s.shape, dtype=np.int64)
    levels = np.broadcast_to(state.safety_car[:, np.newaxis], state.race_time_s.shape)
    return np.asarray(levels, dtype=np.int64)


__all__ = ["SafetyCarStep", "level_of"]
