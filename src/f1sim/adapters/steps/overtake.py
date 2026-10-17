"""Resolve passes with the fitted pass model and the era's overtake aid (P5-04, P5-06).

The lap-time step lets every car run its own lap, so a faster car can end the lap with a lower
cumulative time than the car it started behind. On track it has to get past first, so this step
walks the running order as it stood at the start of the lap, front to back. A car that would end
the lap ahead of (or within ``min_gap_s`` of) the car in front attempts a pass, succeeding with
the probability the :class:`OvertakingModel` gives its pace advantage, the circuit and the aid
(DRS to 2025, Overtake Mode from 2026, read from the season rules through ``LapContext``). A
failed attempt leaves it ``min_gap_s`` behind; a successful one gains exactly one place. Cars that
pitted this lap or have retired take no part: their place follows from their own time.
"""

import numpy as np
from numpy.typing import NDArray

from f1sim.adapters.models.overtaking import OvertakingModel
from f1sim.core.domain.race_state import RaceState, lap_start_times, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "overtake"
MIN_GAP_S = 0.3
"""Closest a car may sit behind one it failed to pass, seconds."""
PASS_MARGIN_S = 0.1
"""How far ahead a car that has just passed sits, at most, seconds."""


class OvertakeStep:
    """Let close, faster cars attempt a pass each lap (P5-04, P5-06).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(
        self,
        model: OvertakingModel,
        *,
        circuit: str | None = None,
        min_gap_s: float = MIN_GAP_S,
        pass_margin_s: float = PASS_MARGIN_S,
        pass_offset: float = 0.0,
    ) -> None:
        """Create the step.

        Args:
            model: The fitted pass-probability model.
            circuit: Circuit ID, for the model's difficulty offset.
            min_gap_s: Following gap after a failed pass, seconds.
            pass_margin_s: Largest lead a passing car is placed with, seconds.
            pass_offset: Log-odds added to every pass, calibrating the engine's pass rate.

        Raises:
            InvariantError: If the gaps are not positive.
        """
        if min_gap_s <= 0.0 or pass_margin_s <= 0.0:
            raise InvariantError("min_gap_s and pass_margin_s must be positive")
        self.model = model
        self.circuit = circuit
        self.min_gap_s = min_gap_s
        self.pass_margin_s = pass_margin_s
        self.pass_offset = pass_offset

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """Resolve every pass attempt of the lap.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; its ``overtake_aid`` selects the log-odds bonus.
            rng: Seeded random generator.
        """
        order = np.argsort(lap_start_times(state), axis=1, kind="stable")
        active = state.running & ~state.pit_this_lap
        times = state.race_time_s.copy()
        chain = _Chain(state.n_sims)
        for k in range(state.n_cars):
            attacker = order[:, k]
            part = active[chain.sims, attacker]
            defender = chain.defender()
            success = self._duel(
                state, times, chain, attacker=attacker, part=part, ctx=ctx, rng=rng
            )
            chain.advance(part, attacker=attacker, defender=defender, success=success)
        after = np.where(state.retired, 0.0, times)
        before = np.where(state.retired, 0.0, state.race_time_s)
        state.lap_time_s = state.lap_time_s + (after - before)
        state.race_time_s = times
        update_order(state)

    def _duel(  # pylint: disable=too-many-arguments
        self,
        state: RaceState,
        times: NDArray[np.float64],
        chain: "_Chain",
        *,
        attacker: NDArray[np.int_],
        part: NDArray[np.bool_],
        ctx: LapContext,
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        defender = chain.defender()
        t_att = times[chain.sims, attacker]
        t_def = times[chain.sims, defender]
        attempt = part & (chain.last >= 0) & (t_att < t_def + self.min_gap_s)
        chance = self._pass_chance(state, chain.sims, attacker, defender, ctx)
        success = attempt & (rng.random(state.n_sims) < chance)
        times[chain.sims, attacker] = np.where(
            success,
            self._squeeze(t_att, chain.floor(times), t_def),
            np.where(attempt, t_def + self.min_gap_s, t_att),
        )
        return success

    def _pass_chance(
        self,
        state: RaceState,
        sims: NDArray[np.int_],
        attacker: NDArray[np.int_],
        defender: NDArray[np.int_],
        ctx: LapContext,
    ) -> NDArray[np.float64]:
        advantage = state.lap_time_s[sims, defender] - state.lap_time_s[sims, attacker]
        return self.model.pass_probability(
            advantage, circuit=self.circuit, aid=ctx.overtake_aid, offset=self.pass_offset
        )

    def _squeeze(
        self, t_att: NDArray[np.float64], floor: NDArray[np.float64], ceiling: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        room = ceiling - floor
        margin = np.minimum(self.pass_margin_s, room / 3.0)
        upper = ceiling - margin
        lower = floor + margin
        return np.minimum(np.maximum(t_att, lower), upper)


class _Chain:
    """The last two participating cars already placed, per simulation.

    Attributes:
        sims: Simulation indices, for fancy indexing.
        last: Column of the last car placed, or -1.
        second: Column of the car placed before it, or -1.
    """

    def __init__(self, n_sims: int) -> None:
        self.sims = np.arange(n_sims, dtype=np.int64)
        self.last = np.full(n_sims, -1, dtype=np.int64)
        self.second = np.full(n_sims, -1, dtype=np.int64)

    def defender(self) -> NDArray[np.int_]:
        """The car the next attacker races: the last one placed.

        Returns:
            Its column, or 0 where there is none yet (masked by the caller).
        """
        return np.where(self.last >= 0, self.last, 0)

    def floor(self, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """The time of the car two places ahead, which a passing car may not get ahead of.

        Args:
            times: Current cumulative times.

        Returns:
            That car's time, or minus infinity where there is none.
        """
        second = np.where(self.second >= 0, self.second, 0)
        return np.where(self.second >= 0, times[self.sims, second], -np.inf)

    def advance(
        self,
        part: NDArray[np.bool_],
        *,
        attacker: NDArray[np.int_],
        defender: NDArray[np.int_],
        success: NDArray[np.bool_],
    ) -> None:
        """Record where this car ended up relative to the car it raced.

        Args:
            part: Whether the attacker took part this lap.
            attacker: The attacker's column.
            defender: The defender's column.
            success: Whether the attacker passed.
        """
        new_last = np.where(success, defender, attacker)
        new_second = np.where(success, attacker, self.last)
        self.last = np.where(part, new_last, self.last)
        self.second = np.where(part, new_second, self.second)
