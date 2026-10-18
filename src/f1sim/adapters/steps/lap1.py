"""Launches and first-corner incidents (P6-05).

Lap 1 reorders the field more than any other lap, and Phase 5 showed why it matters: without a
launch term the simulated grid-to-lap-1 rank correlation was 0.997 against 0.88 real. Two pieces
are modelled:

- **Launch.** A driver's expected change in places over the start, fitted from the grid to the
  end-of-lap-1 position and shrunk toward zero (:class:`~f1sim.adapters.models.hazards.Lap1Hazard`).
  It enters as a time bonus of ``LAUNCH_PLACE_S`` per place gained, so a launch that gains places
  pulls the car up the order the same way pace would; :func:`apply_launch` is shared with the
  red-flag standing restart (P6-02).
- **First-corner incident.** Each car is eliminated on lap 1 with the fitted incident probability,
  which is what makes a start genuinely risky rather than merely noisy.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from f1sim.core.domain.race_state import RaceState, update_order
from f1sim.core.errors import InvariantError
from f1sim.core.ports.lap_step import LapContext

NAME = "lap1"
LAUNCH_PLACE_S = 0.6
"""Seconds of race time gained per place a launch is expected to make up."""
LAUNCH_NOISE_PLACES = 0.5
"""Standard deviation of the places actually made up around a driver's launch mean."""
DEFAULT_INCIDENT = 0.03
"""First-corner incident probability per car when no fit is available."""


class Launch(Protocol):
    """The fixture that :func:`apply_launch` is wrapped as.

    It lets the lap-1 step and the red-flag restart share one launch implementation without
    either depending on the hazard model.
    """

    def __call__(
        self,
        state: RaceState,
        rng: np.random.Generator,
        *,
        mask: NDArray[np.bool_] | None = ...,
    ) -> None:
        """Apply one launch to ``state`` in place."""


def apply_launch(
    state: RaceState,
    launch_places: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    place_s: float = LAUNCH_PLACE_S,
    noise_places: float = LAUNCH_NOISE_PLACES,
    mask: NDArray[np.bool_] | None = None,
) -> None:
    """Pull every running car up the order by its (noisy) launch.

    The bonus is subtracted from ``race_time_s`` but not ``lap_time_s``, so it shows up in
    ``lap_start_times`` exactly as a launch does: the car begins the lap further up the road.

    Args:
        state: Parallel race state, mutated in place.
        launch_places: Expected places gained per car, of shape ``(n_cars,)``.
        rng: Seeded random generator.
        place_s: Seconds of race time per place gained.
        noise_places: Standard deviation of the realised places around the mean.
        mask: Simulations the launch applies to, of shape ``(n_sims,)``; all when None.

    Raises:
        InvariantError: If ``launch_places`` does not have one value per car.
    """
    if launch_places.shape != (state.n_cars,):
        raise InvariantError("launch_places needs one value per car")
    draws = rng.normal(0.0, noise_places, size=state.race_time_s.shape)
    bonus = (launch_places[np.newaxis, :] + draws) * place_s
    active = state.running if mask is None else state.running & mask[:, np.newaxis]
    state.race_time_s = np.where(active, state.race_time_s - bonus, state.race_time_s)
    update_order(state)


def launch_fixture(
    launch_places: NDArray[np.float64],
    *,
    place_s: float = LAUNCH_PLACE_S,
    noise_places: float = LAUNCH_NOISE_PLACES,
) -> Launch:
    """Wrap a launch vector as the fixture both the lap-1 step and the restart call.

    Args:
        launch_places: Expected places gained per car, of shape ``(n_cars,)``.
        place_s: Seconds of race time per place gained.
        noise_places: Standard deviation of the realised places around the mean.

    Returns:
        A callable matching :class:`Launch`.
    """

    def launch(
        state: RaceState, rng: np.random.Generator, *, mask: NDArray[np.bool_] | None = None
    ) -> None:
        apply_launch(
            state, launch_places, rng, place_s=place_s, noise_places=noise_places, mask=mask
        )

    return launch


class Lap1Step:
    """Apply the start to every car on lap 1 (P6-05).

    Attributes:
        name: Step name used in configuration and ablation reports.
    """

    name = NAME

    def __init__(
        self,
        launch_places: NDArray[np.float64],
        *,
        incident: float = DEFAULT_INCIDENT,
        place_s: float = LAUNCH_PLACE_S,
        noise_places: float = LAUNCH_NOISE_PLACES,
    ) -> None:
        """Create the step.

        Args:
            launch_places: Expected places gained per car, of shape ``(n_cars,)``.
            incident: Chance a car is eliminated in a first-corner incident.
            place_s: Seconds of race time per place a launch makes up.
            noise_places: Standard deviation of realised places around the launch mean.

        Raises:
            InvariantError: If the incident probability is not in ``[0, 1]``.
        """
        if not 0.0 <= incident <= 1.0:
            raise InvariantError("incident must be a probability")
        self.launch_places = np.asarray(launch_places, dtype=np.float64)
        self.incident = incident
        self.place_s = place_s
        self.noise_places = noise_places

    def apply(self, state: RaceState, ctx: LapContext, rng: np.random.Generator) -> None:
        """On lap 1 only, launch the field and strike the first-corner incidents.

        Args:
            state: Parallel race state, mutated in place.
            ctx: Facts about the race; unused.
            rng: Seeded random generator.
        """
        del ctx
        if state.lap != 1:
            return
        apply_launch(
            state,
            self.launch_places,
            rng,
            place_s=self.place_s,
            noise_places=self.noise_places,
        )
        crashed = (rng.random(state.race_time_s.shape) < self.incident) & state.running
        state.retired = state.retired | crashed
        state.race_time_s = np.where(crashed, np.inf, state.race_time_s)
        update_order(state)
