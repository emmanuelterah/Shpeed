"""Vectorised race state for the Monte Carlo engine (P5-01).

Every array has shape ``(n_sims, n_cars)``, so one state object holds many parallel races.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RaceState:
    """Mutable per-lap state for ``n_sims`` parallel races; LapSteps update it in place.

    Attributes:
        lap: Laps completed so far.
        position: Running order, from 1.
        gap_to_leader_s: Gap to the leader, in seconds.
        compound: Encoded tyre compound currently fitted.
        tyre_age: Laps on the current tyres.
        fuel_kg: Fuel on board.
        retired: Whether the car is out of the race.
    """

    lap: int
    position: NDArray[np.int_]
    gap_to_leader_s: NDArray[np.float64]
    compound: NDArray[np.int_]
    tyre_age: NDArray[np.int_]
    fuel_kg: NDArray[np.float64]
    retired: NDArray[np.bool_]

    @property
    def n_sims(self) -> int:
        """Number of parallel simulations."""
        return int(self.position.shape[0])

    @property
    def n_cars(self) -> int:
        """Number of cars in each simulation."""
        return int(self.position.shape[1])
