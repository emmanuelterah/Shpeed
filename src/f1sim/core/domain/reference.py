"""Reference data the pace model uses: circuit speed clusters and engine suppliers."""

from collections.abc import Mapping
from dataclasses import dataclass, field

CLUSTERS = ("low", "medium", "high")
"""Speed clusters, slowest first."""

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class CircuitTable:
    """Lap lengths and the speed bands that define circuit clusters (P3-06).

    Attributes:
        lengths_km: Circuit ID to lap length in kilometres.
        low_below: Average speeds below this, in km/h, are the ``low`` cluster.
        high_from: Average speeds at or above this, in km/h, are the ``high`` cluster.
    """

    lengths_km: Mapping[str, float]
    low_below: float = 210.0
    high_from: float = 235.0

    def average_speed(self, circuit_id: str, lap_time_s: float) -> float | None:
        """Average speed of a lap.

        Args:
            circuit_id: Jolpica circuit ID.
            lap_time_s: Lap time in seconds.

        Returns:
            Speed in km/h, or None for an unknown circuit or a non-positive time.
        """
        length = self.lengths_km.get(circuit_id)
        if length is None or lap_time_s <= 0:
            return None
        return length / (lap_time_s / SECONDS_PER_HOUR)

    def cluster(self, speed_kmh: float | None) -> str:
        """Place an average speed in a cluster.

        Args:
            speed_kmh: Average qualifying speed, or None when unknown.

        Returns:
            ``low``, ``medium`` or ``high``; ``medium`` when the speed is unknown.
        """
        if speed_kmh is None:
            return "medium"
        if speed_kmh < self.low_below:
            return "low"
        return "high" if speed_kmh >= self.high_from else "medium"


@dataclass(frozen=True)
class EngineSuppliers:
    """Power-unit supplier of each team, by season (P3-08).

    Attributes:
        defaults: Team ID to supplier, for seasons without an override.
        seasons: Season to team ID to supplier, overriding ``defaults``.
    """

    defaults: Mapping[str, str]
    seasons: Mapping[int, Mapping[str, str]] = field(default_factory=dict[int, Mapping[str, str]])

    def supplier_of(self, season: int, team_id: str) -> str:
        """Find a team's supplier in a season.

        Args:
            season: Championship year.
            team_id: Jolpica constructor ID.

        Returns:
            The supplier ID; a team that is not listed is its own supplier.
        """
        override = self.seasons.get(season, {})
        return override.get(team_id) or self.defaults.get(team_id) or team_id
