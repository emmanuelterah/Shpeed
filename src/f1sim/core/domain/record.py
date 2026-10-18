"""The versioned prediction record (P8-02): what is published for each race.

A record carries the forecast plus the provenance that makes it trustworthy: the snapshot,
model version, git commit, data cutoff and seed that produced it (ACID consistency). It is a
pure value object, so it can be built anywhere and serialised by an adapter.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np

from f1sim.core.domain.entities import EventKey, SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.errors import InvariantError

SCHEMA_VERSION = 1
"""Version of the shared record schema (``schemas/prediction-record.v1.schema.json``)."""

MIN_SEASON_YEAR = 1950
"""Earliest season a record may cover (the first Formula One championship)."""

DECIMALS = 8
"""Probabilities are stored to eight decimals, so a record equals its own JSON round trip.

Six decimals is not enough: across 22 drivers the rounding errors add up past the 1e-6
tolerance of :class:`PositionMatrix`, and a full-grid record would fail its own checks."""

SCORE_DECIMALS = 6
"""Scores are reported to six decimals."""

PROVENANCE_FIELDS = frozenset({"git_commit", "snapshot_id"})
"""Fields that say where a forecast came from without changing what it says (ADR-0011)."""


@dataclass(frozen=True, slots=True)
class DriverForecast:
    """One driver's forecast in a published record.

    Attributes:
        driver_id: Canonical driver ID, matching the data store.
        p_position: P(finish in position ``k``) for ``k = 1..n``, in order.
        p_dnf: Probability the driver does not finish.
        display_name: Name shown to readers; the driver ID when None.
        team_id: Canonical team ID; omitted when unknown.
        grid: Expected starting position; omitted when unknown.
    """

    driver_id: str
    p_position: tuple[float, ...]
    p_dnf: float
    display_name: str | None = None
    team_id: str | None = None
    grid: int | None = None


@dataclass(frozen=True, slots=True)
class RecordScore:
    """How a published forecast scored against the official result (P8-07).

    Attributes:
        rps: Ranked probability score over positions, with DNF as the last category.
        log_loss: Mean negative log probability of each driver's actual outcome.
        brier_win: Brier score of the win probabilities.
        brier_podium: Brier score of the podium probabilities.
        brier_top10: Brier score of the points (top 10) probabilities.
        scored_at: When the score was filled in.
        positions: Each driver's finishing position among the forecast drivers, in record
            order, or None for a driver who was not classified. Kept so calibration can be
            rebuilt from the published records alone (P8-10).
    """

    rps: float
    log_loss: float
    brier_win: float
    brier_podium: float
    brier_top10: float
    scored_at: datetime
    positions: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class PredictionRecord:  # pylint: disable=too-many-instance-attributes
    """One race's forecast plus everything needed to reproduce it.

    Attributes:
        season: Championship year.
        round: Round number within the season.
        data_cutoff: Latest moment whose data the model was allowed to see.
        snapshot_id: DuckLake snapshot the run read (ADR-0002).
        model_version: Version string of the model that produced the forecast.
        git_commit: Commit the run was built from.
        n_sims: Simulated races behind the forecast.
        seed: Random seed the run used.
        drivers: One forecast per starter, in entry (grid) order.
        schema_version: Record schema version; always :data:`SCHEMA_VERSION`.
        event_name: Human-readable event name, when known.
        p_rain: Rain probability fed to the simulator; omitted when not used.
        score: Filled in by the scoring run; None until then.
    """

    season: int
    round: int
    data_cutoff: datetime
    snapshot_id: SnapshotId
    model_version: str
    git_commit: str
    n_sims: int
    seed: int
    drivers: tuple[DriverForecast, ...]
    schema_version: int = SCHEMA_VERSION
    event_name: str | None = None
    p_rain: float | None = None
    score: RecordScore | None = None

    def __post_init__(self) -> None:
        """Check the record invariants.

        Raises:
            InvariantError: If the schema version is unknown, the season or round is out of
                range, there are no drivers, a driver repeats, the probabilities are not a
                valid set of distributions, or the score covers a different number of drivers.
        """
        if self.schema_version != SCHEMA_VERSION:
            raise InvariantError(f"unknown record schema version {self.schema_version}")
        if self.season < MIN_SEASON_YEAR or self.round < 1:
            raise InvariantError("season must be at least 1950 and round at least 1")
        if not self.drivers:
            raise InvariantError("a record must cover at least one driver")
        if len({d.driver_id for d in self.drivers}) != len(self.drivers):
            raise InvariantError("driver IDs must be unique")
        if self.score is not None and len(self.score.positions) != len(self.drivers):
            raise InvariantError("the score must give one finishing position per driver")
        self.matrix()

    @property
    def event(self) -> EventKey:
        """The race weekend this record forecasts.

        Returns:
            The event key.
        """
        return EventKey(self.season, self.round)

    @classmethod
    def from_matrix(  # pylint: disable=too-many-arguments
        cls,
        matrix: PositionMatrix,
        entries: Sequence[RaceEntry],
        *,
        season: int,
        round_: int,
        data_cutoff: datetime,
        snapshot_id: SnapshotId,
        model_version: str,
        git_commit: str,
        n_sims: int,
        seed: int,
        event_name: str | None = None,
        p_rain: float | None = None,
    ) -> "PredictionRecord":
        """Build a record from a fitted model's forecast and the race entries.

        Args:
            matrix: The forecast, with one row per entry in ``entries`` order.
            entries: The starters, in the same order as the matrix rows.
            season: Championship year.
            round_: Round number within the season.
            data_cutoff: Latest moment whose data the model saw.
            snapshot_id: Snapshot the run read.
            model_version: Version string of the model.
            git_commit: Commit the run was built from.
            n_sims: Simulated races behind the forecast.
            seed: Random seed the run used.
            event_name: Human-readable event name, when known.
            p_rain: Rain probability fed to the simulator, when used.

        Returns:
            The record.

        Raises:
            InvariantError: If the matrix rows do not match the entries.
        """
        if matrix.drivers != tuple(e.driver_id for e in entries):
            raise InvariantError("matrix drivers must match the race entries")
        drivers = tuple(
            DriverForecast(
                driver_id=entry.driver_id,
                p_position=tuple(round(float(v), DECIMALS) for v in matrix.p_position[i]),
                p_dnf=round(float(matrix.p_dnf[i]), DECIMALS),
                team_id=entry.team_id or None,
                grid=entry.grid or None,
            )
            for i, entry in enumerate(entries)
        )
        return cls(
            season=season,
            round=round_,
            data_cutoff=data_cutoff,
            snapshot_id=snapshot_id,
            model_version=model_version,
            git_commit=git_commit,
            n_sims=n_sims,
            seed=seed,
            drivers=drivers,
            event_name=event_name,
            p_rain=p_rain,
        )

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "PredictionRecord":
        """Rebuild a record from parsed JSON, as the scoring run reads it back.

        Args:
            record: A mapping valid against the shared schema.

        Returns:
            The record.

        Raises:
            InvariantError: If a required field is missing or malformed.
        """
        try:
            drivers = tuple(
                DriverForecast(
                    driver_id=str(row["driver_id"]),
                    p_position=tuple(float(v) for v in cast(Sequence[Any], row["p_position"])),
                    p_dnf=float(row["p_dnf"]),
                    display_name=_optional_str(row.get("display_name")),
                    team_id=_optional_str(row.get("team_id")),
                    grid=int(row["grid"]) if row.get("grid") is not None else None,
                )
                for row in cast(Sequence[Mapping[str, Any]], record["drivers"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvariantError(f"malformed prediction record: {exc}") from exc
        return cls(
            season=int(record["season"]),
            round=int(record["round"]),
            data_cutoff=_parse_time(str(record["data_cutoff"])),
            snapshot_id=SnapshotId(int(record["snapshot_id"])),
            model_version=str(record["model_version"]),
            git_commit=str(record["git_commit"]),
            n_sims=int(record["n_sims"]),
            seed=int(record["seed"]),
            drivers=drivers,
            schema_version=int(record.get("schema_version", SCHEMA_VERSION)),
            event_name=_optional_str(record.get("event_name")),
            p_rain=float(record["p_rain"]) if record.get("p_rain") is not None else None,
            score=_score_from_mapping(record.get("score")),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialise the record to a JSON-ready mapping.

        Returns:
            A mapping valid against the shared schema.
        """
        body: dict[str, Any] = {
            "schema_version": self.schema_version,
            "season": self.season,
            "round": self.round,
            "data_cutoff": _isoformat(self.data_cutoff),
            "snapshot_id": int(self.snapshot_id),
            "model_version": self.model_version,
            "git_commit": self.git_commit,
            "n_sims": self.n_sims,
            "seed": self.seed,
            "drivers": [self._driver_mapping(driver) for driver in self.drivers],
            "score": _score_mapping(self.score),
        }
        if self.event_name is not None:
            body["event_name"] = self.event_name
        if self.p_rain is not None:
            body["p_rain"] = self.p_rain
        return body

    def matrix(self) -> PositionMatrix:
        """Reconstruct the position matrix from the record.

        Returns:
            The forecast, with drivers in record order.

        Raises:
            InvariantError: If a driver's probabilities do not form a distribution.
        """
        p_position = np.array([d.p_position for d in self.drivers], dtype=np.float64)
        p_dnf = np.array([d.p_dnf for d in self.drivers], dtype=np.float64)
        return PositionMatrix(tuple(d.driver_id for d in self.drivers), p_position, p_dnf)

    def entries(self) -> tuple[RaceEntry, ...]:
        """Rebuild the race entries in forecast order, for scoring.

        Returns:
            One entry per driver; unknown teams and grids become empty defaults.
        """
        return tuple(
            RaceEntry(
                driver_id=d.driver_id,
                team_id=d.team_id or "",
                grid=d.grid or 0,
            )
            for d in self.drivers
        )

    def with_score(self, score: RecordScore) -> "PredictionRecord":
        """Return a copy with the score field filled in, leaving the forecast untouched.

        Args:
            score: The score to attach.

        Returns:
            A new record identical except for ``score``.
        """
        return replace(self, score=score)

    def forecast(self) -> dict[str, Any]:
        """The immutable part of the record: everything except the score.

        Returns:
            The mapping without its ``score`` key, for the immutability check (P8-03).
        """
        return {k: v for k, v in self.to_mapping().items() if k != "score"}

    def same_forecast(self, other: "PredictionRecord") -> bool:
        """Whether two records say the same thing about the same race (ADR-0011).

        Provenance (git commit, snapshot) is left out: a re-run from a newer commit or an
        unchanged newer snapshot that yields identical probabilities is the same forecast,
        so publishing it again is a no-op rather than a new commit.

        Args:
            other: The record to compare with.

        Returns:
            True when everything except the score and provenance matches.
        """
        mine, theirs = self.forecast(), other.forecast()
        return {k: v for k, v in mine.items() if k not in PROVENANCE_FIELDS} == {
            k: v for k, v in theirs.items() if k not in PROVENANCE_FIELDS
        }

    def _driver_mapping(self, driver: DriverForecast) -> dict[str, Any]:
        row: dict[str, Any] = {
            "driver_id": driver.driver_id,
            "p_position": [round(float(v), DECIMALS) for v in driver.p_position],
            "p_dnf": round(driver.p_dnf, DECIMALS),
        }
        if driver.display_name is not None:
            row["display_name"] = driver.display_name
        if driver.team_id is not None:
            row["team_id"] = driver.team_id
        if driver.grid is not None:
            row["grid"] = driver.grid
        return row


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _score_mapping(score: RecordScore | None) -> dict[str, Any] | None:
    if score is None:
        return None
    return {
        "brier_win": round(score.brier_win, SCORE_DECIMALS),
        "brier_podium": round(score.brier_podium, SCORE_DECIMALS),
        "brier_top10": round(score.brier_top10, SCORE_DECIMALS),
        "log_loss": round(score.log_loss, SCORE_DECIMALS),
        "rps": round(score.rps, SCORE_DECIMALS),
        "scored_at": _isoformat(score.scored_at),
        "positions": list(score.positions),
    }


def _score_from_mapping(value: Any) -> RecordScore | None:
    if value is None:
        return None
    row = cast(Mapping[str, Any], value)
    try:
        return RecordScore(
            brier_win=float(row["brier_win"]),
            brier_podium=float(row["brier_podium"]),
            brier_top10=float(row["brier_top10"]),
            log_loss=float(row["log_loss"]),
            rps=float(row["rps"]),
            scored_at=_parse_time(str(row["scored_at"])),
            positions=tuple(
                None if p is None else int(p) for p in cast(Sequence[Any], row["positions"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InvariantError(f"malformed score in prediction record: {exc}") from exc


def _isoformat(moment: datetime) -> str:
    aware = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    return aware.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_time(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed
