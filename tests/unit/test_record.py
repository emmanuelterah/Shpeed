"""PredictionRecord value objects: matrix round trip, invariants and scoring fields."""

from datetime import UTC, datetime

import numpy as np
import pytest

from f1sim.core.domain.entities import SnapshotId
from f1sim.core.domain.prediction import PositionMatrix
from f1sim.core.domain.race import RaceEntry
from f1sim.core.domain.record import (
    SCHEMA_VERSION,
    DriverForecast,
    PredictionRecord,
)
from f1sim.core.errors import InvariantError
from tests.records import CUTOFF, make_record, make_score


def record(
    *,
    drivers: tuple[str, ...] = ("a", "b"),
    season: int = 2024,
    round_: int = 3,
    git_commit: str = "abcdef0",
) -> PredictionRecord:
    """Build a two-driver record for the tests."""
    return make_record(drivers=drivers, season=season, round_=round_, git_commit=git_commit)


def test_mapping_round_trip_preserves_forecast() -> None:
    original = record()
    rebuilt = PredictionRecord.from_mapping(original.to_mapping())
    assert rebuilt == original
    assert rebuilt.schema_version == SCHEMA_VERSION
    assert rebuilt.event.season == 2024
    assert rebuilt.matrix().drivers == ("a", "b")


def test_forecast_excludes_the_score() -> None:
    scored = record().with_score(make_score([1, None], rps=0.4))
    assert "score" in scored.to_mapping()
    assert "score" not in scored.forecast()
    assert scored.forecast() == record().forecast()


def test_with_score_leaves_the_forecast_untouched() -> None:
    original = record()
    scored = original.with_score(make_score([1, None], rps=0.4))
    assert original.score is None
    assert scored.drivers == original.drivers
    assert scored.score is not None
    assert scored.score.rps == 0.4


def test_rejects_a_driver_whose_row_is_not_a_distribution() -> None:
    with pytest.raises(InvariantError):
        PredictionRecord(
            season=2024,
            round=1,
            data_cutoff=CUTOFF,
            snapshot_id=SnapshotId(1),
            model_version="m",
            git_commit="abcdef0",
            n_sims=1,
            seed=1,
            drivers=(DriverForecast("a", (0.9, 0.9), 0.9),),
        )


@pytest.mark.parametrize(("season", "round_"), [(1900, 1), (2024, 0)])
def test_rejects_out_of_range_event(season: int, round_: int) -> None:
    with pytest.raises(InvariantError):
        record(season=season, round_=round_)


def test_rejects_duplicate_drivers() -> None:
    with pytest.raises(InvariantError):
        record(drivers=("a", "a"))


def test_rejects_an_unknown_schema_version() -> None:
    with pytest.raises(InvariantError):
        PredictionRecord(
            season=2024,
            round=1,
            data_cutoff=CUTOFF,
            snapshot_id=SnapshotId(1),
            model_version="m",
            git_commit="abcdef0",
            n_sims=1,
            seed=1,
            drivers=(DriverForecast("a", (1.0,), 0.0),),
            schema_version=99,
        )


def test_from_mapping_rejects_malformed_json() -> None:
    with pytest.raises(InvariantError):
        PredictionRecord.from_mapping({"season": 2024})


def test_cutoff_round_trips_as_utc_isoformat() -> None:
    mapping = record().to_mapping()
    assert mapping["data_cutoff"] == "2024-03-23T16:00:00Z"
    assert PredictionRecord.from_mapping(mapping).data_cutoff == CUTOFF
    assert datetime.fromisoformat(mapping["data_cutoff"].replace("Z", "+00:00")).tzinfo == UTC


def test_score_round_trips_with_positions() -> None:
    scored = record().with_score(make_score([2, None]))
    mapping = scored.to_mapping()
    assert mapping["score"]["positions"] == [2, None]
    assert PredictionRecord.from_mapping(mapping) == scored


def test_a_score_must_cover_every_driver() -> None:
    with pytest.raises(InvariantError):
        record().with_score(make_score([1]))


def test_a_malformed_score_is_rejected() -> None:
    mapping = record().to_mapping()
    mapping["score"] = {"rps": 0.1}
    with pytest.raises(InvariantError):
        PredictionRecord.from_mapping(mapping)


def test_same_forecast_ignores_provenance_but_not_probabilities() -> None:
    assert record().same_forecast(record(git_commit="1234567"))
    assert not record().same_forecast(record(drivers=("a", "c")))
    assert record().same_forecast(record().with_score(make_score([1, 2])))


def test_a_full_grid_record_survives_its_own_round_trip() -> None:
    rng = np.random.default_rng(3)
    n = 22
    counts = np.zeros((n, n + 1))
    for _ in range(500):
        order = rng.permutation(n)
        out = rng.random(n) < 0.08
        finishers = [d for d in order if not out[d]]
        for k, d in enumerate(finishers):
            counts[d, k] += 1
        counts[out, n] += 1
    freq = counts / 500
    drivers = tuple(f"d{i:02d}" for i in range(n))
    matrix = PositionMatrix(drivers, freq[:, :n], freq[:, n])
    entries = tuple(RaceEntry(driver_id=d, team_id="t", grid=i) for i, d in enumerate(drivers, 1))
    built = PredictionRecord.from_matrix(
        matrix,
        entries,
        season=2026,
        round_=1,
        data_cutoff=CUTOFF,
        snapshot_id=SnapshotId(1),
        model_version="m",
        git_commit="abcdef0",
        n_sims=500,
        seed=1,
    )
    assert PredictionRecord.from_mapping(built.to_mapping()) == built
