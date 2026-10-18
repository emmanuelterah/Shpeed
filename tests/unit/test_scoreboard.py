"""The predictions index: scoreboard rows and pooled calibration (P8-07, P8-10)."""

import pytest

from f1sim.adapters.records import validate_index
from f1sim.core.usecases.scoreboard import INDEX_SCHEMA_VERSION, build_index, calibration
from tests.records import make_record, make_score


def test_an_empty_folder_gives_an_empty_valid_index() -> None:
    index = build_index([])
    validate_index(index)
    assert index["latest"] is None
    assert index["races"] == []
    assert index["calibration"]["races_scored"] == 0
    assert [e["event"] for e in index["calibration"]["events"]] == ["win", "podium", "top10"]
    assert all(e["bins"] == [] for e in index["calibration"]["events"])


def test_races_are_sorted_and_the_latest_is_named() -> None:
    index = build_index(
        [
            ("2024/05.json", make_record(round_=5)),
            ("2023/09.json", make_record(season=2023, round_=9)),
            ("2024/01.json", make_record(round_=1)),
        ]
    )
    validate_index(index)
    assert [(r["season"], r["round"]) for r in index["races"]] == [(2023, 9), (2024, 1), (2024, 5)]
    assert index["latest"] == "2024/05.json"
    assert index["schema_version"] == INDEX_SCHEMA_VERSION


def test_a_scored_race_carries_its_metrics_but_not_the_positions() -> None:
    scored = make_record().with_score(make_score([1, 2, None], rps=0.25))
    index = build_index([("2024/03.json", scored), ("2024/04.json", make_record(round_=4))])
    validate_index(index)
    first, second = index["races"]
    assert first["score"]["rps"] == 0.25
    assert "positions" not in first["score"]
    assert first["event_name"] == "Test Grand Prix"
    assert first["n_drivers"] == 3
    assert second["score"] is None


def test_calibration_pools_every_scored_driver_forecast() -> None:
    records = [make_record(round_=r).with_score(make_score([1, 2, 3])) for r in range(1, 5)] + [
        make_record(round_=9)
    ]
    pooled = calibration(records)
    assert pooled["races_scored"] == 4
    assert pooled["forecasts"] == 12
    win = next(e for e in pooled["events"] if e["event"] == "win")
    assert sum(b["count"] for b in win["bins"]) == 12
    favourite = next(b for b in win["bins"] if b["low"] <= 0.5 < b["high"])
    assert favourite["mean_forecast"] == pytest.approx(0.5)
    assert favourite["observed"] == pytest.approx(1.0)
    assert favourite["count"] == 4


def test_a_dnf_counts_as_the_event_not_happening() -> None:
    scored = make_record().with_score(make_score([None, 1, 2]))
    win = next(e for e in calibration([scored])["events"] if e["event"] == "win")
    favourite = next(b for b in win["bins"] if b["low"] <= 0.5 < b["high"])
    assert favourite["observed"] == 0.0
