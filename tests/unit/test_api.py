"""The read-only prediction API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from f1sim.adapters.publishers.local_file import LocalFilePublisher
from f1sim.adapters.records import validate_index, validate_record
from f1sim.api.app import create_app
from f1sim.settings import Settings
from tests.records import make_record, make_score


def _write(root: Path, season: int, round_number: int) -> None:
    folder = root / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "season": season, "round": round_number, "drivers": []}
    (folder / f"{round_number:02d}.json").write_text(json.dumps(record))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A test client for an API serving an empty predictions folder."""
    return TestClient(create_app(Settings(predictions_dir=tmp_path)))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_latest_is_404_before_anything_is_published(client: TestClient) -> None:
    assert client.get("/predictions/latest").status_code == 404


def test_latest_returns_most_recent_round_across_seasons(tmp_path: Path) -> None:
    _write(tmp_path, 2026, 24)
    _write(tmp_path, 2027, 2)
    _write(tmp_path, 2027, 10)
    client = TestClient(create_app(Settings(predictions_dir=tmp_path)))
    assert client.get("/predictions/latest").json()["round"] == 10
    assert client.get("/predictions").json() == [
        {"season": 2026, "round": 24},
        {"season": 2027, "round": 2},
        {"season": 2027, "round": 10},
    ]


def test_get_by_season_and_round(tmp_path: Path) -> None:
    _write(tmp_path, 2027, 3)
    client = TestClient(create_app(Settings(predictions_dir=tmp_path)))
    assert client.get("/predictions/2027/3").json()["round"] == 3
    assert client.get("/predictions/2027/4").status_code == 404


def test_rejects_non_numeric_paths(client: TestClient) -> None:
    assert client.get("/predictions/2027/..%2F..%2Fetc").status_code in (404, 422)


def test_index_is_404_before_anything_is_published(client: TestClient) -> None:
    assert client.get("/predictions/index").status_code == 404


def test_served_records_and_index_satisfy_the_shared_schemas(tmp_path: Path) -> None:
    publisher = LocalFilePublisher(tmp_path)
    publisher.publish(make_record(round_=1).with_score(make_score([1, 2, None])).to_mapping())
    publisher.publish(make_record(round_=2).to_mapping())
    client = TestClient(create_app(Settings(predictions_dir=tmp_path)))
    validate_record(client.get("/predictions/2024/1").json())
    validate_record(client.get("/predictions/latest").json())
    index = client.get("/predictions/index").json()
    validate_index(index)
    assert index["latest"] == "2024/02.json"
    assert index["calibration"]["races_scored"] == 1
