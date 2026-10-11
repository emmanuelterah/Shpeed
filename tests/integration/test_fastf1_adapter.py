from datetime import datetime

import pytest

pytest.importorskip("pandera")

from f1sim.adapters.sources.fastf1_adapter import FastF1Adapter
from f1sim.adapters.store.schemas import validate
from f1sim.core.domain.entities import EventKey, SessionKey, SessionType
from f1sim.core.errors import DataError
from tests.fixtures.fastf1_fakes import fake_session, loaders

pytestmark = pytest.mark.integration
RACE = SessionKey(EventKey(2024, 1), SessionType.RACE)


def _adapter(sessions, now=datetime(2025, 1, 1)) -> FastF1Adapter:  # type: ignore[no-untyped-def]
    load_session, load_schedule = loaders(sessions)
    return FastF1Adapter(load_session=load_session, load_schedule=load_schedule, clock=lambda: now)


def test_lists_finished_sessions_only() -> None:
    adapter = _adapter({}, now=datetime(2024, 3, 2, 12, 0))
    found = set(adapter.sessions(2024))
    assert SessionKey(EventKey(2024, 1), SessionType.QUALIFYING) in found
    assert RACE not in found
    assert all(s.event.round == 1 for s in found)
    assert adapter.sessions(2017) == []


def test_race_laps_weather_and_track_status_validate() -> None:
    adapter = _adapter({(2024, 1, "R"): fake_session()})
    tables = adapter.fetch(RACE)
    laps = validate("laps", tables["laps"])
    validate("weather", tables["weather"])
    status = validate("track_status", tables["track_status"])

    assert set(laps["driver_id"]) == {"max_verstappen", "norris"}
    assert laps["team_id"].dropna().unique().tolist() == ["red_bull", "mclaren"]
    ver = laps[laps["driver_id"] == "max_verstappen"].set_index("lap_number")
    assert bool(ver.loc[3, "pit_in"])
    assert bool(ver.loc[4, "pit_out"])
    # lap 1, the in-lap, the out-lap and the safety-car lap are not clean
    assert ver["clean_lap"].to_dict() == {1: False, 2: True, 3: False, 4: False, 5: True, 6: True}
    assert status["status"].tolist() == ["1", "4", "1"]
    assert laps["available_at"].iloc[0] == datetime(2024, 3, 2, 18, 0)


def test_red_flag_lap_time_is_nulled_not_rejected() -> None:
    adapter = _adapter({(2024, 1, "R"): fake_session(red_flag_lap=5, safety_car_lap=None)})
    laps = validate("laps", adapter.fetch(RACE)["laps"])
    lap5 = laps[laps["lap_number"] == 5]
    assert lap5["lap_time_s"].isna().all()
    assert not lap5["clean_lap"].any()


def test_missing_driver_id_falls_back_to_code() -> None:
    adapter = _adapter({(2024, 1, "R"): fake_session(drop_driver_id=True)})
    laps = validate("laps", adapter.fetch(RACE)["laps"])
    assert "code:nor" in set(laps["driver_id"])


def test_unavailable_session_raises_data_error() -> None:
    with pytest.raises(DataError, match="could not load"):
        _adapter({}).fetch(RACE)
