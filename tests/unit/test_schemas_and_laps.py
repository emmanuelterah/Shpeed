import pandas as pd
import pytest

pytest.importorskip("pandera")

from f1sim.adapters.store.schemas import validate
from f1sim.adapters.store.tables import TABLES
from f1sim.core.domain.laps import mark_clean_laps
from f1sim.core.errors import ValidationError
from tests.support import lap_rows, result_rows


def test_valid_laps_pass_and_come_back_in_canonical_order() -> None:
    df = lap_rows()
    checked = validate("laps", df.iloc[:, ::-1])
    assert list(checked.columns) == list(TABLES["laps"].columns)
    assert str(checked["lap_number"].dtype) == "Int64"


def test_implausible_lap_time_is_rejected() -> None:
    df = lap_rows()
    df.loc[0, "lap_time_s"] = 12.0
    with pytest.raises(ValidationError, match="laps"):
        validate("laps", df)


def test_duplicate_laps_are_rejected() -> None:
    df = pd.concat([lap_rows(), lap_rows().iloc[[0]]])
    with pytest.raises(ValidationError):
        validate("laps", df)


def test_stints_that_go_backwards_are_rejected() -> None:
    df = lap_rows()
    df.loc[df["lap_number"] == 5, "stint"] = 1
    df.loc[df["lap_number"] == 4, "stint"] = 2
    df.loc[df["lap_number"] == 2, "stint"] = 2
    with pytest.raises(ValidationError, match="stints"):
        validate("laps", df)


def test_unknown_columns_and_tables_are_rejected() -> None:
    with pytest.raises(ValidationError):
        validate("laps", lap_rows().assign(extra=1))
    with pytest.raises(ValidationError, match="Unknown table"):
        validate("telemetry", lap_rows())


def test_missing_required_value_is_rejected() -> None:
    df = result_rows()
    df.loc[0, "driver_id"] = None
    with pytest.raises(ValidationError):
        validate("results", df)


def test_clean_lap_rules() -> None:
    df = lap_rows(drivers=("max_verstappen",), n_laps=8)
    df.loc[df["lap_number"] == 6, "track_status"] = "14"  # safety car during lap 6
    df.loc[df["lap_number"] == 7, "track_status"] = "12"  # yellow only: still clean
    df.loc[df["lap_number"] == 8, "is_accurate"] = False
    clean = dict(zip(df["lap_number"], mark_clean_laps(df), strict=True))
    assert clean == {1: False, 2: True, 3: False, 4: False, 5: True, 6: False, 7: True, 8: False}
