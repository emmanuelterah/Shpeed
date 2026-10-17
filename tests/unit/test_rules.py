"""The season-rules schema validates the shipped files and rejects bad ones (P4-06)."""

from pathlib import Path

import pytest
import yaml

from f1sim.adapters.config_files import load_season_rules
from f1sim.core.domain.rules import PowerUnit, SeasonRules
from f1sim.core.errors import DataError, ValidationError

RULES_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"
SHIPPED_SEASONS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)


def _rules(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "season": 2024,
        "race_distance_km": 305.0,
        "dry_compounds_per_event": 3,
        "mandatory_compound_change": True,
        "overtake_aid": "drs",
        "active_aero": False,
        "sprint_format": None,
        "power_unit": {"mguh": True, "mguk_kw": 120.0},
        "notes": "",
    }
    return base | overrides


def test_every_shipped_season_loads() -> None:
    for season in SHIPPED_SEASONS:
        assert load_season_rules(season, RULES_DIR).season == season


def test_2026_reset_is_captured() -> None:
    rules = load_season_rules(2026, RULES_DIR)
    assert rules.overtake_aid == "overtake_mode"
    assert rules.active_aero is True
    assert rules.sprint_format == "sprint_qualifying"
    assert rules.power_unit.mguh is False
    assert rules.power_unit.mguk_kw > 200.0


def test_earliest_season_uses_drs_and_no_sprint() -> None:
    rules = load_season_rules(2018, RULES_DIR)
    assert rules.overtake_aid == "drs"
    assert rules.sprint_format is None


def test_round_trip_through_mapping() -> None:
    rules = load_season_rules(2024, RULES_DIR)
    assert SeasonRules.from_mapping(rules.to_dict()) == rules


def test_missing_season_file_raises() -> None:
    with pytest.raises(DataError):
        load_season_rules(1990, RULES_DIR)


def test_file_naming_another_season_raises(tmp_path: Path) -> None:
    (tmp_path / "2020.yaml").write_text(yaml.safe_dump(_rules(season=2019)), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_season_rules(2020, tmp_path)


def test_out_of_range_values_raise() -> None:
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(season=1200))
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(race_distance_km=100.0))
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(dry_compounds_per_event=9))


def test_bad_choices_raise() -> None:
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(overtake_aid="kers"))
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(sprint_format="sprint_race"))


def test_bad_types_raise() -> None:
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(dry_compounds_per_event="three"))
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(mandatory_compound_change="yes"))
    with pytest.raises(ValidationError):
        SeasonRules.from_mapping(_rules(power_unit=5))


def test_power_unit_range_raises() -> None:
    with pytest.raises(ValidationError):
        PowerUnit(mguk_kw=900.0)


def test_power_unit_round_trip() -> None:
    unit = PowerUnit(mguh=False, mguk_kw=350.0)
    assert PowerUnit.from_mapping(unit.to_dict(), "test") == unit
