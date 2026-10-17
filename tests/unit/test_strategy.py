"""The strategy generator and chooser produce legal plans and pit windows (P4-07 to P4-09)."""

from pathlib import Path
from typing import Any

import pytest

from f1sim.adapters.config_files import load_season_rules
from f1sim.adapters.models.strategy import (
    PitWindow,
    StopLapSpread,
    StrategyChooser,
    StrategyGenerator,
    TeamTendencies,
    fit_team_tendencies,
)
from f1sim.adapters.models.tyres import TyreModel, fit_tyre_model
from f1sim.core.domain.rules import SeasonRules
from f1sim.core.errors import InvariantError
from tests.tyres_world import RACE_LAPS, TyresWorld, build_tyres_world

RULES_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"


@pytest.fixture(scope="module")
def world() -> TyresWorld:
    """Two synthetic seasons of two-stop races."""
    return build_tyres_world()


@pytest.fixture(scope="module")
def tyre(world: TyresWorld) -> TyreModel:
    """The tyre model fitted on every synthetic race."""
    tables = world.tables
    return fit_tyre_model(tables["laps"], tables["events"], tables["pit_stops"])


@pytest.fixture(scope="module")
def rules() -> SeasonRules:
    """The 2024 regulations."""
    return load_season_rules(2024, RULES_DIR)


@pytest.fixture(scope="module")
def tendencies(world: TyresWorld) -> TeamTendencies:
    """Team tendencies fitted on every synthetic race."""
    return fit_team_tendencies(world.tables["pit_stops"], world.tables["laps"])


def _generator(tyre: TyreModel, rules: SeasonRules, **overrides: Any) -> StrategyGenerator:
    options: dict[str, Any] = {
        "total_laps": RACE_LAPS,
        "base_lap_s": 95.0,
        "circuit": "bahrain",
        "team": "mercedes",
    }
    return StrategyGenerator(tyre, rules, **(options | overrides))


def test_plans_are_sorted_and_cover_every_stop_count(tyre: TyreModel, rules: SeasonRules) -> None:
    plans = _generator(tyre, rules).plans()
    assert plans
    assert [p.total_time_s for p in plans] == sorted(p.total_time_s for p in plans)
    assert {p.n_stops for p in plans} == {1, 2, 3}
    for plan in plans:
        assert len(plan.compounds) == plan.n_stops + 1
        assert len(set(plan.compounds)) >= 2


def test_start_compound_is_respected(tyre: TyreModel, rules: SeasonRules) -> None:
    plans = _generator(tyre, rules, start_compound="SOFT").plans()
    assert plans
    assert all(plan.compounds[0] == "SOFT" for plan in plans)


def test_sets_limit_the_compounds_used(tyre: TyreModel, rules: SeasonRules) -> None:
    generator = _generator(tyre, rules, max_stops=1, sets={"SOFT": 1, "MEDIUM": 1})
    plans = generator.plans()
    assert plans
    for plan in plans:
        assert set(plan.compounds) == {"SOFT", "MEDIUM"}


def test_a_race_shorter_than_a_stint_is_rejected(tyre: TyreModel, rules: SeasonRules) -> None:
    with pytest.raises(InvariantError):
        _generator(tyre, rules, total_laps=2)


def test_no_legal_plan_raises(tyre: TyreModel, rules: SeasonRules) -> None:
    generator = _generator(tyre, rules, sets={"SOFT": 0, "MEDIUM": 0, "HARD": 0})
    with pytest.raises(InvariantError):
        generator.plans()


def test_chooser_distribution_sums_to_one(
    tyre: TyreModel, rules: SeasonRules, tendencies: TeamTendencies
) -> None:
    plans = _generator(tyre, rules).plans()
    chooser = StrategyChooser(plans, tendencies)
    distribution = chooser.distribution("mercedes")
    assert len(distribution) == len(plans)
    assert sum(distribution) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in distribution)


def test_chooser_needs_plans(tendencies: TeamTendencies) -> None:
    with pytest.raises(InvariantError):
        StrategyChooser([], tendencies)


def test_stop_lap_distribution_is_normalised(
    tyre: TyreModel, rules: SeasonRules, tendencies: TeamTendencies
) -> None:
    plans = _generator(tyre, rules).plans()
    chooser = StrategyChooser(plans, tendencies)
    laps = chooser.stop_lap_distribution("mercedes", 0)
    assert laps
    assert sum(laps.values()) == pytest.approx(1.0)
    assert all(1 <= lap <= RACE_LAPS for lap in laps)


def test_windows_are_within_the_race(
    tyre: TyreModel, rules: SeasonRules, tendencies: TeamTendencies
) -> None:
    plans = _generator(tyre, rules).plans()
    chooser = StrategyChooser(plans, tendencies)
    windows = chooser.windows("mercedes")
    assert windows
    for window in windows:
        assert 1 <= window.low_lap <= window.high_lap <= RACE_LAPS


def test_stop_lap_spread_clips_to_the_race() -> None:
    spread = StopLapSpread({0: (-1.0, 1.0)}, (-0.1, 0.1))
    assert spread.range(0, 20, 50) == (1, 50)
    assert spread.range(1, 20, 50) == (15, 25)


def test_windows_with_a_spread_centre_on_the_mean_lap(
    tyre: TyreModel, rules: SeasonRules, tendencies: TeamTendencies
) -> None:
    plans = _generator(tyre, rules).plans()
    chooser = StrategyChooser(plans, tendencies)
    means = chooser.stop_lap_mean("mercedes")
    spread = StopLapSpread({0: (0.0, 0.0)}, (-0.1, 0.1))
    windows = chooser.windows("mercedes", spread=spread, total_laps=RACE_LAPS)
    assert windows[0] == PitWindow(0, round(means[0]), round(means[0]))


def test_windows_with_a_spread_need_the_race_distance(
    tyre: TyreModel, rules: SeasonRules, tendencies: TeamTendencies
) -> None:
    chooser = StrategyChooser(_generator(tyre, rules).plans(), tendencies)
    with pytest.raises(InvariantError):
        chooser.windows("mercedes", spread=StopLapSpread({}, (0.0, 0.0)))


def test_team_tendencies_cover_the_grid(tendencies: TeamTendencies, world: TyresWorld) -> None:
    teams = set(
        world.tables["pit_stops"].merge(
            world.tables["laps"][["season", "round", "driver_id", "team_id"]],
            on=["season", "round", "driver_id"],
        )["team_id"]
    )
    for team in teams:
        start = tendencies.for_team(team).start_compounds
        assert sum(start.values()) == pytest.approx(1.0)
    assert tendencies.for_team("nobody") is tendencies.default
