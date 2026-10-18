"""Recovery curves after a regulation reset (P7-03)."""

import pytest

from f1sim.adapters.models.registry import ModelOptions
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import SnapshotId
from f1sim.core.domain.natural_experiment import LAP_LEVEL, NaturalExperiment
from f1sim.reports.recovery import (
    STRATEGIES,
    STRATEGY_LABELS,
    RecoveryPoint,
    RecoveryStudy,
    StrategyCurve,
    render_recovery,
    run_recovery,
)
from tests.pace_world import world_store

EXPERIMENT = NaturalExperiment(
    experiment_id="reg-fake",
    name="Synthetic reset",
    level=LAP_LEVEL,
    seasons=(2021, 2022),
    reset_season=2022,
    reset_round=1,
)


@pytest.fixture(scope="module")
def store() -> InMemoryStore:
    """Two synthetic seasons, so 2022 races onto a rebuilt field."""
    return world_store(seasons=(2021, 2022), rounds=5)


def _study(store: InMemoryStore, **kwargs: object) -> RecoveryStudy:
    return run_recovery(
        store,
        snapshot=store.latest_snapshot(),
        experiment=EXPERIMENT,
        seed=7,
        options=ModelOptions(),
        workers=2,
        **kwargs,  # type: ignore[arg-type]
    )


def test_every_strategy_gets_a_curve_with_the_pre_reset_reference(
    store: InMemoryStore,
) -> None:
    study = _study(store)
    assert [curve.strategy for curve in study.curves] == list(STRATEGIES)
    assert [curve.label for curve in study.curves] == [STRATEGY_LABELS[s] for s in STRATEGIES]
    assert study.snapshot == store.latest_snapshot()
    for curve in study.curves:
        assert curve.pre_reset_rps is not None
        assert [p.race for p in curve.points] == [1, 2, 3, 4, 5]


def test_the_study_does_not_depend_on_the_workers(store: InMemoryStore) -> None:
    one = run_recovery(
        store,
        snapshot=store.latest_snapshot(),
        experiment=EXPERIMENT,
        seed=7,
        options=ModelOptions(),
        workers=1,
    )
    parallel = _study(store)
    assert [p.rps for p in one.curves[0].points] == [p.rps for p in parallel.curves[0].points]


def test_a_subset_of_strategies_can_be_compared(store: InMemoryStore) -> None:
    study = _study(store, strategies=("full", "reset"))
    assert [curve.strategy for curve in study.curves] == ["full", "reset"]


def test_recovery_race_finds_the_first_race_at_or_below_the_target() -> None:
    curve = StrategyCurve(
        "reset",
        "Reset priors",
        0.1300,
        [RecoveryPoint(1, 0.2000), RecoveryPoint(2, 0.1200), RecoveryPoint(3, 0.1000)],
    )
    assert curve.recovery_race(0.1300) == 2
    assert curve.recovery_race(0.0500) is None


def test_fastest_prefers_the_earlier_race_and_then_the_lower_score() -> None:
    study = RecoveryStudy(
        "reg-x",
        "X",
        LAP_LEVEL,
        2022,
        (2021, 2022),
        SnapshotId(1),
        [
            StrategyCurve(
                "full", "Full history", 0.13, [RecoveryPoint(1, 0.2), RecoveryPoint(2, 0.12)]
            ),
            StrategyCurve(
                "reset", "Reset priors", 0.13, [RecoveryPoint(1, 0.2), RecoveryPoint(2, 0.11)]
            ),
            StrategyCurve("decay", "Exponential decay", None, [RecoveryPoint(1, 0.05)]),
        ],
    )
    assert study.fastest() == "reset"


def test_render_reports_milestones_and_every_race(store: InMemoryStore) -> None:
    markdown = render_recovery(_study(store))
    assert "Synthetic reset (reg-fake)" in markdown
    assert "Ground" not in markdown
    assert "## Milestones" in markdown
    assert "## Per race" in markdown
    for label in STRATEGY_LABELS.values():
        assert label in markdown
    assert "| Race |" in markdown
    assert "| 5 |" in markdown
    assert markdown.endswith("\n")


def test_render_says_so_when_no_strategy_recovers() -> None:
    study = RecoveryStudy(
        "reg-x",
        "X",
        LAP_LEVEL,
        2022,
        (2021, 2022),
        SnapshotId(1),
        [StrategyCurve("full", "Full history", 0.01, [RecoveryPoint(1, 0.9)])],
    )
    assert "No strategy returned" in render_recovery(study)
