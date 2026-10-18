"""The ablation runner and report (P6-09)."""

import numpy as np
import pytest

from f1sim.adapters.models.registry import ModelOptions, simulator_context
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.domain.entities import SnapshotId
from f1sim.reports.ablation import AblationResult, AblationRow, render_ablation, run_ablation
from tests.pace_world import world_store
from tests.unit.test_pace_model import FITTED

EVENT_SEASONS = (2024,)


@pytest.fixture(scope="module")
def store() -> InMemoryStore:
    """A small world store shared across the module's tests."""
    return world_store()


def test_run_ablation_scores_the_baseline_and_each_component(store: InMemoryStore) -> None:
    options = ModelOptions(pace_hyper=FITTED, n_sims=120)
    result = run_ablation(
        store,
        snapshot=store.latest_snapshot(),
        seasons=EVENT_SEASONS,
        seed=1,
        options=options,
        workers=1,
        components=("safety_car", "lap1"),
    )
    assert result.seasons == EVENT_SEASONS
    assert result.baseline_rps > 0.0
    assert [row.component for row in result.rows] == ["safety_car", "lap1"]
    for row in result.rows:
        assert np.isfinite(row.rps)
        assert row.delta_rps == pytest.approx(row.rps - result.baseline_rps)


def test_render_ablation_flags_a_harmful_component() -> None:
    result = AblationResult(
        seasons=EVENT_SEASONS,
        snapshot=SnapshotId(1),
        baseline_rps=0.2,
        rows=[
            AblationRow("lap1", "Lap 1 (P6-05)", 0.21, 0.01, False),
            AblationRow("rain", "Rain regime (P6-07)", 0.18, -0.02, True),
        ],
    )
    markdown = render_ablation(result)
    assert "| Lap 1 (P6-05)" in markdown
    assert "harmful" in markdown
    assert "noise band" in markdown
    assert result.harmful() == ("rain",)


def test_disabled_and_rain_reach_the_simulator_context() -> None:
    context = simulator_context(ModelOptions(disabled=frozenset({"rain"}), p_rain=0.3))
    assert context.disabled == frozenset({"rain"})
    assert context.p_rain == 0.3
