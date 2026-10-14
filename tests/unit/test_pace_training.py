"""NUTS hyperparameter fitting, race-day noise calibration and the statsmodels reference."""

import numpy as np
import pytest

from f1sim.adapters.config_files import load_team_lineage
from f1sim.adapters.models.pace_filter import PaceFilter, PaceHyper
from f1sim.adapters.models.pace_hyper import (
    GROUPS,
    HALF_NORMAL_SCALES,
    KINDS,
    HyperDiagnostics,
    build_panel,
    fit_hyper,
    log_likelihood,
)
from f1sim.adapters.models.pace_reference import fit_reference, prepare_laps
from f1sim.adapters.models.pace_training import SamplerSettings, season_observations, train_pace
from f1sim.adapters.models.registry import ModelOptions, pace_context
from f1sim.adapters.store.memory_store import InMemoryStore
from f1sim.core.errors import DataError
from tests.pace_world import build_world


@pytest.fixture(scope="module")
def store() -> InMemoryStore:
    """Two short synthetic seasons."""
    s = InMemoryStore()
    s.commit(build_world(seasons=(2023, 2024), rounds=4).tables, run_id="t")
    return s


def test_panel_lays_out_every_observation(store: InMemoryStore) -> None:
    context = pace_context(ModelOptions())
    observations = season_observations(store, store.latest_snapshot(), (2023, 2024), context)
    panel = build_panel(observations, context.lineage, context.suppliers)
    assert panel.observations == sum(len(o.drivers) for o in observations)
    assert panel.y.shape == panel.mask.shape == (len(observations), 10)
    assert panel.flags[:, 1].sum() == 1
    assert panel.flags[:, 0].sum() == 7
    assert panel.appear.sum() == panel.kind.shape[0]
    assert set(panel.group.tolist()) <= set(range(len(GROUPS)))
    assert set(panel.kind.tolist()) == set(range(len(KINDS)))
    assert panel.pairs.shape == (10, 2)
    with pytest.raises(DataError):
        build_panel([], context.lineage, context.suppliers)


def test_jax_likelihood_equals_the_filter(store: InMemoryStore) -> None:
    context = pace_context(ModelOptions())
    observations = season_observations(store, store.latest_snapshot(), (2023, 2024), context)
    hyper = PaceHyper.from_dict({"supplier_step": 0.05, "driver_corr": -0.3})
    pace = PaceFilter(hyper, context.lineage, context.suppliers)
    expected = sum(pace.update(obs).log_density for obs in observations)
    panel = build_panel(observations, context.lineage, context.suppliers)
    assert float(log_likelihood(panel, hyper.to_dict())) == pytest.approx(expected, abs=1e-4)


def test_laplace_agrees_with_the_truth(store: InMemoryStore) -> None:
    context = pace_context(ModelOptions())
    observations = season_observations(store, store.latest_snapshot(), (2023, 2024), context)
    fit = fit_hyper(build_panel(observations, context.lineage, context.suppliers), method="laplace")
    assert fit.method == "laplace"
    # Synthetic drivers never change between seasons, so that step has its mode at zero,
    # where a Gaussian approximation breaks down; the fit must say so.
    assert fit.diagnostics["driver_season_step"].median < 0.05
    assert fit.converged == all(
        0 < d.sd < 3 * scale
        for name, d in fit.diagnostics.items()
        for scale in [HALF_NORMAL_SCALES.get(name, 1.0)]
    )
    assert fit.hyper.team.step == pytest.approx(0.1, abs=0.05)
    assert fit.hyper.noise.q == pytest.approx(0.05, abs=0.03)
    with pytest.raises(DataError):
        fit_hyper(build_panel(observations, context.lineage, context.suppliers), method="vi")


def test_training_runs_nuts_and_picks_race_day_noise(store: InMemoryStore) -> None:
    context = pace_context(ModelOptions())
    result = train_pace(
        store,
        snapshot=store.latest_snapshot(),
        seasons=(2023, 2024),
        context=context,
        sampler=SamplerSettings(warmup=60, samples=60, chains=2, seed=1),
    )
    fit = result.fit
    assert set(fit.diagnostics) >= {"team_step", "noise_r", "driver_corr"}
    assert all(np.isfinite(d.r_hat) and d.ess > 0 for d in fit.diagnostics.values())
    scores = result.order_noise_rps
    assert fit.hyper.order_noise == min(scores, key=lambda value: scores[value])
    # Synthetic race laps carry about 0.07% of lap-time noise per driver average.
    assert fit.hyper.noise.r == pytest.approx(0.07, abs=0.04)
    record = result.record("abc")
    assert record["hyper"]["order_noise"] == fit.hyper.order_noise
    assert record["seasons"] == [2023, 2024]
    assert record["converged"] == fit.converged


def test_diagnostic_limits() -> None:
    assert HyperDiagnostics(1.0, 0.1, 1.01, 400).converged
    assert not HyperDiagnostics(1.0, 0.1, 1.2, 400).converged
    assert not HyperDiagnostics(1.0, 0.1, 1.0, 20).converged


def test_reference_model_beats_ignoring_drivers() -> None:
    laps = build_world(seasons=(2024,), rounds=4).tables["laps"]
    result = fit_reference(prepare_laps(laps, load_team_lineage()))
    assert result.rmse_pct < result.baseline_rmse_pct
    assert result.team_sd > result.driver_sd > 0
    assert len(result.team_effects) == 5
    with pytest.raises(DataError):
        fit_reference(prepare_laps(laps.head(50), load_team_lineage()))
