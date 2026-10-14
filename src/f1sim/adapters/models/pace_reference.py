"""Mixed-effects reference model of lap times (P3-03), fitted with statsmodels.

``lap time (%) ~ compound + compound x tyre age + lap number + weekend``, with random
intercepts for each team and for each driver within a team. It is quick to fit and easy to
inspect, which makes it the sanity check for the Bayesian model: both should agree on who
is fast. Lap time is in percent of the weekend's median clean lap, so circuits share a
scale.

The reported error is on held-out laps: a random share of laps is set aside, the model is
fitted on the rest, and the held-out laps are predicted from fixed effects plus the
estimated team and driver effects. The same is done without team and driver effects, so
the gain from modelling who is driving is visible.
"""

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.formula.api
from numpy.typing import NDArray

from f1sim.adapters.models.pace_corrections import dry_laps
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.errors import DataError

HOLDOUT_SHARE = 0.2
PERCENT = 100.0
MIN_LAPS = 200
FORMULA = "y ~ C(compound) + C(compound):tyre_life + lap_number + C(weekend)"


@dataclass(frozen=True)
class ReferenceResult:
    """Held-out error of the mixed-effects model and of the same model without drivers.

    Attributes:
        train_laps: Laps the model was fitted on.
        test_laps: Held-out laps.
        rmse_pct: Held-out root mean squared error, percent of lap time.
        baseline_rmse_pct: The same without team and driver effects.
        team_sd: Estimated spread of team effects, percent.
        driver_sd: Estimated spread of driver-within-team effects, percent.
        team_effects: Estimated team effects (random intercepts), percent.
    """

    train_laps: int
    test_laps: int
    rmse_pct: float
    baseline_rmse_pct: float
    team_sd: float
    driver_sd: float
    team_effects: dict[str, float]


def prepare_laps(laps: pd.DataFrame, lineage: TeamLineage) -> pd.DataFrame:
    """Keep clean dry race laps and express them in percent of the weekend median.

    Args:
        laps: Canonical ``laps`` rows.
        lineage: Maps constructor IDs to continuing teams.

    Returns:
        Columns ``y``, ``compound``, ``tyre_life``, ``lap_number``, ``weekend``,
        ``team`` and ``driver``.
    """
    race = dry_laps(laps.loc[laps["session"] == "R"])
    weekend = race["season"].astype(str) + "-" + race["round"].astype(str)
    median = race.groupby(weekend)["lap_time_s"].transform("median")
    teams = [lineage.lineage_of(str(t)) for t in race["team_id"].tolist()]
    return pd.DataFrame(
        {
            "y": PERCENT * (race["lap_time_s"] / median - 1.0),
            "compound": race["compound"].astype(str).to_numpy(),
            "tyre_life": race["tyre_life"].fillna(0).astype(float).to_numpy(),
            "lap_number": race["lap_number"].astype(float).to_numpy(),
            "weekend": weekend.to_numpy(),
            "team": teams,
            "driver": race["driver_id"].astype(str).to_numpy(),
        }
    ).reset_index(drop=True)


def fit_reference(data: pd.DataFrame, *, seed: int = 0) -> ReferenceResult:
    """Fit the mixed model on most laps and score it on the rest.

    Args:
        data: Output of ``prepare_laps``.
        seed: Seed for choosing held-out laps.

    Returns:
        Held-out errors and the estimated effects.

    Raises:
        DataError: If there are too few laps.
    """
    if len(data) < MIN_LAPS:
        raise DataError(f"need at least {MIN_LAPS} clean race laps, got {len(data)}")
    rng = np.random.default_rng(seed)
    test = rng.random(len(data)) < HOLDOUT_SHARE
    train, held = data.loc[~test], data.loc[test]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = statsmodels.formula.api.mixedlm(
            FORMULA,
            train,
            groups=train["team"],
            re_formula="1",
            vc_formula={"driver": "0 + C(driver)"},
        ).fit(reml=True)
        plain = statsmodels.formula.api.ols(FORMULA, train).fit()
    effects = cast(dict[str, Any], mixed.random_effects)
    fixed = np.asarray(mixed.predict(held), dtype=np.float64)
    random_part = np.array(
        [
            _random_effect(effects, team, driver)
            for team, driver in zip(held["team"].tolist(), held["driver"].tolist(), strict=True)
        ]
    )
    y = held["y"].to_numpy(dtype=np.float64)
    baseline = np.asarray(plain.predict(held), dtype=np.float64)
    return ReferenceResult(
        train_laps=len(train),
        test_laps=len(held),
        rmse_pct=_rmse(y - fixed - random_part),
        baseline_rmse_pct=_rmse(y - baseline),
        team_sd=float(np.sqrt(max(float(np.asarray(mixed.cov_re)[0, 0]), 0.0))),
        driver_sd=float(np.sqrt(max(float(np.asarray(mixed.vcomp)[0]), 0.0))),
        team_effects={str(t): float(np.asarray(v)[0]) for t, v in effects.items()},
    )


def _random_effect(effects: dict[str, Any], team: str, driver: str) -> float:
    if team not in effects:
        return 0.0
    values = effects[team]
    total = float(np.asarray(values)[0])
    name = f"driver[C(driver)[{driver}]]"
    index = list(values.index)
    return total + (float(values[name]) if name in index else 0.0)


def _rmse(errors: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))
