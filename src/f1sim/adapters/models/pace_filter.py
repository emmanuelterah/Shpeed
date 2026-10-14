"""The Bayesian pace model as a linear-Gaussian state-space model (P3-04 to P3-09).

Every observation is a sum of latent effects plus noise, all in percent of lap time:

``pace = team + supplier + team x circuit cluster + (qualifying: team quali offset +
driver quali effect | race: driver race effect) + session level + noise``

- **Team** pace follows a random walk from race to race, with a bigger step at the start
  of each season, because cars are developed during the year and rebuilt over winter.
  Teams are tracked by lineage, so a rebrand keeps its history.
- **Supplier** pace (engine) is shared by works and customer teams and changes between
  seasons.
- **Team x cluster** lets a car be stronger at slow or fast circuits.
- **Driver** effects come in a correlated pair, qualifying and race, partially pooled
  across seasons: each new season adds a little variance rather than starting over. A
  driver's effect is identified against their teammate, who shares the car.
- **Session level** absorbs the session's reference (its median car); it gets a vague
  prior and is integrated out, so only differences between drivers inform the model.

With the hyperparameters fixed, the posterior over every latent effect is exactly Gaussian,
and updating it with a new session is a Kalman filter step: exact, and milliseconds fast.
That is what makes weekend updating cheap. The hyperparameters themselves are fitted by
NUTS in ``pace_hyper`` (ADR-0003).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from f1sim.adapters.models.pace_data import QUALI, SessionObs
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.reference import EngineSuppliers
from f1sim.core.errors import DataError

SESSION_LEVEL_SD = 10.0
"""Prior standard deviation of a session's level, in percent: effectively flat."""

JITTER = 1e-9
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class WalkHyper:
    """Spread and drift of an effect that changes over time, in percent of lap time.

    Attributes:
        sd: Spread when the effect first appears.
        step: Random-walk step between races; 0 for effects fixed within a season.
        season_step: Extra step at the start of a season.
    """

    sd: float
    step: float
    season_step: float


@dataclass(frozen=True)
class DriverHyper:
    """Driver effects, in percent of lap time.

    Attributes:
        quali_sd: Spread of qualifying effects.
        race_sd: Spread of race effects.
        corr: Correlation between a driver's qualifying and race effects.
        season_step: Change in both effects from one season to the next.
    """

    quali_sd: float = 0.3
    race_sd: float = 0.3
    corr: float = 0.7
    season_step: float = 0.1


@dataclass(frozen=True)
class NoiseHyper:
    """Observation noise by session group, in percent of lap time.

    Attributes:
        q: Qualifying and sprint qualifying.
        r: Races.
        s: Sprints.
        fp: Practice long runs.
    """

    q: float = 0.1
    r: float = 0.12
    s: float = 0.15
    fp: float = 0.35


FLAT_NAMES: dict[str, tuple[str, str]] = {
    "team_sd": ("team", "sd"),
    "team_step": ("team", "step"),
    "team_season_step": ("team", "season_step"),
    "supplier_sd": ("supplier", "sd"),
    "supplier_step": ("supplier", "step"),
    "supplier_season_step": ("supplier", "season_step"),
    "quali_offset_sd": ("", "quali_offset_sd"),
    "cluster_sd": ("", "cluster_sd"),
    "driver_quali_sd": ("driver", "quali_sd"),
    "driver_race_sd": ("driver", "race_sd"),
    "driver_corr": ("driver", "corr"),
    "driver_season_step": ("driver", "season_step"),
    "noise_q": ("noise", "q"),
    "noise_r": ("noise", "r"),
    "noise_s": ("noise", "s"),
    "noise_fp": ("noise", "fp"),
    "order_noise": ("", "order_noise"),
}
"""Flat name of every hyperparameter (as in JSON files and NUTS) to its place here."""


@dataclass(frozen=True)
class PaceHyper:
    """Hyperparameters of the pace model, in percent of lap time.

    Attributes:
        team: Team pace: spread, per-race drift and season-start drift.
        supplier: Engine supplier pace; developed between seasons, not within them.
        quali_offset_sd: Team qualifying-versus-race offset; re-drawn in part each season.
        cluster_sd: Team x circuit-cluster effect; re-drawn in part each season.
        driver: Driver effects.
        noise: Observation noise by session group.
        order_noise: Race-day pace noise used when ranking a forecast race.
    """

    team: WalkHyper = WalkHyper(1.0, 0.1, 0.5)
    supplier: WalkHyper = WalkHyper(0.3, 0.0, 0.2)
    quali_offset_sd: float = 0.15
    cluster_sd: float = 0.15
    driver: DriverHyper = DriverHyper()
    noise: NoiseHyper = NoiseHyper()
    order_noise: float = 0.4

    def noise_sd(self, group: str) -> float:
        """Observation noise of a session group.

        Args:
            group: ``Q``, ``R``, ``S`` or ``FP``.

        Returns:
            The noise standard deviation, in percent.
        """
        n = self.noise
        return {"Q": n.q, "R": n.r, "S": n.s}.get(group, n.fp)

    def to_dict(self) -> dict[str, float]:
        """Flatten for a JSON file, using the names in ``FLAT_NAMES``.

        Returns:
            Flat name to value.
        """
        out: dict[str, float] = {}
        for name, (group, attr) in FLAT_NAMES.items():
            holder = getattr(self, group) if group else self
            out[name] = float(getattr(holder, attr))
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaceHyper":
        """Read flat hyperparameters, keeping defaults for missing names.

        Args:
            data: Flat name to value; unknown names are ignored.

        Returns:
            The hyperparameters.

        Raises:
            DataError: If a value is not a number, is negative, or ``driver_corr`` is not
                between -1 and 1.
        """
        try:
            values = {k: float(v) for k, v in data.items() if k in FLAT_NAMES}
        except (TypeError, ValueError) as exc:
            raise DataError(f"pace hyperparameters must be numbers: {exc}") from exc
        if any(v < 0 for k, v in values.items() if k != "driver_corr"):
            raise DataError("pace hyperparameters must be non-negative")
        if abs(values.get("driver_corr", 0.0)) >= 1.0:
            raise DataError("driver_corr must be between -1 and 1")
        merged = cls().to_dict() | values
        groups: dict[str, dict[str, float]] = {
            "team": {},
            "supplier": {},
            "driver": {},
            "noise": {},
        }
        top: dict[str, float] = {}
        for name, value in merged.items():
            group, attr = FLAT_NAMES[name]
            (groups[group] if group else top)[attr] = value
        return cls(
            team=WalkHyper(**groups["team"]),
            supplier=WalkHyper(**groups["supplier"]),
            driver=DriverHyper(**groups["driver"]),
            noise=NoiseHyper(**groups["noise"]),
            **top,
        )


@dataclass
class PaceState:
    """The Gaussian posterior over every latent effect seen so far.

    Attributes:
        labels: One name per effect, such as ``team:woking`` or ``dr:max_verstappen``.
        mean: Posterior mean of each effect.
        cov: Posterior covariance.
        last_event: ``(season, round)`` of the last session processed, or None.
    """

    labels: list[str] = field(default_factory=list[str])
    mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cov: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    last_event: tuple[int, int] | None = None

    def index(self) -> dict[str, int]:
        """Position of each label.

        Returns:
            Label to row of ``mean``.
        """
        return {label: i for i, label in enumerate(self.labels)}

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the fitted-state store.

        Returns:
            A JSON-compatible dictionary.
        """
        return {
            "labels": list(self.labels),
            "mean": self.mean.tolist(),
            "cov": self.cov.tolist(),
            "last_event": list(self.last_event) if self.last_event else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaceState":
        """Rebuild a state saved with ``to_dict``.

        Args:
            data: The saved dictionary.

        Returns:
            The state.

        Raises:
            DataError: If the shapes do not match the labels.
        """
        labels = [str(v) for v in cast(list[Any], data["labels"])]
        mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)
        cov = np.asarray(data["cov"], dtype=np.float64).reshape(len(mean), -1)
        if len(labels) != len(mean) or cov.shape != (len(mean), len(mean)):
            raise DataError("saved pace state has inconsistent shapes")
        last = cast(list[int] | None, data.get("last_event"))
        return cls(labels, mean, cov, (int(last[0]), int(last[1])) if last else None)


@dataclass(frozen=True)
class Predictive:
    """The model's forecast of one session, made before seeing it.

    Attributes:
        residuals: Observed minus forecast pace, each centred on the session mean.
        sd: Forecast standard deviation of each centred residual.
        mahalanobis: Squared standardised distance of the whole session, which is
            chi-squared with ``n - 1`` degrees of freedom if the model is right.
        log_density: Log predictive density of the session's values, with the session
            level integrated out; summed over sessions it is the model's log likelihood.
    """

    residuals: NDArray[np.float64]
    sd: NDArray[np.float64]
    mahalanobis: float
    log_density: float


class PaceFilter:
    """Kalman filter over the latent pace effects."""

    def __init__(
        self,
        hyper: PaceHyper,
        lineage: TeamLineage,
        suppliers: EngineSuppliers,
        state: PaceState | None = None,
    ) -> None:
        """Start from an empty or saved posterior.

        Args:
            hyper: Hyperparameters.
            lineage: Maps constructor IDs to continuing teams.
            suppliers: Power-unit supplier of each team and season.
            state: A saved posterior to continue from; empty when None.
        """
        self.hyper = hyper
        self._lineage = lineage
        self._suppliers = suppliers
        self.state = state or PaceState()
        self._index = self.state.index()

    def advance(self, season: int, round_: int) -> None:
        """Let the latent effects drift up to a new race weekend.

        Args:
            season: Season of the weekend.
            round_: Round of the weekend.
        """
        last = self.state.last_event
        if last == (season, round_):
            return
        if last is not None:
            h = self.hyper
            new_season = season != last[0]
            self._add_variance("team:", _step(h.team, new_season))
            self._add_variance("sup:", _step(h.supplier, new_season))
            if new_season:
                self._add_variance("tq:", h.quali_offset_sd**2)
                self._add_variance("tc:", h.cluster_sd**2)
                self._driver_step()
        self.state.last_event = (season, round_)

    def update(self, obs: SessionObs) -> Predictive:
        """Condition on one session's observations.

        Args:
            obs: The session.

        Returns:
            How well the model forecast the session before seeing it.
        """
        self.advance(obs.season, obs.round)
        rows = self._design(obs.season, obs.cluster, obs.kind, obs.drivers, obs.teams)
        n = len(obs.drivers)
        dim = len(self.state.mean)
        h = np.zeros((n, dim + 1))
        for i, row in enumerate(rows):
            h[i, row] = 1.0
        h[:, dim] = 1.0
        mean = np.append(self.state.mean, 0.0)
        cov = np.zeros((dim + 1, dim + 1))
        cov[:dim, :dim] = self.state.cov
        cov[dim, dim] = SESSION_LEVEL_SD**2
        noise = self.hyper.noise_sd(obs.group) ** 2
        s = h @ cov @ h.T + noise * np.eye(n)
        innovation = obs.values - h @ mean
        predictive = _predictive(
            obs.values, h, mean, cov, noise, log_density=_log_density(innovation, s)
        )
        gain = np.linalg.solve(s, h @ cov).T
        mean = mean + gain @ innovation
        cov = cov - gain @ h @ cov
        self.state.mean = mean[:dim]
        self.state.cov = 0.5 * (cov[:dim, :dim] + cov[:dim, :dim].T)
        return predictive

    def design(self, obs: SessionObs) -> list[list[int]]:
        """Register a session's effects without updating, for building the NUTS panel.

        Args:
            obs: The session.

        Returns:
            For each driver, the state indices of the effects in their observation.
        """
        return self._design(obs.season, obs.cluster, obs.kind, obs.drivers, obs.teams)

    def race_pace(
        self, season: int, cluster: str, drivers: Sequence[str], teams: Sequence[str]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forecast race pace for some drivers at the current weekend.

        Args:
            season: Season of the race.
            cluster: Speed cluster of the circuit.
            drivers: Driver IDs.
            teams: Each driver's constructor ID.

        Returns:
            Mean and covariance of race pace, in percent; lower is faster.
        """
        rows = self._design(season, cluster, "r", drivers, teams)
        h = np.zeros((len(drivers), len(self.state.mean)))
        for i, row in enumerate(rows):
            h[i, row] = 1.0
        return h @ self.state.mean, h @ self.state.cov @ h.T

    def effect(self, label: str) -> tuple[float, float]:
        """Posterior mean and standard deviation of one effect.

        Args:
            label: Such as ``team:woking`` or ``dr:max_verstappen``.

        Returns:
            ``(mean, sd)``; ``(0, nan)`` for an effect not seen yet.
        """
        i = self._index.get(label)
        if i is None:
            return 0.0, float("nan")
        return float(self.state.mean[i]), float(np.sqrt(self.state.cov[i, i]))

    def _design(
        self,
        season: int,
        cluster: str,
        kind: str,
        drivers: Sequence[str],
        teams: Sequence[str],
    ) -> list[list[int]]:
        rows: list[list[int]] = []
        for driver, team_id in zip(drivers, teams, strict=True):
            team = self._lineage.lineage_of(team_id)
            supplier = self._suppliers.supplier_of(season, team_id)
            labels = [f"team:{team}", f"sup:{supplier}", f"tc:{team}:{cluster}"]
            labels += [f"tq:{team}", f"dq:{driver}"] if kind == QUALI else [f"dr:{driver}"]
            self._ensure_driver(driver)
            for label in labels:
                if not label.startswith(("dq:", "dr:")):
                    self._ensure(label)
            rows.append([self._index[label] for label in labels])
        return rows

    def _prior_sd(self, label: str) -> float:
        h = self.hyper
        prefix = label.split(":", 1)[0]
        return {
            "team": h.team.sd,
            "sup": h.supplier.sd,
            "tq": h.quali_offset_sd,
            "tc": h.cluster_sd,
        }[prefix]

    def _ensure(self, label: str) -> None:
        if label not in self._index:
            self._grow([label], np.array([[self._prior_sd(label) ** 2]]))

    def _ensure_driver(self, driver: str) -> None:
        if f"dq:{driver}" in self._index:
            return
        self._grow([f"dq:{driver}", f"dr:{driver}"], self._driver_cov())

    def _driver_cov(self) -> NDArray[np.float64]:
        d = self.hyper.driver
        sq, sr = d.quali_sd, d.race_sd
        cross = d.corr * sq * sr
        return np.array([[sq**2, cross], [cross, sr**2]])

    def _driver_step(self) -> None:
        d = self.hyper.driver
        step = d.season_step**2 * np.array([[1.0, d.corr], [d.corr, 1.0]])
        for label in [lab for lab in self.state.labels if lab.startswith("dq:")]:
            i = self._index[label]
            j = self._index["dr:" + label[3:]]
            self.state.cov[np.ix_([i, j], [i, j])] += step

    def _add_variance(self, prefix: str, variance: float) -> None:
        idx = [i for i, label in enumerate(self.state.labels) if label.startswith(prefix)]
        self.state.cov[idx, idx] += variance

    def _grow(self, labels: list[str], block: NDArray[np.float64]) -> None:
        dim = len(self.state.mean)
        k = len(labels)
        cov = np.zeros((dim + k, dim + k))
        cov[:dim, :dim] = self.state.cov
        cov[dim:, dim:] = block + JITTER * np.eye(k)
        self.state.cov = cov
        self.state.mean = np.append(self.state.mean, np.zeros(k))
        for label in labels:
            self._index[label] = len(self.state.labels)
            self.state.labels.append(label)


def _step(walk: WalkHyper, new_season: bool) -> float:
    """Variance a random-walk effect gains between two weekends."""
    return walk.step**2 + (walk.season_step**2 if new_season else 0.0)


def _predictive(
    y: NDArray[np.float64],
    h: NDArray[np.float64],
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
    noise: float,
    *,
    log_density: float,
) -> Predictive:
    """Forecast of the session's centred values; centring removes the session level."""
    n = len(y)
    centre = np.eye(n) - np.full((n, n), 1.0 / n)
    hc = centre @ h[:, :-1]
    forecast_cov = hc @ cov[:-1, :-1] @ hc.T + noise * centre
    residuals = centre @ (y - h[:, :-1] @ mean[:-1])
    sd = np.sqrt(np.maximum(np.diag(forecast_cov), JITTER))
    distance = float(residuals @ np.linalg.pinv(forecast_cov, rcond=1e-10) @ residuals)
    return Predictive(residuals, sd, distance, log_density)


def _log_density(innovation: NDArray[np.float64], s: NDArray[np.float64]) -> float:
    """Log of the Gaussian density of ``innovation`` with covariance ``s``."""
    chol = np.linalg.cholesky(s)
    whitened = np.linalg.solve(chol, innovation)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return -0.5 * (float(whitened @ whitened) + log_det + len(innovation) * LOG_2PI)
