"""Fit the pace model's hyperparameters with NUTS in NumPyro (P3-04 to P3-07, P3-11).

The pace model is linear and Gaussian once its hyperparameters are fixed, so its latent
effects (team, supplier, quali offset, team x cluster, driver) can be integrated out
exactly: the Kalman filter's log likelihood *is* the marginal likelihood of the
hyperparameters. NUTS therefore samples only the 15 hyperparameters, with the likelihood
computed by the same filter as ``pace_filter``, written in JAX. Two consequences:

- the hyperparameters are fitted for exactly the model the forecaster runs, including how
  new teams and drivers enter and how effects drift between races and seasons;
- the sampler works in 15 dimensions instead of thousands, so it mixes well and R-hat and
  effective sample size are meaningful.

R-hat and effective sample size are reported for every hyperparameter, so a fit that has
not converged is visible before it is used.
"""

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import numpyro
import numpyro.diagnostics
import numpyro.distributions
import numpyro.infer
import numpyro.infer.autoguide
import numpyro.optim
from numpy.typing import NDArray

from f1sim.adapters.models.pace_data import SessionObs
from f1sim.adapters.models.pace_filter import LOG_2PI, SESSION_LEVEL_SD, PaceFilter, PaceHyper
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.reference import EngineSuppliers
from f1sim.core.errors import DataError

# The filter needs double precision: session levels have a very wide prior.
jax.config.update("jax_enable_x64", True)

GROUPS = ("Q", "R", "S", "FP")
KINDS = ("team", "sup", "tq", "tc", "dq", "dr")
"""Effect types, by the label prefixes ``pace_filter`` uses."""

HALF_NORMAL_SCALES: dict[str, float] = {
    "team_sd": 1.5,
    "team_step": 0.3,
    "team_season_step": 1.0,
    "supplier_sd": 0.5,
    "supplier_season_step": 0.5,
    "quali_offset_sd": 0.3,
    "cluster_sd": 0.3,
    "driver_quali_sd": 0.5,
    "driver_race_sd": 0.5,
    "driver_season_step": 0.2,
    "noise_q": 0.5,
    "noise_r": 0.5,
    "noise_s": 0.5,
    "noise_fp": 1.0,
}
"""Half-normal prior scale of each hyperparameter, in percent of lap time."""

CORR_BOUND = 0.95
METHODS = ("nuts", "laplace")
"""``nuts`` samples the posterior; ``laplace`` finds its mode and curvature, much faster."""

LAPLACE_DRAWS = 4000
LAPLACE_SPREAD_LIMIT = 3.0
"""A Laplace fit is rejected if a hyperparameter's posterior spread exceeds this many
prior scales; the approximation has then broken down."""
HYPER_SITES = (*HALF_NORMAL_SCALES, "driver_corr")
NOISE_SITES = {"Q": "noise_q", "R": "noise_r", "S": "noise_s", "FP": "noise_fp"}
UNUSED_NOISE = 1.0
"""Stands in for the noise of a session group with no data; it never enters the fit."""
RHAT_LIMIT = 1.05
MIN_ESS = 100.0


@dataclass(frozen=True)
class Panel:
    """Sessions as padded arrays, ready for the JAX filter.

    Attributes:
        y: Observed pace per session and driver slot, shape ``(T, N)``; 0 where unused.
        mask: Which slots hold an observation, shape ``(T, N)``.
        design: Which effects each observation sums, shape ``(T, N, D)``.
        group: Noise group of each session, into ``GROUPS``, shape ``(T,)``.
        flags: Whether each session starts a new race weekend (column 0) and a new
            season (column 1), shape ``(T, 2)``.
        appear: Effects that first appear in each session, shape ``(T, D)``.
        kind: Type of each effect, into ``KINDS``, shape ``(D,)``.
        pairs: ``(qualifying, race)`` effect indices of each driver, shape ``(P, 2)``.
    """

    y: NDArray[np.float64]
    mask: NDArray[np.bool_]
    design: NDArray[np.float64]
    group: NDArray[np.int_]
    flags: NDArray[np.float64]
    appear: NDArray[np.float64]
    kind: NDArray[np.int_]
    pairs: NDArray[np.int_]

    @property
    def observations(self) -> int:
        """Number of observations.

        Returns:
            The count of used slots.
        """
        return int(self.mask.sum())


@dataclass(frozen=True)
class HyperDiagnostics:
    """Convergence of one hyperparameter.

    Attributes:
        median: Posterior median.
        sd: Posterior standard deviation.
        r_hat: Split R-hat across chains; near 1 when chains agree.
        ess: Effective sample size.
    """

    median: float
    sd: float
    r_hat: float
    ess: float

    @property
    def converged(self) -> bool:
        """Whether R-hat and ESS are within the limits.

        Returns:
            True when ``r_hat <= RHAT_LIMIT`` and ``ess >= MIN_ESS``.
        """
        return self.r_hat <= RHAT_LIMIT and self.ess >= MIN_ESS


@dataclass(frozen=True)
class HyperFit:
    """The result of a NUTS fit.

    Attributes:
        hyper: Posterior-median hyperparameters (``order_noise`` kept from the input).
        diagnostics: Per-hyperparameter summaries; R-hat and ESS are NaN for ``laplace``.
        seconds: Wall time of the fit.
        observations: Observations used.
        method: ``nuts`` or ``laplace``.
        optimum_found: For ``laplace``, whether the optimiser reached a finite optimum.
    """

    hyper: PaceHyper
    diagnostics: dict[str, HyperDiagnostics]
    seconds: float
    observations: int
    method: str = "nuts"
    optimum_found: bool = True

    @property
    def converged(self) -> bool:
        """Whether the fit can be trusted.

        Returns:
            For NUTS, True when every R-hat and ESS is within limits; for Laplace, True
            when the optimum is finite and the curvature is usable for every
            hyperparameter.
        """
        if self.method == "laplace":
            return self.optimum_found
        return all(d.converged for d in self.diagnostics.values())


def build_panel(
    observations: Sequence[SessionObs], lineage: TeamLineage, suppliers: EngineSuppliers
) -> Panel:
    """Lay out sessions for the JAX filter, with effects indexed as ``PaceFilter`` does.

    Args:
        observations: Sessions in calendar order.
        lineage: Maps constructor IDs to continuing teams.
        suppliers: Power-unit supplier of each team and season.

    Returns:
        The panel.

    Raises:
        DataError: If there are no observations.
    """
    if not observations:
        raise DataError("no pace observations to fit")
    rows, appeared, labels = _structure(observations, lineage, suppliers)
    shape = (len(observations), max(len(o.drivers) for o in observations), len(labels))
    y, mask = np.zeros(shape[:2]), np.zeros(shape[:2], dtype=bool)
    design, appear = np.zeros(shape), np.zeros((shape[0], shape[2]))
    for i, obs in enumerate(observations):
        y[i, : len(obs.values)] = obs.values
        mask[i, : len(obs.values)] = True
        for j, effects in enumerate(rows[i]):
            design[i, j, effects] = 1.0
        appear[i, list(appeared[i])] = 1.0
    return Panel(
        y=y,
        mask=mask,
        design=design,
        group=np.array([GROUPS.index(o.group) for o in observations], dtype=np.int64),
        flags=_flags(observations),
        appear=appear,
        kind=np.array([KINDS.index(label.split(":", 1)[0]) for label in labels]),
        pairs=_pairs(labels),
    )


def _pairs(labels: list[str]) -> NDArray[np.int_]:
    """``(qualifying, race)`` effect indices of each driver."""
    index = {label: i for i, label in enumerate(labels)}
    pairs = [(index[q], index["dr:" + q[3:]]) for q in labels if q.startswith("dq:")]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _structure(
    observations: Sequence[SessionObs], lineage: TeamLineage, suppliers: EngineSuppliers
) -> tuple[list[list[list[int]]], list[range], list[str]]:
    """Index effects exactly as ``PaceFilter`` does, noting where each first appears."""
    structure = PaceFilter(PaceHyper(), lineage, suppliers)
    rows: list[list[list[int]]] = []
    appeared: list[range] = []
    for obs in observations:
        before = len(structure.state.labels)
        rows.append(structure.design(obs))
        appeared.append(range(before, len(structure.state.labels)))
    return rows, appeared, structure.state.labels


def log_likelihood(panel: Panel, hp: dict[str, Any]) -> Any:
    """Log marginal likelihood of all sessions, by running the Kalman filter in JAX.

    Args:
        panel: The sessions.
        hp: Hyperparameters by flat name (``team_sd``, ``noise_r`` and so on), as JAX or
            Python numbers. ``supplier_step`` is optional and defaults to 0.

    Returns:
        The log likelihood, a JAX scalar. It equals the sum of ``Predictive.log_density``
        from ``PaceFilter`` with the same hyperparameters.
    """
    variances = _variances(panel, hp)
    noise = jnp.stack([hp["noise_q"], hp["noise_r"], hp["noise_s"], hp["noise_fp"]]) ** 2
    dim = panel.kind.shape[0]
    pair_q, pair_r = jnp.asarray(panel.pairs[:, 0]), jnp.asarray(panel.pairs[:, 1])

    def _cross(cov: Any, amount: Any) -> Any:
        return cov.at[pair_q, pair_r].add(amount).at[pair_r, pair_q].add(amount)

    def _drift(mean: Any, cov: Any, flags: Any, appear: Any) -> tuple[Any, Any]:
        """Drift effects to a new weekend or season, and start new effects at the prior."""
        cov = cov + jnp.diag(flags[0] * variances["event"] + flags[1] * variances["season"])
        cov = _cross(cov, flags[1] * variances["season_cross"])
        keep = 1.0 - appear
        cov = cov * keep[:, None] * keep[None, :] + jnp.diag(appear * variances["prior"])
        return mean * keep, _cross(cov, appear[pair_q] * variances["prior_cross"])

    def _update(mean: Any, cov: Any, x: Any) -> tuple[Any, Any, Any]:
        """Condition on one session, with its level integrated out."""
        y, mask, h, group = x
        both = mask[:, None] & mask[None, :]
        s = jnp.where(both, h @ cov @ h.T + SESSION_LEVEL_SD**2, 0.0)
        s = s + jnp.diag(jnp.where(mask, noise[group], 1.0))
        innovation = jnp.where(mask, y - h @ mean, 0.0)
        chol = jnp.linalg.cholesky(s)
        whitened = jax.scipy.linalg.solve_triangular(chol, innovation, lower=True)
        log_det = 2.0 * jnp.sum(jnp.where(mask, jnp.log(jnp.diag(chol)), 0.0))
        gain = jax.scipy.linalg.cho_solve((chol, True), h @ cov).T
        cov = cov - gain @ (h @ cov)
        log_p = -0.5 * (whitened @ whitened + log_det + mask.sum() * LOG_2PI)
        return mean + gain @ innovation, 0.5 * (cov + cov.T), log_p

    def _step(carry: Any, x: Any) -> Any:
        mean, cov = _drift(carry[0], carry[1], x[4], x[5])
        mean, cov, log_p = _update(mean, cov, x[:4])
        return (mean, cov), log_p

    arrays = (panel.y, panel.mask, panel.design, panel.group, panel.flags, panel.appear)
    _, log_p = jax.lax.scan(
        _step,
        (jnp.zeros(dim), jnp.zeros((dim, dim))),
        tuple(jnp.asarray(a) for a in arrays),
    )
    return jnp.sum(log_p)


def fitted_sites(panel: Panel) -> tuple[str, ...]:
    """Hyperparameters the data can inform.

    Args:
        panel: The sessions.

    Returns:
        Every hyperparameter except the noise of session groups with no sessions (a
        season without sprints says nothing about sprint noise).
    """
    present = {GROUPS[int(g)] for g in np.unique(panel.group)}
    unused = {site for group, site in NOISE_SITES.items() if group not in present}
    return tuple(site for site in HYPER_SITES if site not in unused)


def pace_model(panel: Panel) -> None:
    """Hyperpriors plus the exact marginal likelihood; see the module docstring.

    Args:
        panel: The sessions.
    """
    sites = fitted_sites(panel)
    hp: dict[str, Any] = {
        name: (
            numpyro.sample(name, numpyro.distributions.HalfNormal(scale))
            if name in sites
            else UNUSED_NOISE
        )
        for name, scale in HALF_NORMAL_SCALES.items()
    }
    hp["driver_corr"] = numpyro.sample(
        "driver_corr", numpyro.distributions.Uniform(-CORR_BOUND, CORR_BOUND)
    )
    numpyro.factor("log_likelihood", log_likelihood(panel, hp))


def fit_hyper(
    panel: Panel,
    *,
    base: PaceHyper | None = None,
    method: str = "nuts",
    sampler: tuple[int, int, int] = (500, 500, 2),
    seed: int = 0,
) -> HyperFit:
    """Fit the hyperparameters and summarise them.

    Args:
        panel: The sessions.
        base: Hyperparameters to keep for anything not fitted (``order_noise``,
            ``supplier_step``).
        method: ``nuts`` to sample the posterior, or ``laplace`` to approximate it by
            a Gaussian at its mode (minutes rather than hours on full seasons).
        sampler: NUTS ``(warm-up draws, kept draws, chains)``; chains run one after
            another and need to be at least two for R-hat. Ignored by ``laplace``.
        seed: Random seed.

    Returns:
        Posterior-median hyperparameters with their diagnostics.

    Raises:
        DataError: If ``method`` is unknown.
    """
    if method not in METHODS:
        raise DataError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")
    started = time.perf_counter()
    key = jax.random.PRNGKey(seed)
    sites = fitted_sites(panel)
    if method == "nuts":
        samples, stats = _nuts(panel, key, sampler, sites)
        found = True
    else:
        samples, found = _laplace(panel, key, sites)
        stats = {name: {"r_hat": float("nan"), "n_eff": float("nan")} for name in sites}
    diagnostics = {
        name: HyperDiagnostics(
            median=float(np.median(samples[name])),
            sd=float(np.std(samples[name])),
            r_hat=float(stats[name]["r_hat"]),
            ess=float(stats[name]["n_eff"]),
        )
        for name in sites
    }
    values = {name: d.median for name, d in diagnostics.items()}
    hyper = PaceHyper.from_dict((base or PaceHyper()).to_dict() | values)
    seconds = time.perf_counter() - started
    return HyperFit(hyper, diagnostics, seconds, panel.observations, method, found)


def _nuts(
    panel: Panel, key: Any, sampler: tuple[int, int, int], sites: tuple[str, ...]
) -> tuple[dict[str, NDArray[np.float64]], dict[str, dict[str, Any]]]:
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(pace_model, target_accept_prob=0.9),
        num_warmup=sampler[0],
        num_samples=sampler[1],
        num_chains=sampler[2],
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(key, panel)
    grouped: dict[str, Any] = mcmc.get_samples(group_by_chain=True)
    by_chain = {k: np.asarray(grouped[k], dtype=np.float64) for k in sites}
    stats = numpyro.diagnostics.summary(by_chain, group_by_chain=True)
    return {k: v.reshape(-1) for k, v in by_chain.items()}, stats


def _laplace(
    panel: Panel, key: Any, sites: tuple[str, ...]
) -> tuple[dict[str, NDArray[np.float64]], bool]:
    """Mode and curvature of the posterior, found with BFGS, then sampled.

    The fit counts as found only if the optimum is finite and the curvature is positive
    in every direction; a flat direction means some hyperparameter is not identified.
    """
    guide = numpyro.infer.autoguide.AutoLaplaceApproximation(pace_model)
    svi = numpyro.infer.SVI(pace_model, guide, numpyro.optim.Minimize(), numpyro.infer.Trace_ELBO())
    state = svi.init(key, panel)
    state, loss = svi.update(state, panel)
    params = svi.get_params(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        draws: dict[str, Any] = guide.sample_posterior(key, params, sample_shape=(LAPLACE_DRAWS,))
    samples = {k: np.asarray(draws[k], dtype=np.float64) for k in sites}
    finite = bool(np.isfinite(float(loss))) and all(np.isfinite(v).all() for v in samples.values())
    return samples, finite and all(_well_curved(k, v) for k, v in samples.items())


def _well_curved(name: str, draws: NDArray[np.float64]) -> bool:
    """Whether the Gaussian approximation is usable for one hyperparameter.

    A mode at the boundary (a spread near zero) has little curvature, and the Laplace draws
    then spread far wider than the prior; that is a sign to run NUTS instead.
    """
    spread = float(np.std(draws))
    limit = LAPLACE_SPREAD_LIMIT * HALF_NORMAL_SCALES.get(name, 1.0)
    return 0.0 < spread < limit


def _flags(observations: Sequence[SessionObs]) -> NDArray[np.float64]:
    events = [(o.season, o.round) for o in observations]
    flags = np.zeros((len(events), 2))
    for i in range(1, len(events)):
        flags[i] = (events[i] != events[i - 1], events[i][0] != events[i - 1][0])
    return flags


def _variances(panel: Panel, hp: dict[str, Any]) -> dict[str, Any]:
    """Per-effect prior, per-race and per-season variances, as in ``PaceFilter``."""
    kind = jnp.asarray(panel.kind)
    is_ = {k: (kind == i).astype(float) for i, k in enumerate(KINDS)}
    sd_of = {
        "team": hp["team_sd"],
        "sup": hp["supplier_sd"],
        "tq": hp["quali_offset_sd"],
        "tc": hp["cluster_sd"],
        "dq": hp["driver_quali_sd"],
        "dr": hp["driver_race_sd"],
    }
    season_of = {
        "team": hp["team_season_step"],
        "sup": hp["supplier_season_step"],
        "tq": hp["quali_offset_sd"],
        "tc": hp["cluster_sd"],
        "dq": hp["driver_season_step"],
        "dr": hp["driver_season_step"],
    }
    supplier_step = hp.get("supplier_step", 0.0)
    return {
        "prior": sum(is_[k] * sd**2 for k, sd in sd_of.items()),
        "event": is_["team"] * hp["team_step"] ** 2 + is_["sup"] * supplier_step**2,
        "season": sum(is_[k] * sd**2 for k, sd in season_of.items()),
        "prior_cross": hp["driver_corr"] * hp["driver_quali_sd"] * hp["driver_race_sd"],
        "season_cross": hp["driver_corr"] * hp["driver_season_step"] ** 2,
    }
