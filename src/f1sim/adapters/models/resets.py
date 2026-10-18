"""Prior-handling strategies for a regulation reset (P7-02).

A regulation change makes the era before it a poor guide to the era after, so Phase 7 asks
which way of treating that history recovers fastest. Each strategy is a pure transform of the
pace filter's Gaussian posterior at a weekend boundary, chosen by name from config, so the
filter, the engine and the backtester do not change (open/closed):

- ``full``: keep every past race, the Phase 3 baseline.
- ``decay``: fade the information a little every weekend, so a race a half-life back counts half.
- ``reset``: at the reset season, drop the car effects and re-draw them from a wide prior.
- ``testing``: like ``reset``, but the reset season's opening practice carries more weight,
  standing in for the pre-season testing the model does not otherwise see.

Only the car effects (team, supplier, team x circuit cluster and team qualifying offset) are
reset: the reset replaces the cars, not the drivers, whose effects carry over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

from f1sim.core.errors import DataError

if TYPE_CHECKING:
    from f1sim.adapters.models.pace_filter import PaceState

CAR_PREFIXES = ("team:", "sup:", "tq:", "tc:")
"""Label prefixes of the car effects a reset replaces; driver effects are kept."""

STRATEGIES: tuple[str, ...] = ("full", "decay", "reset", "testing")
"""Named strategies, in the order the study reports them."""

LAP_LEVEL_ONLY = "reset and testing strategies need a reset season"
DEFAULT_HALF_LIFE = 10.0
"""Races after which kept history counts half, the default for ``decay``."""
DEFAULT_RESET_SD = 1.5
"""Spread of the reset car prior, percent of lap time; the fitted team prior is 1.0."""
DEFAULT_PRACTICE_WEIGHT = 4.0
"""How much more the reset season's opening practice counts for ``testing``."""


@dataclass(frozen=True)
class ResetConfig:
    """Which strategy to use and its knobs, as read from options or the CLI (P7-02).

    Attributes:
        strategy: One of ``STRATEGIES``; ``full`` keeps the Phase 3 behaviour.
        reset_season: First season of the new era; required by ``reset`` and ``testing``.
        half_life: Races a kept race counts half after, for ``decay``.
        sd: Spread of the reset car prior, percent, for ``reset`` and ``testing``.
        practice_weight: Weight of the reset season's opening practice, for ``testing``.
    """

    strategy: str = "full"
    reset_season: int | None = None
    half_life: float = DEFAULT_HALF_LIFE
    sd: float = DEFAULT_RESET_SD
    practice_weight: float = DEFAULT_PRACTICE_WEIGHT


@dataclass(frozen=True)
class FullHistory:
    """Keep every past race; the Phase 3 baseline (P7-02)."""

    name: ClassVar[str] = "full"

    def at_weekend(self, state: PaceState, *, season: int, new_season: bool) -> None:
        """Do nothing: the filter's own random walk already ages the effects.

        Args:
            state: The posterior, unused.
            season: Season being advanced to, unused.
            new_season: Whether this weekend begins a season, unused.
        """
        del state, season, new_season

    def noise_scale(self, group: str, season: int) -> float:
        """Leave every session's observation noise unchanged.

        Args:
            group: Noise group, such as ``FP``.
            season: Season of the session.

        Returns:
            Always 1.0.
        """
        del group, season
        return 1.0


@dataclass(frozen=True)
class ExponentialDecay:
    """Fade the kept history a little every weekend (P7-02).

    Attributes:
        half_life: Races after which a race's information counts half.
    """

    half_life: float = DEFAULT_HALF_LIFE
    name: ClassVar[str] = "decay"

    def __post_init__(self) -> None:
        """Reject a non-positive half-life, which would define no decay.

        Raises:
            DataError: If the half-life is not positive.
        """
        if self.half_life <= 0.0:
            raise DataError("decay needs a positive half-life")

    @property
    def factor(self) -> float:
        """Covariance inflation per weekend that halves weight every ``half_life`` races.

        Returns:
            ``2 ** (1 / half_life)``.
        """
        return 2.0 ** (1.0 / self.half_life)

    def at_weekend(self, state: PaceState, *, season: int, new_season: bool) -> None:
        """Inflate the whole posterior, so older sessions inform it less.

        Args:
            state: The posterior, scaled in place.
            season: Season being advanced to, unused.
            new_season: Whether this weekend begins a season, unused.
        """
        del season, new_season
        state.cov *= self.factor

    def noise_scale(self, group: str, season: int) -> float:
        """Leave every session's observation noise unchanged.

        Args:
            group: Noise group, such as ``FP``.
            season: Season of the session.

        Returns:
            Always 1.0.
        """
        del group, season
        return 1.0


@dataclass(frozen=True)
class ResetPriors:
    """Drop the old era's cars and re-draw them from a wide prior (P7-02).

    Attributes:
        reset_season: First season of the new era.
        sd: Spread of the reset car prior, percent of lap time.
    """

    reset_season: int
    sd: float = DEFAULT_RESET_SD
    name: ClassVar[str] = "reset"

    def __post_init__(self) -> None:
        """Reject a non-positive prior spread.

        Raises:
            DataError: If the spread is not positive.
        """
        if self.sd <= 0.0:
            raise DataError("a reset prior needs a positive spread")

    def at_weekend(self, state: PaceState, *, season: int, new_season: bool) -> None:
        """At the reset season, replace the car effects with prior draws.

        Args:
            state: The posterior, rewritten in place.
            season: Season being advanced to.
            new_season: Whether this weekend begins a season.
        """
        if new_season and season == self.reset_season:
            reset_car_effects(state, self.sd)

    def noise_scale(self, group: str, season: int) -> float:
        """Leave every session's observation noise unchanged.

        Args:
            group: Noise group, such as ``FP``.
            season: Season of the session.

        Returns:
            Always 1.0.
        """
        del group, season
        return 1.0


@dataclass(frozen=True)
class TestingInformedPriors(ResetPriors):
    """A ``reset`` whose new era is guided by its opening practice (P7-02).

    The pre-season testing that would set the new era's priors is not in the store, so its
    opening race weekend's practice stands in for it: those long runs count ``practice_weight``
    times more, narrowing the reset prior toward what the new cars actually did.

    Attributes:
        practice_weight: How much more the reset season's opening practice counts.
    """

    practice_weight: float = DEFAULT_PRACTICE_WEIGHT
    name: ClassVar[str] = "testing"

    def __post_init__(self) -> None:
        """Reject a non-positive practice weight.

        Raises:
            DataError: If the weight is not positive.
        """
        super().__post_init__()
        if self.practice_weight <= 0.0:
            raise DataError("testing needs a positive practice weight")

    def noise_scale(self, group: str, season: int) -> float:
        """Shrink practice noise in the new era, so its long runs inform the prior more.

        Args:
            group: Noise group, such as ``FP``.
            season: Season of the session.

        Returns:
            ``1 / sqrt(practice_weight)`` for the reset season's practice, else 1.0.
        """
        if group == "FP" and season >= self.reset_season:
            return 1.0 / math.sqrt(self.practice_weight)
        return 1.0


type ResetStrategy = FullHistory | ExponentialDecay | ResetPriors | TestingInformedPriors
"""The strategies the pace filter accepts."""


def build_reset(config: ResetConfig | None) -> ResetStrategy:
    """Build the strategy a config names, defaulting to full history.

    Args:
        config: The strategy and its knobs; ``None`` is full history.

    Returns:
        The strategy.

    Raises:
        DataError: If the strategy is unknown, or a reset strategy has no reset season.
    """
    if config is None or config.strategy == "full":
        return FullHistory()
    if config.strategy == "decay":
        return ExponentialDecay(config.half_life)
    if config.strategy not in STRATEGIES:
        raise DataError(f"Unknown reset strategy {config.strategy!r}; choose from {STRATEGIES}")
    if config.reset_season is None:
        raise DataError(f"{config.strategy} needs a reset season: {LAP_LEVEL_ONLY}")
    if config.strategy == "reset":
        return ResetPriors(config.reset_season, config.sd)
    return TestingInformedPriors(config.reset_season, config.sd, config.practice_weight)


def reset_signature(strategy: ResetStrategy) -> dict[str, object]:
    """Fingerprint a strategy's name and knobs, so saved states are not reused across them.

    Args:
        strategy: The strategy to fingerprint.

    Returns:
        A JSON-safe mapping of the strategy's name and dataclass fields.
    """
    return {
        "name": strategy.name,
        **{f.name: getattr(strategy, f.name) for f in fields(strategy)},
    }


def reset_car_effects(state: PaceState, sd: float) -> None:
    """Replace the car effects' posterior with a wide, uncorrelated prior.

    The mean is set to zero and the covariance entries touching a car effect are cleared,
    except its own variance, which becomes ``sd ** 2``. Driver effects are left alone.

    Args:
        state: The posterior, rewritten in place.
        sd: Spread of the reset car prior, percent of lap time.
    """
    variance = sd**2
    for i, label in enumerate(state.labels):
        if label.startswith(CAR_PREFIXES):
            state.mean[i] = 0.0
            state.cov[i, :] = 0.0
            state.cov[:, i] = 0.0
            state.cov[i, i] = variance
