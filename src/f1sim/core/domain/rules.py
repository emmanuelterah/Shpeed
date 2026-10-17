"""Season rules (P4-06): the regulations that change between seasons.

A rules change, such as the 2026 reset, is a data change rather than a code change: the dry
compounds per event, the minimum race distance, the sprint format and the overtake aid live
in ``configs/rules/<season>.yaml`` and are validated here. The loader is in
``f1sim.adapters.config_files`` because the core layer does no file I/O.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from f1sim.core.errors import ValidationError

OVERTAKE_AIDS = ("drs", "overtake_mode")
"""Ways a car can be helped past another, from 2018 (DRS) to the 2026 Overtake Mode."""

SPRINT_FORMATS = ("sprint", "sprint_qualifying", "sprint_shootout")
"""Sprint weekend formats; a conventional weekend stores None."""

SEASON_RANGE = (1950, 2100)
RACE_DISTANCE_KM_RANGE = (200.0, 700.0)
DRY_COMPOUNDS_RANGE = (1, 5)
MGUK_KW_RANGE = (0.0, 800.0)


@dataclass(frozen=True)
class PowerUnit:
    """The power unit's energy recovery (P4-06).

    Attributes:
        mguh: Whether the MGU-H is fitted; it was removed for 2026.
        mguk_kw: MGU-K deployment in kilowatts.
    """

    mguh: bool = False
    mguk_kw: float = 0.0

    def __post_init__(self) -> None:
        """Check the deployment figure is plausible.

        Raises:
            ValidationError: If ``mguk_kw`` is outside ``MGUK_KW_RANGE``.
        """
        low, high = MGUK_KW_RANGE
        if not low <= self.mguk_kw <= high:
            raise ValidationError(f"power_unit.mguk_kw {self.mguk_kw} not in [{low}, {high}]")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], label: str) -> "PowerUnit":
        """Build a power unit from a parsed ``power_unit`` mapping.

        Args:
            data: The ``power_unit`` mapping.
            label: Source label used in error messages.

        Returns:
            The power unit.

        Raises:
            ValidationError: If a field is missing or the wrong type.
        """
        return cls(
            mguh=_flag(data.get("mguh", False), label, "power_unit.mguh"),
            mguk_kw=_number(data.get("mguk_kw", 0.0), label, "power_unit.mguk_kw"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the ``power_unit`` mapping.

        Returns:
            The mapping written into a rules file.
        """
        return {"mguh": self.mguh, "mguk_kw": self.mguk_kw}


@dataclass(frozen=True)
class SeasonRules:
    """The regulations for one season (P4-06).

    Attributes:
        season: Championship year.
        race_distance_km: Minimum race distance in kilometres.
        dry_compounds_per_event: Number of dry compounds nominated per event.
        mandatory_compound_change: Whether a dry race needs two different dry compounds.
        overtake_aid: ``drs``, ``overtake_mode`` or None.
        active_aero: Whether moveable aero is allowed; 2026 onwards.
        sprint_format: Sprint weekend format, or None for a conventional weekend.
        power_unit: The season's power-unit rules.
        notes: Free-text provenance, such as the source of the figures.
    """

    season: int
    race_distance_km: float
    dry_compounds_per_event: int
    mandatory_compound_change: bool
    overtake_aid: str | None
    active_aero: bool
    sprint_format: str | None
    power_unit: PowerUnit
    notes: str = ""

    def __post_init__(self) -> None:
        """Check every field is plausible and in range.

        Raises:
            ValidationError: If a range is violated or a choice is not allowed.
        """
        low, high = SEASON_RANGE
        if not low <= self.season <= high:
            raise ValidationError(f"season {self.season} not in [{low}, {high}]")
        low_km, high_km = RACE_DISTANCE_KM_RANGE
        if not low_km <= self.race_distance_km <= high_km:
            raise ValidationError(
                f"race_distance_km {self.race_distance_km} not in [{low_km}, {high_km}]"
            )
        low_c, high_c = DRY_COMPOUNDS_RANGE
        if not low_c <= self.dry_compounds_per_event <= high_c:
            raise ValidationError(
                f"dry_compounds_per_event {self.dry_compounds_per_event} not in [{low_c}, {high_c}]"
            )
        _check_choice(self.overtake_aid, OVERTAKE_AIDS, "overtake_aid")
        _check_choice(self.sprint_format, SPRINT_FORMATS, "sprint_format")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], label: str = "rules") -> "SeasonRules":
        """Build the rules from a parsed top-level YAML mapping.

        Args:
            data: The top-level rules mapping.
            label: Source label used in error messages.

        Returns:
            The validated rules.

        Raises:
            ValidationError: If a required field is missing or the wrong type.
        """
        raw_power = data.get("power_unit", {})
        if not isinstance(raw_power, Mapping):
            raise ValidationError(f"{label}: power_unit must be a mapping")
        power = cast(Mapping[str, Any], raw_power)
        return cls(
            season=_integer(data.get("season"), label, "season"),
            race_distance_km=_number(data.get("race_distance_km"), label, "race_distance_km"),
            dry_compounds_per_event=_integer(
                data.get("dry_compounds_per_event"), label, "dry_compounds_per_event"
            ),
            mandatory_compound_change=_flag(
                data.get("mandatory_compound_change", True), label, "mandatory_compound_change"
            ),
            overtake_aid=_optional_text(data.get("overtake_aid"), label, "overtake_aid"),
            active_aero=_flag(data.get("active_aero", False), label, "active_aero"),
            sprint_format=_optional_text(data.get("sprint_format"), label, "sprint_format"),
            power_unit=PowerUnit.from_mapping(power, label),
            notes=_text(data.get("notes", ""), label, "notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the rules-file mapping.

        Returns:
            A mapping with the keys of ``configs/rules/<season>.yaml``.
        """
        return {
            "season": self.season,
            "race_distance_km": self.race_distance_km,
            "dry_compounds_per_event": self.dry_compounds_per_event,
            "mandatory_compound_change": self.mandatory_compound_change,
            "overtake_aid": self.overtake_aid,
            "active_aero": self.active_aero,
            "sprint_format": self.sprint_format,
            "power_unit": self.power_unit.to_dict(),
            "notes": self.notes,
        }


def _check_choice(value: str | None, allowed: tuple[str, ...], field: str) -> None:
    if value is not None and value not in allowed:
        raise ValidationError(f"{field} {value!r} is not one of {', '.join(allowed)} or null")


def _number(value: Any, label: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label}: {field} must be a number")
    return float(value)


def _integer(value: Any, label: str, field: str) -> int:
    number = _number(value, label, field)
    if number != int(number):
        raise ValidationError(f"{label}: {field} must be a whole number")
    return int(number)


def _flag(value: Any, label: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{label}: {field} must be true or false")
    return value


def _text(value: Any, label: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{label}: {field} must be text")
    return value


def _optional_text(value: Any, label: str, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, label, field) or None
