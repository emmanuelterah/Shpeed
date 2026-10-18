"""Load reference data: team lineage, circuit lengths and clusters, engine suppliers.

This lives in the adapters layer because the core does no file I/O. Defaults ship inside
the package (``f1sim/resources``), so loading works the same when installed from PyPI.
"""

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import yaml

from f1sim.adapters.models.pace_filter import PaceHyper
from f1sim.adapters.models.tyres import TyreModel
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.natural_experiment import LEVELS, NaturalExperiment
from f1sim.core.domain.reference import CircuitTable, EngineSuppliers
from f1sim.core.domain.rules import SeasonRules
from f1sim.core.errors import DataError, ValidationError

TEAM_LINEAGE_RESOURCE = "team_lineage.yaml"
CIRCUITS_RESOURCE = "circuits.yaml"
ENGINE_SUPPLIERS_RESOURCE = "engine_suppliers.yaml"


def load_team_lineage(path: Path | None = None) -> TeamLineage:
    """Load the team lineage table.

    Args:
        path: A YAML file to load instead of the packaged default.

    Returns:
        The lineage index.

    Raises:
        DataError: If the file cannot be read or has no ``lineages`` mapping.
    """
    data, label = _load(TEAM_LINEAGE_RESOURCE, path)
    lineages = _mapping(data.get("lineages"), label, "lineages")
    typed = cast(dict[str, list[Any]], lineages)
    return TeamLineage({str(k): [str(t) for t in v] for k, v in typed.items()})


def load_circuits(path: Path | None = None) -> CircuitTable:
    """Load circuit lap lengths and the speed bands of the clusters.

    Args:
        path: A YAML file to load instead of the packaged default.

    Returns:
        The circuit table.

    Raises:
        DataError: If the file cannot be read or lacks ``circuits`` or ``clusters``.
    """
    data, label = _load(CIRCUITS_RESOURCE, path)
    circuits = _mapping(data.get("circuits"), label, "circuits")
    bands = _mapping(data.get("clusters"), label, "clusters")
    try:
        return CircuitTable(
            lengths_km={str(k): float(v) for k, v in circuits.items()},
            low_below=float(bands["low_below"]),
            high_from=float(bands["high_from"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"{label}: bad circuit length or cluster band: {exc}") from exc


def load_engine_suppliers(path: Path | None = None) -> EngineSuppliers:
    """Load the power-unit supplier of each team.

    Args:
        path: A YAML file to load instead of the packaged default.

    Returns:
        The supplier table.

    Raises:
        DataError: If the file cannot be read or lacks a ``defaults`` mapping.
    """
    data, label = _load(ENGINE_SUPPLIERS_RESOURCE, path)
    defaults = _mapping(data.get("defaults"), label, "defaults")
    seasons = _mapping(data.get("seasons", {}), label, "seasons")
    return EngineSuppliers(
        defaults={str(k): str(v) for k, v in defaults.items()},
        seasons={
            int(season): {str(k): str(v) for k, v in _mapping(teams, label, str(season)).items()}
            for season, teams in seasons.items()
        },
    )


def load_season_rules(season: int, directory: Path | None = None) -> SeasonRules:
    """Load one season's rules from ``configs/rules/<season>.yaml`` (P4-06).

    Args:
        season: Championship year.
        directory: Folder holding the rules files; defaults to ``configs/rules``.

    Returns:
        The validated rules.

    Raises:
        DataError: If the file is missing or cannot be read.
        ValidationError: If the file does not validate, or names a different season.
    """
    folder = directory if directory is not None else Path("configs/rules")
    path = folder / f"{season}.yaml"
    if not path.exists():
        raise DataError(f"no rules file for {season}: expected {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DataError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a mapping")
    rules = SeasonRules.from_mapping(cast(dict[str, Any], data), str(path))
    if rules.season != season:
        raise ValidationError(f"{path} declares season {rules.season}, expected {season}")
    return rules


def load_natural_experiments(path: Path | None = None) -> tuple[NaturalExperiment, ...]:
    """Load the natural-experiment manifest for the regulation-reset study (P7-01).

    Args:
        path: YAML file to read; defaults to ``configs/natural-experiments.yaml``.

    Returns:
        One entry per natural experiment, in file order.

    Raises:
        DataError: If the file is missing, cannot be read, is not a mapping, lists no
            experiments, or an experiment is missing a field or names a bad level.
    """
    file = path if path is not None else Path("configs/natural-experiments.yaml")
    if not file.exists():
        raise DataError(f"no natural-experiment manifest at {file}")
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"Cannot read {file}: {exc}") from exc
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DataError(f"{file} is not valid YAML: {exc}") from exc
    top = _mapping(data, str(file), "top level")
    entries = top.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise DataError(f"{file} must list at least one experiment")
    typed = cast(list[Any], entries)
    return tuple(_experiment(entry, str(file)) for entry in typed)


def _experiment(entry: Any, label: str) -> NaturalExperiment:
    row = _mapping(entry, label, "experiment")
    try:
        seasons = tuple(int(s) for s in cast(list[Any], row["seasons"]))
        experiment = NaturalExperiment(
            experiment_id=str(row["id"]),
            name=str(row["name"]),
            level=str(row["level"]),
            seasons=seasons,
            reset_season=int(row["reset_season"]),
            reset_round=int(row.get("reset_round", 1)),
            snapshot=int(row["snapshot"]) if row.get("snapshot") is not None else None,
            notes=str(row.get("notes", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"{label}: bad experiment entry: {exc}") from exc
    if experiment.level not in LEVELS:
        raise DataError(f"{label}: {experiment.experiment_id} level must be one of {LEVELS}")
    if experiment.reset_season not in experiment.seasons:
        raise DataError(f"{label}: {experiment.experiment_id} reset season not in its seasons")
    return experiment


def load_tyre_model(path: Path) -> TyreModel:
    """Load the fitted tyre and strategy model written by ``f1sim tyres fit``.

    Args:
        path: JSON file written by ``f1sim tyres fit``.

    Returns:
        The tyre model.

    Raises:
        DataError: If the file is missing, cannot be read, is not a mapping, or does not
            contain a usable model.
    """
    if not path.exists():
        raise DataError(f"no tyre model at {path}; run `f1sim tyres fit` first")
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc
    mapping = _mapping(data, str(path), "model")
    try:
        return TyreModel.from_dict(mapping)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise DataError(f"{path} is not a usable tyre model: {exc}") from exc


def load_pace_hyper(path: Path) -> PaceHyper:
    """Load fitted pace hyperparameters, or the defaults if the file does not exist.

    Args:
        path: JSON written by ``f1sim pace fit``, with a ``hyper`` mapping.

    Returns:
        The hyperparameters.

    Raises:
        DataError: If the file exists but cannot be read or has no ``hyper`` mapping.
    """
    if not path.exists():
        return PaceHyper()
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot read {path}: {exc}") from exc
    hyper = _mapping(_mapping(data, str(path), "top level").get("hyper"), str(path), "hyper")
    return PaceHyper.from_dict(hyper)


def _load(resource_name: str, path: Path | None) -> tuple[dict[str, Any], str]:
    resource = files("f1sim.resources") / resource_name
    label = str(path) if path is not None else f"f1sim/resources/{resource_name}"
    try:
        text = path.read_text(encoding="utf-8") if path is not None else resource.read_text()
    except OSError as exc:
        raise DataError(f"Cannot read {label}: {exc}") from exc
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise DataError(f"{label} is not valid YAML: {exc}") from exc
    return _mapping(data, label, "top level"), label


def _mapping(value: Any, label: str, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataError(f"{label} must contain a '{key}' mapping")
    return {str(k): v for k, v in cast(dict[Any, Any], value).items()}
