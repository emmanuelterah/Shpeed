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
from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.domain.reference import CircuitTable, EngineSuppliers
from f1sim.core.errors import DataError

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
    data: Any = yaml.safe_load(text)
    return _mapping(data, label, "top level"), label


def _mapping(value: Any, label: str, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataError(f"{label} must contain a '{key}' mapping")
    return {str(k): v for k, v in cast(dict[Any, Any], value).items()}
