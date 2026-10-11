"""Load reference data such as the team lineage table.

This lives in the adapters layer because the core does no file I/O. Defaults ship inside
the package (``f1sim/resources``), so loading works the same when installed from PyPI.
"""

from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import yaml

from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.errors import DataError

TEAM_LINEAGE_RESOURCE = "team_lineage.yaml"


def load_team_lineage(path: Path | None = None) -> TeamLineage:
    """Load the team lineage table.

    Args:
        path: A YAML file to load instead of the packaged default.

    Returns:
        The lineage index.

    Raises:
        DataError: If the file cannot be read or has no ``lineages`` mapping.
    """
    resource = files("f1sim.resources") / TEAM_LINEAGE_RESOURCE
    label = str(path) if path is not None else f"f1sim/resources/{TEAM_LINEAGE_RESOURCE}"
    try:
        text = path.read_text(encoding="utf-8") if path is not None else resource.read_text()
    except OSError as exc:
        raise DataError(f"Cannot read team lineage file {label}: {exc}") from exc
    data: Any = yaml.safe_load(text)
    lineages: Any = cast(dict[str, Any], data).get("lineages") if isinstance(data, dict) else None
    if not isinstance(lineages, dict):
        raise DataError(f"{label} must contain a 'lineages' mapping")
    typed = cast(dict[str, list[Any]], lineages)
    return TeamLineage({str(k): [str(t) for t in v] for k, v in typed.items()})
