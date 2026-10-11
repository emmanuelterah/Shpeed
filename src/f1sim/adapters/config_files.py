"""Load reference data from configs/ (kept out of the core, which does no file I/O)."""

from pathlib import Path
from typing import Any, cast

import yaml

from f1sim.core.domain.lineage import TeamLineage
from f1sim.core.errors import DataError

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def load_team_lineage(path: Path | None = None) -> TeamLineage:
    file = path or CONFIGS / "mappings" / "team_lineage.yaml"
    try:
        data: Any = yaml.safe_load(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataError(f"Cannot read team lineage file {file}: {exc}") from exc
    lineages: Any = cast(dict[str, Any], data).get("lineages") if isinstance(data, dict) else None
    if not isinstance(lineages, dict):
        raise DataError(f"{file} must contain a 'lineages' mapping")
    typed = cast(dict[str, list[Any]], lineages)
    return TeamLineage({str(k): [str(t) for t in v] for k, v in typed.items()})
