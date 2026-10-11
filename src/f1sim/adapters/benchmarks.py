"""Benchmark registry and regression gate (P2-13, P2-14).

Each model has ``benchmarks/<model>.json``: every recorded backtest run, with the git
commit, snapshot, seasons and summary scores, so any number in the README can be traced
to the code and data that produced it. The gate fails a change that makes a model's RPS
worse than its best recorded run on the same seasons.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from f1sim.core.errors import DataError
from f1sim.core.usecases.backtest import BacktestResult

GATE_METRIC = "rps"
GATE_TOLERANCE = 0.002
"""RPS slack before the gate fails; Monte Carlo noise alone moves RPS by about 0.001."""

UNKNOWN_COMMIT = "unknown"


@dataclass(frozen=True)
class GateResult:
    """Outcome of the regression gate.

    Attributes:
        passed: Whether the new run is within tolerance of the best recorded one.
        current: The new run's RPS.
        best: Best recorded RPS on the same seasons, or None if there is no record.
        message: One line for the terminal or CI log.
    """

    passed: bool
    current: float
    best: float | None
    message: str


def current_commit(repo: Path = Path()) -> str:
    """Find the git commit of the working tree.

    Args:
        repo: Folder inside the repository.

    Returns:
        The full commit hash, with ``-dirty`` appended when there are uncommitted
        changes, or ``"unknown"`` outside a git checkout.
    """
    try:
        head = _git(repo, "rev-parse", "HEAD")
        dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN_COMMIT
    return f"{head}-dirty" if dirty else head


def load_runs(folder: Path, model: str) -> list[dict[str, Any]]:
    """Read every recorded run of a model.

    Args:
        folder: The benchmarks folder.
        model: Registered model name.

    Returns:
        The runs, oldest first; empty if the model has no file yet.

    Raises:
        DataError: If the file exists but is not a valid benchmark file.
    """
    path = folder / f"{model}.json"
    if not path.exists():
        return []
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot read benchmark file {path}: {exc}") from exc
    runs: Any = cast(dict[str, Any], data).get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, list):
        raise DataError(f"{path} must contain a 'runs' list")
    return [cast(dict[str, Any], r) for r in cast(list[Any], runs) if isinstance(r, dict)]


def record_run(
    folder: Path, result: BacktestResult, *, commit: str, seed: int, now: datetime | None = None
) -> Path:
    """Append a backtest run to its model's benchmark file.

    Args:
        folder: The benchmarks folder; created if missing.
        result: The backtest to record.
        commit: Git commit that produced it.
        seed: Base seed of the backtest.
        now: Recording time; defaults to the current UTC time.

    Returns:
        The file written.
    """
    folder.mkdir(parents=True, exist_ok=True)
    runs = load_runs(folder, result.model)
    moment = now or datetime.now(UTC)
    runs.append(
        {
            "recorded_at": moment.isoformat(timespec="seconds"),
            "git_commit": commit,
            "snapshot_id": int(result.snapshot),
            "seasons": list(result.seasons),
            "seed": seed,
            "skipped": len(result.skipped),
            "summary": {k: round(v, 6) for k, v in result.summary().items()},
        }
    )
    path = folder / f"{result.model}.json"
    body = {"model": result.model, "runs": runs}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def check_regression(folder: Path, result: BacktestResult) -> GateResult:
    """Compare a backtest with the best recorded run of the same model and seasons.

    Args:
        folder: The benchmarks folder.
        result: The new backtest.

    Returns:
        Whether it passes. With no earlier run on these seasons it passes.

    Raises:
        DataError: If the backtest scored no races.
    """
    summary = result.summary()
    if GATE_METRIC not in summary:
        raise DataError(f"{result.model}: the backtest scored no races")
    current = summary[GATE_METRIC]
    seasons = list(result.seasons)
    best = min(
        (
            float(run["summary"][GATE_METRIC])
            for run in load_runs(folder, result.model)
            if run.get("seasons") == seasons and GATE_METRIC in run.get("summary", {})
        ),
        default=None,
    )
    if best is None:
        return GateResult(True, current, None, f"{result.model}: RPS {current:.4f}, no record")
    passed = current <= best + GATE_TOLERANCE
    verdict = "ok" if passed else "REGRESSION"
    message = f"{result.model}: RPS {current:.4f} vs best {best:.4f} ({verdict})"
    return GateResult(passed, current, best, message)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
    )
    return done.stdout.strip()
