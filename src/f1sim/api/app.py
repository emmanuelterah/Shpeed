"""Read-only HTTP API for published prediction records (P8-04).

Serves the JSON files under ``predictions/<season>/<round>.json``. It never writes, so it
runs on a read-only volume and needs no database.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from f1sim import __version__
from f1sim.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings; production uses environment variables."""
    cfg = settings or Settings()
    root = cfg.predictions_dir
    app = FastAPI(title="Shpeed API", version=__version__)

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    def records() -> list[Path]:
        return sorted(root.glob("[0-9][0-9][0-9][0-9]/*.json"), key=_season_round)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/predictions")
    def list_predictions() -> list[dict[str, int]]:
        """Every published race, oldest first."""
        return [{"season": s, "round": r} for s, r in map(_season_round, records())]

    @app.get("/predictions/latest")
    def latest_prediction() -> dict[str, Any]:
        found = records()
        if not found:
            raise HTTPException(status_code=404, detail="No predictions published yet")
        return _load(found[-1])

    @app.get("/predictions/{season}/{round_number}")
    def get_prediction(season: int, round_number: int) -> dict[str, Any]:
        for name in (f"{round_number:02d}.json", f"{round_number}.json"):
            path = root / str(season) / name
            if path.is_file():
                return _load(path)
        raise HTTPException(
            status_code=404, detail=f"No prediction for {season} round {round_number}"
        )

    return app


def _season_round(path: Path) -> tuple[int, int]:
    return int(path.parent.name), int(path.stem)


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


app = create_app()
