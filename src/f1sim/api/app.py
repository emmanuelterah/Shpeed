"""Read-only HTTP API for published prediction records (P8-04).

Serves the JSON files under ``predictions/<season>/<round>.json``. It never writes, so it
runs on a read-only volume and needs no database. Requires the ``api`` extra:
``pip install "f1sim[api]"``.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from f1sim import __version__
from f1sim.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Settings to use; None reads them from the environment.

    Returns:
        A FastAPI app with ``/health`` and the ``/predictions`` routes.
    """
    cfg = settings or Settings()
    root = cfg.predictions_dir
    api = FastAPI(title="Shpeed API", version=__version__)

    if cfg.cors_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    def _records() -> list[Path]:
        return sorted(root.glob("[0-9][0-9][0-9][0-9]/*.json"), key=_season_round)

    @api.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is up.

        Returns:
            ``{"status": "ok", "version": ...}``.
        """
        return {"status": "ok", "version": __version__}

    @api.get("/predictions")
    def list_predictions() -> list[dict[str, int]]:
        """List every published race.

        Returns:
            ``{"season": ..., "round": ...}`` for each record, oldest first.
        """
        return [{"season": s, "round": r} for s, r in map(_season_round, _records())]

    @api.get("/predictions/latest")
    def latest_prediction() -> dict[str, Any]:
        """Return the most recent prediction record.

        Returns:
            The record for the latest season and round.

        Raises:
            HTTPException: 404 if nothing is published yet.
        """
        found = _records()
        if not found:
            raise HTTPException(status_code=404, detail="No predictions published yet")
        return _load(found[-1])

    @api.get("/predictions/{season}/{round_number}")
    def get_prediction(season: int, round_number: int) -> dict[str, Any]:
        """Return one race's prediction record.

        Args:
            season: Season of the race.
            round_number: Round of the race.

        Returns:
            The record.

        Raises:
            HTTPException: 404 if that race has no published record.
        """
        for name in (f"{round_number:02d}.json", f"{round_number}.json"):
            path = root / str(season) / name
            if path.is_file():
                return _load(path)
        raise HTTPException(
            status_code=404, detail=f"No prediction for {season} round {round_number}"
        )

    return api


def _season_round(path: Path) -> tuple[int, int]:
    return int(path.parent.name), int(path.stem)


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


app = create_app()
