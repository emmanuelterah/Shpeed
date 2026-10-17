"""Serve recorded Jolpica-F1 responses from tests/fixtures/recorded/jolpica.

Each fixture holds a complete response; the transport applies ``limit`` and ``offset``
itself, exactly as the API paginates, so adapters' paging logic is exercised for real.
"""

import copy
import json
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).parent / "recorded" / "jolpica"
NESTED = ("Results", "QualifyingResults", "SprintResults", "PitStops")


def _page(data: dict[str, Any], limit: int, offset: int) -> dict[str, Any]:
    out = copy.deepcopy(data)
    mr = out["MRData"]
    if "RaceTable" in mr:
        total = _page_races(mr["RaceTable"], limit, offset)
    elif "StandingsTable" in mr:
        total = _page_standings(mr["StandingsTable"], limit, offset)
    else:
        total = 0
    mr.update(limit=str(limit), offset=str(offset), total=str(total))
    return out


def _page_races(table: dict[str, Any], limit: int, offset: int) -> int:
    races: list[dict[str, Any]] = table["Races"]
    key = next((k for k in NESTED if races and k in races[0]), None)
    if key is None:
        table["Races"] = races[offset : offset + limit]
        return len(races)
    items = [(i, item) for i, race in enumerate(races) for item in race[key]]
    window = items[offset : offset + limit]
    kept: dict[int, dict[str, Any]] = {}
    for i, item in window:
        kept.setdefault(i, {**races[i], key: []})[key].append(item)
    table["Races"] = list(kept.values())
    return len(items)


def _page_standings(table: dict[str, Any], limit: int, offset: int) -> int:
    lists: list[dict[str, Any]] = table.get("StandingsLists", [])
    if not lists:
        return 0
    entry = lists[0]
    key = "DriverStandings" if "DriverStandings" in entry else "ConstructorStandings"
    items: list[dict[str, Any]] = entry[key]
    entry[key] = items[offset : offset + limit]
    return len(items)


def jolpica_transport(requests: list[str] | None = None) -> httpx.MockTransport:
    """An httpx transport that serves recorded Jolpica responses, paginated like the API.

    Args:
        requests: If given, each request is appended as ``"<path>@<offset>"``.

    Returns:
        A transport for ``HttpClient``; unknown paths return 404.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/ergast/f1/", 1)[1].removesuffix(".json")
        limit = int(request.url.params.get("limit", "30"))
        offset = int(request.url.params.get("offset", "0"))
        if requests is not None:
            requests.append(f"{path}@{offset}")
        file = FIXTURES / f"{path.replace('/', '_')}.json"
        if not file.is_file():
            return httpx.Response(404, json={"detail": f"no fixture for {path}"})
        data = json.loads(file.read_text(encoding="utf-8"))
        return httpx.Response(200, json=_page(data, limit, offset))

    return httpx.MockTransport(handler)
