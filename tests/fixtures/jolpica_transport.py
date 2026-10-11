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
    races: list[dict[str, Any]] = mr["RaceTable"]["Races"]
    key = next((k for k in NESTED if races and k in races[0]), None)
    if key is None:
        mr["RaceTable"]["Races"] = races[offset : offset + limit]
        total = len(races)
    else:
        items = [(i, item) for i, race in enumerate(races) for item in race[key]]
        total = len(items)
        window = items[offset : offset + limit]
        kept: dict[int, dict[str, Any]] = {}
        for i, item in window:
            kept.setdefault(i, {**races[i], key: []})[key].append(item)
        mr["RaceTable"]["Races"] = list(kept.values())
    mr.update(limit=str(limit), offset=str(offset), total=str(total))
    return out


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
