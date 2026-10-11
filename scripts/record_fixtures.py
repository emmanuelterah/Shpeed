"""Record real Jolpica-F1 responses as test fixtures (P1-18).

Run on a machine with internet access, then commit the files it writes:

    uv run python scripts/record_fixtures.py 2024 2024/1/results 2024/1/qualifying \
        2024/1/pitstops 2024/6/sprint

Each argument is an API path without the ``.json`` suffix. Every page is fetched and merged
into one complete response, which the test transport paginates again on demand.
"""

import json
import sys
from pathlib import Path
from typing import Any

import httpx

BASE = "https://api.jolpi.ca/ergast/f1"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "recorded" / "jolpica"
NESTED = ("Results", "QualifyingResults", "SprintResults", "PitStops")
PAGE = 100


def record(path: str, client: httpx.Client) -> Path:
    merged: dict[str, Any] | None = None
    offset = 0
    while True:
        response = client.get(f"{BASE}/{path}.json", params={"limit": PAGE, "offset": offset})
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        if merged is None:
            merged = data
        else:
            new_races = data["MRData"]["RaceTable"]["Races"]
            races = merged["MRData"]["RaceTable"]["Races"]
            key = next((k for k in NESTED if new_races and k in new_races[0]), None)
            if key is None:
                races.extend(new_races)
            else:
                races[-1][key].extend(item for race in new_races for item in race[key])
        offset += PAGE
        if offset >= int(data["MRData"]["total"]):
            break
    assert merged is not None
    merged["MRData"].update(limit=str(PAGE), offset="0")
    out = OUT / f"{path.replace('/', '_')}.json"
    out.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, headers={"User-Agent": "shpeed-fixture-recorder"}) as client:
        for path in paths:
            print(f"recorded {record(path, client).relative_to(OUT.parents[3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
