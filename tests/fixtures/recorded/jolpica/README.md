# Jolpica-F1 fixtures

Files named `<path with / as _>.json`, each a complete response, in the exact Jolpica-F1 (Ergast) response format.
They are hand-written from the public format, with values close to the real 2024 Bahrain and Miami
weekends, because the build workspace could not reach the API. Replace them with real recordings:

    uv run python scripts/record_fixtures.py 2024 2024/1/results 2024/1/qualifying 2024/1/pitstops 2024/6/sprint

Tests serve these files through `httpx.MockTransport`; CI never touches the network.
