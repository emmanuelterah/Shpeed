# syntax=docker/dockerfile:1
# Shpeed API image (P8-04, P8-05). Build context is the repository root.
#   runtime (default): slim, non-root, read-only API
#   dev:               editable install for `uvicorn --reload` with ./src mounted

FROM python:3.12-slim AS build
# uv must be new enough to read uv.lock (revision 3, written by uv 0.11+); 0.5.x cannot
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
# UV_HTTP_TIMEOUT: large wheels (jax, numpy, scipy) can take minutes on a slow connection
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300
WORKDIR /app
# Dependencies first, so code changes don't invalidate this layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra api --no-install-project
COPY src ./src
# The wheel bundles the JSON Schemas (pyproject: tool.hatch...force-include)
COPY schemas ./schemas
RUN uv sync --frozen --no-dev --extra api --no-editable

FROM build AS dev
RUN uv sync --frozen --extra api
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    F1SIM_PREDICTIONS_DIR=/app/predictions
EXPOSE 8000
CMD ["uvicorn", "f1sim.api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "/app/src"]

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 shpeed
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY predictions ./predictions
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    F1SIM_PREDICTIONS_DIR=/app/predictions
USER shpeed
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-m", "f1sim.api.healthcheck"]
# Forwarded headers are trusted only from the addresses in FORWARDED_ALLOW_IPS (uvicorn's
# default is 127.0.0.1). Never "*": the API port can be reached without the proxy, and any
# client could then spoof X-Forwarded-For and X-Forwarded-Proto.
CMD ["uvicorn", "f1sim.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
