"""Structured JSON logging with a run ID on every line (P0-14)."""

import logging
import uuid

import structlog


def configure_logging(level: str = "INFO") -> str:
    """Configure structlog for JSON output and bind a fresh run ID to every log line.

    Args:
        level: Minimum log level name, such as ``"INFO"`` or ``"DEBUG"`` (case-insensitive).

    Returns:
        The 12-character run ID bound to this process's logs.

    Raises:
        KeyError: If ``level`` is not a standard logging level name.
    """
    run_id = uuid.uuid4().hex[:12]
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
    )
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return run_id
