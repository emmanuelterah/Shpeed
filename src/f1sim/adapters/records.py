"""Validate published JSON against the shared JSON Schemas (P8-02, P8-13).

The schemas in ``schemas/`` are the single source of truth for what is published: the backend
validates every prediction record and the predictions index with them before writing, and the
frontend generates its TypeScript types from the same files. The schemas ship inside the
wheel so validation also works after ``pip install f1sim``.
"""

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaError

from f1sim.core.errors import DataError, ValidationError

RECORD_SCHEMA = "prediction-record.v1.schema.json"
INDEX_SCHEMA = "predictions-index.v1.schema.json"
SCHEMAS = (RECORD_SCHEMA, INDEX_SCHEMA)


def schema_path(name: str = RECORD_SCHEMA) -> Path:
    """Locate a shared schema on disk.

    Looks for the repository's ``schemas/`` folder first (so edits are picked up
    immediately), then the copy packaged inside the wheel.

    Args:
        name: Schema file name, one of ``SCHEMAS``.

    Returns:
        Path to the schema file.

    Raises:
        DataError: If neither copy exists.
    """
    repo = Path(__file__).resolve().parents[3] / "schemas" / name
    if repo.is_file():
        return repo
    packaged = files("f1sim.resources").joinpath(name)
    if packaged.is_file():
        return Path(str(packaged))
    raise DataError(f"schema {name} not found (looked for {repo})")


@lru_cache(maxsize=len(SCHEMAS))
def _validator(name: str) -> Draft202012Validator:
    try:
        schema: Any = json.loads(schema_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read the schema {name}: {exc}") from exc
    return Draft202012Validator(schema)


def validate_against(name: str, document: Mapping[str, Any]) -> None:
    """Check a JSON document against one of the shared schemas.

    Args:
        name: Schema file name, one of ``SCHEMAS``.
        document: The JSON-ready document.

    Raises:
        ValidationError: If the document does not validate, naming the first problem.
    """
    errors = sorted(_errors(name, document), key=lambda e: [str(part) for part in e.path])
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise ValidationError(f"{name} is not satisfied at {location}: {first.message}")


def _errors(name: str, document: Mapping[str, Any]) -> list[JsonSchemaError]:
    # The library's iter_errors overload is only partly typed; its items are always
    # jsonschema ValidationErrors, so the result is typed here once.
    checker: Any = _validator(name)
    return list(checker.iter_errors(dict(document)))


def validate_record(record: Mapping[str, Any]) -> None:
    """Check a prediction record against its schema.

    Args:
        record: A JSON-ready prediction record.

    Raises:
        ValidationError: If the record does not validate.
    """
    validate_against(RECORD_SCHEMA, record)


def validate_index(index: Mapping[str, Any]) -> None:
    """Check the predictions index against its schema.

    Args:
        index: The JSON-ready index.

    Raises:
        ValidationError: If the index does not validate.
    """
    validate_against(INDEX_SCHEMA, index)


class JsonSchemaValidator:
    """A :class:`~f1sim.core.ports.publisher.RecordValidator` backed by the JSON Schema."""

    def validate(self, record: Mapping[str, Any]) -> None:
        """Check a record against the shared schema.

        Args:
            record: A JSON-ready prediction record.

        Raises:
            ValidationError: If the record does not validate.
        """
        validate_record(record)
