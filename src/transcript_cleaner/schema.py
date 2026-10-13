"""Tiny dependency-free validator for the JSON-Schema subset used by schemas/cleaned_transcript.schema.json."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).parent / "schemas" / "cleaned_transcript.schema.json"

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_against_schema(instance: Any, schema: dict | None = None, path: str = "$", limit: int = 20) -> list[str]:
    """Return a list of human-readable violations (empty = valid). Stops collecting after `limit`."""
    schema = schema or load_schema()
    errors: list[str] = []
    _walk(instance, schema, path, errors, limit)
    return errors


def _walk(value: Any, schema: dict, path: str, errors: list[str], limit: int) -> None:
    if len(errors) >= limit:
        return
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        if not any(_TYPES[t](value) for t in types):
            errors.append(f"{path}: expected {'|'.join(types)}, got {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "pattern" in schema and isinstance(value, str) and not re.search(schema["pattern"], value):
        errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _walk(value[key], sub, f"{path}.{key}", errors, limit)
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _walk(item, schema["items"], f"{path}[{i}]", errors, limit)
            if len(errors) >= limit:
                return
