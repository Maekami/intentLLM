import hashlib
import json
from typing import Any

from pydantic import BaseModel


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a detached schema with closed object definitions at every level."""

    schema = model.model_json_schema()
    return _close_objects(schema)


def _close_objects(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _close_objects(item) for key, item in value.items()}
        if result.get("type") == "object":
            result.setdefault("additionalProperties", False)
        return result
    if isinstance(value, list):
        return [_close_objects(item) for item in value]
    return value


def canonical_schema_json(model: type[BaseModel]) -> str:
    return json.dumps(
        strict_json_schema(model),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def schema_hash(model: type[BaseModel]) -> str:
    return hashlib.sha256(canonical_schema_json(model).encode()).hexdigest()
