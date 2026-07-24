from dataclasses import dataclass

from pydantic import BaseModel

from user_simulator.exceptions import StructuredOutputError
from user_simulator.llm.schema_utils import schema_hash, strict_json_schema
from user_simulator.llm.schemas import (
    ControllerResultV2,
    SatisfactionUpdateResultV2,
    UserGenerationResultV2,
)


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    version: int
    model: type[BaseModel]

    @property
    def hash(self) -> str:
        return schema_hash(self.model)

    @property
    def json_schema(self) -> dict:
        return strict_json_schema(self.model)


SCHEMA_REGISTRY: dict[tuple[str, int], SchemaSpec] = {
    ("controller_result", 2): SchemaSpec("controller_result", 2, ControllerResultV2),
    ("satisfaction_update_result", 2): SchemaSpec(
        "satisfaction_update_result", 2, SatisfactionUpdateResultV2
    ),
    ("user_generation_result", 2): SchemaSpec("user_generation_result", 2, UserGenerationResultV2),
}


def resolve_schema(name: str, version: int) -> SchemaSpec:
    try:
        return SCHEMA_REGISTRY[(name, version)]
    except KeyError as exc:
        known = ", ".join(f"{key[0]}@{key[1]}" for key in SCHEMA_REGISTRY)
        raise StructuredOutputError(
            f"unknown structured-output schema {name}@{version}; known: {known}"
        ) from exc
