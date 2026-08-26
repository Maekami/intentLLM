from dataclasses import dataclass

from pydantic import BaseModel

from user_simulator.domain.results import (
    ControllerResult,
    SatisfactionUpdateResult,
    UserGenerationResult,
)
from user_simulator.exceptions import StructuredOutputError
from user_simulator.llm.schema_utils import schema_hash, strict_json_schema


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    model: type[BaseModel]

    @property
    def hash(self) -> str:
        return schema_hash(self.model)

    @property
    def json_schema(self) -> dict:
        return strict_json_schema(self.model)


SCHEMA_REGISTRY: dict[str, SchemaSpec] = {
    "controller_result": SchemaSpec(
        name="controller_result",
        model=ControllerResult,
    ),
    "satisfaction_update_result": SchemaSpec(
        name="satisfaction_update_result",
        model=SatisfactionUpdateResult,
    ),
    "user_generation_result": SchemaSpec(
        name="user_generation_result",
        model=UserGenerationResult,
    ),
}


def resolve_schema(name: str) -> SchemaSpec:
    try:
        return SCHEMA_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(SCHEMA_REGISTRY))
        raise StructuredOutputError(
            f"unknown structured-output schema {name!r}; known: {known}"
        ) from exc
