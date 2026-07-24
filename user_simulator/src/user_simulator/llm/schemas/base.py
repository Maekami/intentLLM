from pydantic import BaseModel, ConfigDict


class StrictWireModel(BaseModel):
    """Base for externally generated values; coercion and extra fields are forbidden."""

    model_config = ConfigDict(extra="forbid", strict=True)
