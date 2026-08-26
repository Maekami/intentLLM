from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    # vLLM/SGLang expose Qwen thinking through reasoning_content. OpenRouter
    # may expose an equivalent reasoning_details structure. Both are retained
    # internally so Qwen's preserve_thinking chat-template option can work.
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
