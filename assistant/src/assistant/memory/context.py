from __future__ import annotations

from typing import Any

from assistant.config import MemoryContextSettings
from assistant.memory.models import RetrievalResult

REMEM_SYSTEM_PROMPT = """You are a helpful assistant with access to LOCAL EXPERIENCE MEMORY.
Use a Think-Refine-Act loop:
- Think: reason internally about the current request.
- Think-Prune: remove irrelevant retrieved memories by their displayed IDs.
- Final Answer: provide the answer that should be shown to the user.

Use relevant memories as prior task experience and prune memories that do not help."""


def build_exprag_context(
    query: str,
    retrieved: list[RetrievalResult],
    settings: MemoryContextSettings,
) -> str:
    if not retrieved:
        return query
    experiences = "\n\n".join(
        f"[Experience #{index}]\n"
        + result.entry.to_text(
            include_trajectory=settings.include_trajectory,
            include_feedback=settings.include_feedback,
        )
        for index, result in enumerate(retrieved, 1)
    )
    experiences = _truncate(experiences, settings.max_characters)
    return f"""==================================================
RELEVANT EXPERIENCE FROM SIMILAR TASKS
==================================================
{experiences}

==================================================
YOUR CURRENT TASK
==================================================
{query}"""


def build_remem_context(
    query: str,
    retrieved: list[RetrievalResult],
    reasoning_trace: list[str],
    settings: MemoryContextSettings,
) -> str:
    parts: list[str] = []
    if retrieved:
        memories = []
        for index, result in enumerate(retrieved, 1):
            entry = result.entry
            block = [
                f"[Memory {index}]",
                f"Question: {_truncate(entry.input_text, 300)}",
                f"Answer: {_truncate(entry.output_text, 300)}",
            ]
            if settings.include_trajectory and entry.trajectory:
                trajectory_lines = []
                for step in entry.trajectory:
                    if user := step.get("user"):
                        trajectory_lines.append(f"User: {user}")
                    if assistant := step.get("assistant"):
                        trajectory_lines.append(f"Assistant: {assistant}")
                if trajectory_lines:
                    block.append("Trajectory:\n" + _truncate("\n".join(trajectory_lines), 1000))
            if settings.include_feedback and entry.feedback:
                block.append(f"Feedback: {_truncate(entry.feedback, 300)}")
            block.append(f"Result: {'Success' if entry.is_successful else 'Failure'}")
            memories.append("\n".join(block))
        parts.append(
            "RETRIEVED LOCAL EXPERIENCE MEMORIES:\n"
            + _truncate("\n\n".join(memories), settings.max_characters)
        )
    if reasoning_trace:
        trace = "\n".join(reasoning_trace)
        parts.append(
            "CURRENT INTERNAL OPERATION TRACE:\n" + _truncate(trace, settings.max_characters)
        )
    parts.append(f"CURRENT USER REQUEST:\n{query}")
    parts.append(
        """Respond in exactly one of these formats:
- Think: <internal reasoning>
- Think-Prune: <memory IDs, such as 1,3 or 2-4>
- Final Answer: <the complete user-visible answer>"""
    )
    return "\n\n".join(parts)


def replace_latest_user_content(
    messages: list[dict[str, Any]],
    content: str,
) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") == "user":
            result[index]["content"] = content
            return result
    raise ValueError("baseline messages contain no user message to synthesize")


def add_system_prompt(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    if result and result[0].get("role") == "system":
        existing = str(result[0].get("content", "")).strip()
        result[0]["content"] = f"{existing}\n\n{system_prompt}" if existing else system_prompt
        return result
    return [{"role": "system", "content": system_prompt}, *result]


def _truncate(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    return value[:maximum] + "..."
