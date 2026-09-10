from __future__ import annotations

import asyncio
import difflib
import json
import re
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from assistant.config import EnvironmentSettings as AssistantEnvironmentSettings
from assistant.config import GenerationSettings, ModelProfile, load_model_profile
from assistant.llm.base import ChatLLMClient, GeneratedResponse
from assistant.llm.openrouter_client import OpenAICompatibleChatClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from user_simulator.audit.logger import read_git_commit

from interaction_pipeline.trace2skill import (
    BUILD_ARTIFACT_TYPE,
    NO_SKILL,
    Trace2SkillCorpus,
    Trace2SkillError,
    Trace2SkillEvolutionResult,
    Trace2SkillRunContract,
    VisibleTrajectory,
    _atomic_json,
    _atomic_text,
    _canonical_json,
    _hash_text,
    _messages_hash,
    _profile_behavior_sha256,
    _profile_behavior_snapshot,
    _publish_profile_bound_skill,
    _read_json,
    _resolve_run_contract,
    _sha256_file,
    _validate_build_profile,
    _write_assistant_override,
    fixed_chunks,
    load_trace2skill_corpus,
)
from interaction_pipeline.trace2skill_config import Trace2SkillConfig, Trace2SkillRunSettings

# The implementation was audited against this immutable upstream revision.
OFFICIAL_TRACE2SKILL_COMMIT = "3d0b52a140f002a512930252b613c49048f7d5ac"
BUILD_SCHEMA_VERSION = 2
BUILD_PIPELINE = "official-parallel-json-adapted-v1"
ANALYSIS_ARTIFACT_TYPE = "trace2skill_analysis_record"
MAP_ARTIFACT_TYPE = "trace2skill_map_patch"
MERGE_ARTIFACT_TYPE = "trace2skill_merge"
TRANSLATION_ARTIFACT_TYPE = "trace2skill_translation"

SUPPORTED_PATCH_OPS = {
    "insert_after",
    "insert_before",
    "append_to_section",
    "replace_in_section",
    "add_section",
    "delete_section",
}
PATCH_OP_ALIASES = {
    "create_file": "create",
    "createFile": "create",
    "deleteFile": "delete_file",
}
REFERENCE_PATH_PATTERN = re.compile(r"references/[\w.\-]+\.md")


# These prompts preserve the upstream roles, workflows, record schema, and patch
# schema. Domain-specific spreadsheet/filesystem instructions were replaced only
# where intentLLM exposes a visible dialogue and consumes one Markdown system prompt.
SUCCESS_ANALYSIS_SYSTEM_PROMPT = """# Role
You are an expert in successful AI-agent trajectory analysis.

# Mission
Given a successful visible user/assistant dialogue, produce two things:

1. **Lean Solution Path** — distill the minimal, clean sequence of dialogue actions that
actually led to completion. Remove wrong turns, dead ends, and self-corrections.
2. **Success Memory Items** — extract generalizable lessons that could help an assistant
handle similar conversations in the future.

Your analysis must be evidence-driven. Use only actions and observations in the supplied
dialogue. Do not infer hidden intents, evaluator state, metadata, or unstated requirements.

# Required Workflow
1. Understand the visible user request.
2. Identify the minimal successful interaction path.
3. Express that path as a compact sequence.
4. Extract no more than three generalizable Success Memory Items.

# Output Requirements

```markdown
# Lean Solution Path

## Overview
<what the user visibly asked and what interaction approach worked>

## Step 1: <Action title>
<concrete visible action>

# Success Memory Item 1

## Title
<short title>

## Description
<one-sentence summary>

## Content
<one to three sentences describing the reusable strategy>
```

Every memory item must be generalizable and grounded in observable dialogue. Never mention
ground truth, hidden task graphs, evaluator decisions, user metadata, or internal simulator state.
Do not fabricate actions or evidence.
"""


FAILURE_ANALYSIS_SYSTEM_PROMPT = """# Role
You are an expert failure-analysis agent for AI-assistant conversations.

# Mission
Given a visible dialogue that did not naturally complete within its fixed turn budget, diagnose
causal failures only when they are supported by observable dialogue. You have no ground truth,
hidden intent, evaluator state, task graph, metadata, tools, or artifacts. If the dialogue does
not causally establish a failure mechanism, do not guess and emit no item headings.

# Required Workflow
1. Understand the visible user request and interaction.
2. Identify a concrete wrong decision, assumption, omission, or repeated behavior only when the
dialogue demonstrates it.
3. Trace each failure claim to observable assistant behavior.
4. Extract no more than three generalizable Failure Memory Items.

# Output Requirements

```markdown
# Failure Cause Item 1

## Title
<short title>

## Description
<one-sentence causal summary>

## Content
<one to three evidence-grounded sentences>

# Failure Memory Item 1

## Title
<short title>

## Description
<one-sentence lesson>

## Content
<one to three sentences describing the reusable strategy>
```

Never mention or imply ground truth, hidden intents, evaluator state, task graphs, metadata, or
unstated requirements. Do not fabricate evidence. When no causal diagnosis is supported, respond
with `# No Causally Supported Failure` and a brief explanation, with no item headings.
"""


ANALYSIS_USER_TEMPLATE = """Here is the target assistant's visible dialogue. The outcome was
already used to select the appropriate analyst role.

<agent_log>
{agent_log}
</agent_log>

Follow the workflow and output format in the system prompt exactly.
"""


MAP_SYSTEM_PROMPT = """You are a skill editor specializing in both reducing AI-assistant
failures and reinforcing successful behaviors observed in practice. Your task is to refine an
assistant skill so that it avoids recurring mistakes while preserving workflows that repeatedly
succeed.

## What is the Skill

For this task adaptation, the complete skill is one file, `SKILL.md`. Its Markdown content is used
verbatim as the assistant's only system message. It shares the context window with the task and
dialogue, so conciseness matters. There are no reference files, scripts, assets, or YAML
frontmatter. The frozen initial `SKILL.md` is empty because the development trajectories were
collected with No Skill.

You receive:
1. The current contents of `SKILL.md`.
2. A parsed error or success analysis record from a trajectory collected with this initial skill.
3. The skill size constraint.

Your job is to propose a minimal `SKILL.md` patch that helps prevent recurring failures,
reinforces proven successful behavior, and keeps the skill concise and organized.

## Modification Strategies

Apply whichever strategies fit the supplied evidence. Not every record requires every strategy.

### Strategy 1: Balance Failure Prevention with Proven Success

- Use failures to identify what must change and successes to identify what must survive.
- If failure and success evidence point to the same root issue, write one targeted instruction
  that prevents the mistake and reinforces the working approach.
- Distill Failure Memory items into actions the assistant must take, and Success Memory items into
  actions it should keep taking.
- Be specific. “Handle requests carefully” is weak; an observable decision rule or verification
  step is useful.

### Strategy 2: Adjust Degrees of Freedom from Both Signals

Match specificity to the evidence:

- **High freedom**: text instructions when multiple approaches succeed without the failure.
- **Medium freedom**: pseudocode or parameterized steps when one family of workflows works best.
- **Low freedom**: exact sequences when repeated failures disappear only with that sequence.

When failures repeat, tighten the guidance with concrete checks or warnings. When success evidence
shows guidance would be too rigid, loosen it without reintroducing the failure mode.

### Strategy 3: Organize Around Reusable Patterns

Keep `SKILL.md` focused on the minimum workflow that both avoids common failures and preserves
proven wins. This runtime cannot load references, so all retained guidance must be concise and
self-contained in `SKILL.md`.

### Strategy 4: Prefer Evidence-Weighted Concision

- Remove redundant guidance and consolidate overlapping lessons.
- Keep instructions short enough to coexist with the dialogue context.
- Ignore one-off evidence that does not generalize.
- Prefer changes supported by repeated evidence over anecdotes.

### Strategy 5: Other Best Practices

- Use imperative form (for example, “Verify the answer,” not “You should verify the answer”).
- Do not add frontmatter, README files, changelogs, references, or auxiliary files.
- Preserve existing correct guidance.

## Understanding Mixed Analysis Records

Error-record items are `failure_cause` or `failure_memory`; use them to identify failure modes,
missing checks, and weak guidance. Success-record items are `success_memory`; use them to identify
robust workflows and verification habits worth reinforcing.

Prevent recurring failures, reinforce proven strategies, resolve conflicts in favor of guidance
that preserves successful workflows, and prefer minimal patches. Ignore one-off records, bloated
guidance, and suggestions that conflict with stronger evidence.

## Constraints

1. Do not remove currently useful guidance unless the evidence supports a leaner replacement.
2. Minimal patches are preferred over large rewrites; multiple small edits are better than one
   large rewrite.
3. Do not introduce YAML frontmatter.
4. `SKILL.md` must remain at or below five hundred lines.
5. Every change must trace to observed failure or success evidence in the supplied record.
6. Use only `SKILL.md`; do not create references or other files.

## Output Format

Respond with JSON in a fenced `json` block:

```json
{"reasoning":"two or three sentences explaining the evidence and edits","edits":[{"file":"SKILL.md","op":"add_section","target_section":"## Section Name","content":"concise guidance"}],"changelog_entries":["Brief description of change"]}
```

Supported operations:
- `insert_after`: insert content after `target_text` within `target_section`
- `insert_before`: insert content before `target_text` within `target_section`
- `append_to_section`: append content at the end of `target_section`
- `replace_in_section`: replace `old_text` with `content` within `target_section`
- `add_section`: add a new section, using `after_section` for placement
- `delete_section`: remove `target_section` entirely

If no change is justified, return an empty `edits` list. Propose minimal, targeted edits rather
than complete file rewrites.
"""


MERGE_SYSTEM_PROMPT = """You are a skill edit coordinator. You receive multiple independently
proposed patches against the same frozen initial `SKILL.md`, based on mixed failure and success
evidence. Merge them into one coherent, non-redundant patch.

Guidelines:
1. Deduplicate similar edits and keep the strongest wording.
2. Resolve contradictions by choosing the stronger evidence fit or synthesizing a better edit.
3. Preserve unique, useful insights from different patches.
4. Keep the merged patch no larger than the sum of unique input edits; remove redundancy.
5. Keep the same operation format.
6. Ensure edits are line-level independent: no two edits may target overlapping lines or the same
   passage. Multiple edits to distinct sections are allowed.
7. Use only `SKILL.md`; never introduce another file.

Return one or more fenced JSON patch objects using exactly this schema:

```json
{"reasoning":"what was merged and deduplicated","edits":[{"file":"SKILL.md","op":"add_section","target_section":"## Section Name","content":"merged guidance"}],"changelog_entries":["Merged: description"]}
```

Supported operations: insert_after, insert_before, append_to_section, replace_in_section,
add_section, delete_section. An empty `edits` list is allowed when no change remains justified.
"""


TRANSLATION_SYSTEM_PROMPT = """You are a skill editor assistant. You receive the current content
of `SKILL.md` and one suggested edit. Its text references may be inexact. Translate it so that
`target_section`, `target_text`, or `old_text` exactly matches the current file when possible,
while preserving the intent, operation, and content. If a target cannot be matched, return the
edit unchanged. Return edits only for `SKILL.md`.

Respond with JSON in a fenced `json` block:

```json
{"reasoning":"Brief note on what was corrected","edits":[{"file":"SKILL.md","op":"replace_in_section","target_section":"## Exact Header","old_text":"exact existing text","content":"replacement"}],"changelog_entries":[]}
```

Use only: insert_after, insert_before, append_to_section, replace_in_section, add_section,
delete_section.
"""


VERIFICATION_SYSTEM_PROMPT = """You are a skill editor performing a validation-fix pass. The
single-file assistant skill was just modified but failed format validation. Fix only the reported
issue while preserving all correct content.

The valid task format is a non-empty Markdown `SKILL.md` of at most five hundred lines. It is
used verbatim as a system message and therefore has no YAML frontmatter.

Return a fenced JSON patch with `reasoning`, `edits`, and `changelog_entries`. Use only
`SKILL.md` and these operations: insert_after, insert_before, append_to_section,
replace_in_section, add_section, delete_section.
"""


CONTINUE_JSON_PROMPT = (
    "Your response was cut off or missing a complete fenced json block. Continue from where "
    "you stopped or provide the complete JSON in a single fenced json block. Output ONLY the "
    "remaining text."
)
JSON_FORMAT_FIX_TEMPLATE = """Your previous response could not be parsed as valid JSON. Keep the
same intended edits/content, but fix only the formatting. Here is the exact parser feedback:
{feedback} Use this exact format:
```json
{{"reasoning":"Brief summary","edits":[],"changelog_entries":[]}}
```
Output ONLY the corrected fenced json block.
"""


class AnalysisItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["failure_cause", "failure_memory", "success_memory"]
    number: int = Field(ge=1)
    title: str = ""
    description: str = ""
    content: str = ""
    relation_to_skill: str = ""
    skill_reflection: str = ""


class AnalysisRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_source: Literal["error", "success"]
    items: list[AnalysisItem]


class PatchEdit(BaseModel):
    """The upstream JSON PatchEdit schema before deterministic sanitization."""

    model_config = ConfigDict(extra="ignore")

    file: str = Field(min_length=1)
    op: str = Field(min_length=1)
    target_section: str = ""
    target_text: str = ""
    content: str = ""
    old_text: str = ""
    after_section: str = ""

    @model_validator(mode="after")
    def normalize_operation_alias(self) -> PatchEdit:
        self.op = PATCH_OP_ALIASES.get(self.op, self.op)
        return self


class SkillPatch(BaseModel):
    """A concise instruction-based patch used by official parallel JSON evolution."""

    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""
    edits: list[PatchEdit] = Field(default_factory=list)
    changelog_entries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_strings(self) -> SkillPatch:
        self.reasoning = self.reasoning.strip()
        self.changelog_entries = [str(item).strip() for item in self.changelog_entries]
        return self

    def merger_view(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PatchCallError(Trace2SkillError):
    def __init__(
        self,
        message: str,
        *,
        calls: list[dict[str, Any]],
        conversation: list[dict[str, str]],
        final_response: str | None,
    ) -> None:
        super().__init__(message)
        self.calls = calls
        self.conversation = conversation
        self.final_response = final_response


BuildProgressCallback = Callable[[int, int, dict[str, Any]], None]


_SUCCESS_ITEM_RE = re.compile(
    r"^(?P<level>#+)\s+Success Memory Item\s+(?P<number>\d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_FAILURE_ITEM_RE = re.compile(
    r"^#\s+(?P<kind>Failure Cause Item|Failure Memory Item)\s+"
    r"(?P<number>\d+)\s*\n(?P<body>.*?)(?=\n#\s+(?:Failure Cause Item|Failure Memory Item)\s+\d+|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_SECTION_RE_TEMPLATE = r"^##\s+{name}\s*\n(.*?)(?=\n##\s+|\n#\s+|\Z)"


def _strip_think_prefix(text: str) -> str:
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def _strip_outer_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```\w*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _section(body: str, name: str) -> str:
    match = re.search(
        _SECTION_RE_TEMPLATE.format(name=re.escape(name)),
        body,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def parse_analysis_items(text: str, *, outcome: str) -> list[AnalysisItem]:
    """Parse the two official analysis-report schemas; zero items means exclusion."""

    value = _strip_outer_fence(_strip_think_prefix(text))
    items: list[AnalysisItem] = []
    if outcome == "SUCCESS":
        matches = list(_SUCCESS_ITEM_RE.finditer(value))
        for index, match in enumerate(matches):
            body_start = match.end()
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
            body = value[body_start:body_end]
            items.append(
                AnalysisItem(
                    type="success_memory",
                    number=int(match.group("number")),
                    title=_section(body, "Title"),
                    description=_section(body, "Description"),
                    content=_section(body, "Content"),
                )
            )
    else:
        for match in _FAILURE_ITEM_RE.finditer(value):
            kind = match.group("kind").casefold()
            item_type = "failure_cause" if "cause" in kind else "failure_memory"
            body = match.group("body")
            items.append(
                AnalysisItem(
                    type=cast(Any, item_type),
                    number=int(match.group("number")),
                    title=_section(body, "Title"),
                    description=_section(body, "Description"),
                    content=_section(body, "Content"),
                    relation_to_skill=_section(body, "Relation to Skill"),
                    skill_reflection=_section(body, "Skill Reflection"),
                )
            )
    return items


def _dialogue_log(trajectory: VisibleTrajectory) -> str:
    parts: list[str] = []
    for message in trajectory.dialogue:
        parts.extend((f"## {message.role.title()}", message.content, ""))
    return "\n".join(parts).strip()


def build_analysis_messages(trajectory: VisibleTrajectory) -> list[dict[str, str]]:
    system = (
        SUCCESS_ANALYSIS_SYSTEM_PROMPT
        if trajectory.outcome == "SUCCESS"
        else FAILURE_ANALYSIS_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": ANALYSIS_USER_TEMPLATE.format(agent_log=_dialogue_log(trajectory)),
        },
    ]


def _format_analysis_item(item: AnalysisItem) -> str:
    labels = {
        "failure_cause": "Failure Cause",
        "failure_memory": "Failure Memory",
        "success_memory": "Success Memory",
    }
    parts = [f"**{labels[item.type]}: {item.title or 'Untitled'}**"]
    if item.description:
        parts.append(f"*{item.description}*")
    if item.content:
        parts.append(item.content)
    if item.relation_to_skill:
        parts.append(f"*Relation To Skill (suggestion)*: {item.relation_to_skill}")
    if item.skill_reflection:
        parts.append(f"*Skill Reflection (suggestion)*: {item.skill_reflection}")
    return "\n".join(parts)


def build_map_messages(record: AnalysisRecord) -> list[dict[str, str]]:
    source_label = "Error" if record.record_source == "error" else "Success"
    items = "\n\n".join(_format_analysis_item(item) for item in record.items)
    user = f"""## Current Skill Folder Contents (Frozen)

### SKILL.md (0 lines)
```markdown
```

## Combined Analysis Record

### {source_label} Record
{items}

## Skill Folder Size Status
- SKILL.md: 0 lines (limit: 500)
- Reference files: unsupported in this task adaptation
"""
    return [
        {"role": "system", "content": MAP_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_merge_messages(patches: Sequence[SkillPatch]) -> list[dict[str, str]]:
    parts = [
        "# Original Skill Folder (Frozen)",
        "",
        "### SKILL.md (0 lines)",
        "```markdown",
        "```",
        "",
        f"# Patches to Merge ({len(patches)})",
        "",
    ]
    for index, patch in enumerate(patches, 1):
        parts.extend((f"### Patch {index}", f"**Reasoning**: {patch.reasoning}"))
        parts.append(f"**Edits** ({len(patch.edits)}):")
        for edit in patch.edits:
            edit_data: dict[str, str] = {"file": edit.file, "op": edit.op}
            for field in (
                "target_section",
                "target_text",
                "content",
                "old_text",
                "after_section",
            ):
                value = getattr(edit, field)
                if value:
                    edit_data[field] = value
            parts.append(f"  - {json.dumps(edit_data, ensure_ascii=False)}")
        parts.extend((f"**Changelog**: {patch.changelog_entries}", ""))
    return [
        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_translation_messages(skill_content: str, edit: PatchEdit) -> list[dict[str, str]]:
    user = f"""# Current Content of SKILL.md
```markdown
{skill_content}
```

# Edit to Translate
```json
{json.dumps(edit.model_dump(mode="json"), ensure_ascii=False, indent=2)}
```

Correct text references only and return all translated edits for SKILL.md.
"""
    return [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_verification_messages(skill_content: str, error: str) -> list[dict[str, str]]:
    user = f"""# Skill Folder That Failed Validation

### SKILL.md
```markdown
{skill_content}
```

# Validation Error
```
{error}
```

Return the minimal patch required to fix this error.
"""
    return [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _patch_payload_issues(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["Top-level JSON must be an object."]
    issues: list[str] = []
    if "reasoning" not in value:
        issues.append("Missing top-level field: reasoning")
    if "edits" not in value:
        issues.append("Missing top-level field: edits")
    elif not isinstance(value.get("edits"), list):
        issues.append("Top-level field 'edits' must be a list")
    if "changelog_entries" not in value:
        issues.append("Missing top-level field: changelog_entries")
    elif not isinstance(value.get("changelog_entries"), list):
        issues.append("Top-level field 'changelog_entries' must be a list")
    edits = value.get("edits")
    if isinstance(edits, list):
        for index, edit in enumerate(edits, 1):
            if not isinstance(edit, dict):
                issues.append(f"Edit #{index} must be an object")
                continue
            if not edit.get("file"):
                issues.append(f"Edit #{index} is missing required field: file")
            if not edit.get("op"):
                issues.append(f"Edit #{index} is missing required field: op")
    return issues


def _extract_outer_json_block(text: str) -> tuple[str | None, bool]:
    opening = re.search(r"```json[^\n]*\n", text, re.IGNORECASE)
    if opening is None:
        return None, False
    tail = text[opening.end() :]
    closings = list(re.finditer(r"(?m)^```[ \t]*\r?$", tail))
    if not closings:
        return tail.strip(), False
    return tail[: closings[-1].start()].strip(), True


def _find_jsonish_key(text: str, key: str, start: int = 0) -> int | None:
    for pattern in (
        rf'"{re.escape(key)}"\s*:',
        rf'{re.escape(key)}"\s*:',
        rf'"{re.escape(key)}\s*:',
        rf"(?<![A-Za-z0-9_]){re.escape(key)}\s*:",
    ):
        match = re.search(pattern, text[start:])
        if match:
            return start + match.end()
    return None


def _extract_jsonish_string(text: str, start: int) -> tuple[str | None, int]:
    if start >= len(text) or text[start] != '"':
        return None, start
    index = start + 1
    chars: list[str] = []
    while index < len(text):
        character = text[index]
        if character == "\\":
            if index + 1 >= len(text):
                break
            escaped = text[index + 1]
            chars.append(
                {
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "b": "\b",
                    "f": "\f",
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                }.get(escaped, escaped)
            )
            index += 2
            continue
        if character == '"':
            return "".join(chars), index + 1
        chars.append(character)
        index += 1
    return None, start


def _extract_jsonish_string_field(text: str, key: str) -> str | None:
    start = _find_jsonish_key(text, key)
    if start is None:
        return None
    while start < len(text) and text[start].isspace():
        start += 1
    value, _ = _extract_jsonish_string(text, start)
    return value


def _extract_jsonish_object_chunks(array_text: str) -> list[str]:
    inner = array_text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    chunks: list[str] = []
    depth = 0
    start: int | None = None
    for index, character in enumerate(inner):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(inner[start : index + 1])
                start = None
    return chunks


def _extract_jsonish_string_list(array_text: str) -> list[str]:
    inner = array_text.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    values: list[str] = []
    index = 0
    while index < len(inner):
        if inner[index] != '"':
            index += 1
            continue
        value, next_index = _extract_jsonish_string(inner, index)
        if value is None:
            break
        values.append(value)
        index = next_index
    return values


def _heuristic_parse_patch_payload(raw: str) -> dict[str, Any] | None:
    """Port of upstream's narrow recovery for malformed patch JSON."""

    reasoning = _extract_jsonish_string_field(raw, "reasoning")
    edits_match = re.search(
        r'"?edits"?\s*:\s*(\[.*?\])\s*,\s*"?changelog_entries"?\s*:',
        raw,
        re.DOTALL,
    )
    changelog_match = re.search(r'"?changelog_entries"?\s*:\s*(\[[\s\S]*\])', raw)
    if reasoning is None or edits_match is None or changelog_match is None:
        return None
    edits: list[dict[str, str]] = []
    for chunk in _extract_jsonish_object_chunks(edits_match.group(1)):
        edit: dict[str, str] = {}
        for key in (
            "file",
            "op",
            "target_section",
            "target_text",
            "content",
            "old_text",
            "after_section",
        ):
            value = _extract_jsonish_string_field(chunk, key)
            if value is not None:
                edit[key] = value
        if edit:
            edits.append(edit)
    payload: dict[str, Any] = {
        "reasoning": reasoning,
        "edits": edits,
        "changelog_entries": _extract_jsonish_string_list(changelog_match.group(1)),
    }
    return None if _patch_payload_issues(payload) else payload


def parse_patch_response(text: str) -> tuple[list[SkillPatch], str]:
    """Mirror upstream fenced/bare parsing and heuristic malformed-JSON recovery."""

    stripped = _strip_think_prefix(text).strip()
    block, closed = _extract_outer_json_block(stripped)
    raw: str
    if block is not None:
        if not closed:
            return [], "Found an opening ```json fence but the closing ``` fence is missing."
        raw = block
    elif stripped.startswith("{"):
        raw = stripped
    else:
        return [], "No fenced json block found."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        payload = _heuristic_parse_patch_payload(raw)
        if payload is None:
            return [], (
                f"JSON decode error: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}, char {exc.pos})"
            )
    issues = _patch_payload_issues(payload)
    if issues:
        return [], "; ".join(issues)
    try:
        return [SkillPatch.model_validate(payload)], ""
    except ValidationError as exc:
        return [], str(exc)


def coalesce_patches(patches: Sequence[SkillPatch]) -> SkillPatch:
    if not patches:
        raise ValueError("cannot coalesce an empty patch sequence")
    if len(patches) == 1:
        return patches[0]
    return SkillPatch(
        reasoning="\n\n".join(patch.reasoning for patch in patches if patch.reasoning),
        edits=[edit for patch in patches for edit in patch.edits],
        changelog_entries=[entry for patch in patches for entry in patch.changelog_entries],
    )


def _public_llm_metadata(client: ChatLLMClient, response: GeneratedResponse) -> dict[str, Any]:
    response_metadata = getattr(response, "metadata", None)
    metadata = (
        response_metadata
        if isinstance(response_metadata, dict)
        else getattr(client, "last_call_metadata", {})
    )
    if not isinstance(metadata, dict):
        return {}
    forbidden = {"messages", "response", "reasoning_content", "reasoning_details"}
    return {key: value for key, value in metadata.items() if key not in forbidden}


def _response_call_record(
    client: ChatLLMClient,
    response: GeneratedResponse,
    *,
    sequence: int,
    kind: str,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "kind": kind,
        "status": "response",
        "response_sha256": _hash_text(response.content),
        "response": response.content,
        "llm_call": _public_llm_metadata(client, response),
    }


def _error_call_record(error: Exception, *, sequence: int, kind: str) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "kind": kind,
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
        "llm_call": {},
    }


async def call_patch_with_official_repair(
    *,
    client: ChatLLMClient,
    generation: GenerationSettings,
    messages: list[dict[str, str]],
    max_continuations: int = 2,
    max_format_fix_rounds: int = 2,
) -> tuple[list[SkillPatch], list[dict[str, Any]], list[dict[str, str]], str]:
    """Mirror the official initial -> continuation -> format-fix conversation."""

    conversation = [dict(message) for message in messages]
    calls: list[dict[str, Any]] = []
    sequence = 0

    async def invoke(kind: str) -> GeneratedResponse:
        nonlocal sequence
        sequence += 1
        try:
            response = await client.generate(messages=conversation, generation=generation)
        except Exception as exc:
            calls.append(_error_call_record(exc, sequence=sequence, kind=kind))
            raise PatchCallError(
                f"{kind} request failed: {type(exc).__name__}: {exc}",
                calls=calls,
                conversation=conversation,
                final_response=None,
            ) from exc
        calls.append(_response_call_record(client, response, sequence=sequence, kind=kind))
        return response

    response = await invoke("initial")
    last_response = response.content
    full_response = last_response

    for _ in range(max_continuations):
        patches, _ = parse_patch_response(full_response)
        if patches:
            break
        # Upstream does not ask for a continuation when a fenced JSON block is
        # already closed but malformed; it moves directly to targeted format repair.
        _, has_closed_fence = _extract_outer_json_block(full_response)
        if has_closed_fence:
            break
        # The upstream implementation asks for continuation only while a complete
        # parseable block is absent, and retains the prior assistant turn.
        conversation.extend(
            (
                {"role": "assistant", "content": _strip_think_prefix(last_response)},
                {"role": "user", "content": CONTINUE_JSON_PROMPT},
            )
        )
        continuation = await invoke("continuation")
        last_response = continuation.content
        full_response += continuation.content

    for _ in range(max_format_fix_rounds):
        patches, feedback = parse_patch_response(full_response)
        if patches:
            return (
                patches,
                calls,
                [
                    *conversation,
                    {"role": "assistant", "content": _strip_think_prefix(last_response)},
                ],
                full_response,
            )
        conversation.extend(
            (
                {"role": "assistant", "content": _strip_think_prefix(last_response)},
                {
                    "role": "user",
                    "content": JSON_FORMAT_FIX_TEMPLATE.format(feedback=feedback),
                },
            )
        )
        fixed = await invoke("format_fix")
        last_response = fixed.content
        full_response = last_response

    patches, feedback = parse_patch_response(full_response)
    if patches:
        return (
            patches,
            calls,
            [
                *conversation,
                {"role": "assistant", "content": _strip_think_prefix(last_response)},
            ],
            full_response,
        )
    raise PatchCallError(
        f"patch response remained unparseable after official repair: {feedback}",
        calls=calls,
        conversation=conversation,
        final_response=full_response,
    )


async def _call_analysis_once(
    *,
    client: ChatLLMClient,
    generation: GenerationSettings,
    messages: list[dict[str, str]],
) -> tuple[GeneratedResponse | None, list[dict[str, Any]], str | None]:
    try:
        response = await client.generate(messages=messages, generation=generation)
    except Exception as exc:  # noqa: BLE001 - one official analysis slot is an isolation boundary
        return None, [_error_call_record(exc, sequence=1, kind="analysis")], str(exc)
    return (
        response,
        [_response_call_record(client, response, sequence=1, kind="analysis")],
        None,
    )


def _analysis_artifact_hash(artifact: dict[str, Any]) -> str:
    record = artifact.get("record")
    return _hash_text(_canonical_json(record)) if isinstance(record, dict) else _hash_text("")


async def process_analysis_slot(
    *,
    artifact_path: Path,
    dataset_index: int,
    sample_id: str,
    source_trajectory_sha256: str,
    trajectory: VisibleTrajectory,
    client: ChatLLMClient,
    generation: GenerationSettings,
) -> dict[str, Any]:
    if artifact_path.exists():
        existing = _read_json(artifact_path)
        if (
            existing.get("schema_version") != BUILD_SCHEMA_VERSION
            or existing.get("artifact_type") != ANALYSIS_ARTIFACT_TYPE
            or existing.get("dataset_index") != dataset_index
            or existing.get("sample_id") != sample_id
            or existing.get("source_trajectory_sha256") != source_trajectory_sha256
        ):
            raise Trace2SkillError(f"incompatible analysis artifact: {artifact_path}")
        if existing.get("status") in {"complete", "skipped"}:
            if existing.get("status") == "complete":
                AnalysisRecord.model_validate(existing.get("record"))
            return existing

    messages = build_analysis_messages(trajectory)
    response, calls, error = await _call_analysis_once(
        client=client,
        generation=generation,
        messages=messages,
    )
    analyst = "success" if trajectory.outcome == "SUCCESS" else "failure"
    base = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_ARTIFACT_TYPE,
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "source_trajectory_sha256": source_trajectory_sha256,
        "analyst": analyst,
        "system_prompt_sha256": _hash_text(messages[0]["content"]),
        "request_sha256": _messages_hash(messages),
        "calls": calls,
    }
    if response is None:
        artifact = {
            **base,
            "conversation": messages,
            "status": "skipped",
            "skip_reason": "analysis_request_failed_after_profile_transport_retries",
            "error": error,
            "record": None,
        }
        _atomic_json(artifact_path, artifact)
        return artifact

    items = parse_analysis_items(response.content, outcome=trajectory.outcome)
    conversation = [
        *messages,
        {"role": "assistant", "content": _strip_think_prefix(response.content)},
    ]
    if not items:
        artifact = {
            **base,
            "conversation": conversation,
            "status": "skipped",
            "skip_reason": "no_analysis_items_parsed",
            "analysis_report": response.content,
            "record": None,
        }
        _atomic_json(artifact_path, artifact)
        return artifact

    record = AnalysisRecord(
        record_source="success" if trajectory.outcome == "SUCCESS" else "error",
        items=items,
    )
    artifact = {
        **base,
        "conversation": conversation,
        "status": "complete",
        "analysis_report": response.content,
        "record": record.model_dump(mode="json"),
    }
    _atomic_json(artifact_path, artifact)
    return artifact


async def process_map_slot(
    *,
    artifact_path: Path,
    dataset_index: int,
    sample_id: str,
    source_analysis: dict[str, Any],
    client: ChatLLMClient,
    generation: GenerationSettings,
    settings: Trace2SkillRunSettings,
) -> dict[str, Any]:
    source_hash = _analysis_artifact_hash(source_analysis)
    if artifact_path.exists():
        existing = _read_json(artifact_path)
        if (
            existing.get("schema_version") != BUILD_SCHEMA_VERSION
            or existing.get("artifact_type") != MAP_ARTIFACT_TYPE
            or existing.get("dataset_index") != dataset_index
            or existing.get("sample_id") != sample_id
            or existing.get("source_analysis_sha256") != source_hash
        ):
            raise Trace2SkillError(f"incompatible MAP artifact: {artifact_path}")
        if existing.get("status") in {"complete", "skipped"}:
            if existing.get("status") == "complete":
                SkillPatch.model_validate(existing.get("patch"))
            return existing

    base = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "artifact_type": MAP_ARTIFACT_TYPE,
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "source_analysis_sha256": source_hash,
    }
    if source_analysis.get("status") != "complete":
        artifact = {
            **base,
            "status": "skipped",
            "skip_reason": "source_analysis_excluded",
            "calls": [],
            "patch": None,
        }
        _atomic_json(artifact_path, artifact)
        return artifact

    record = AnalysisRecord.model_validate(source_analysis["record"])
    messages = build_map_messages(record)
    base.update(
        {
            "system_prompt_sha256": _hash_text(messages[0]["content"]),
            "request_sha256": _messages_hash(messages),
        }
    )
    try:
        patches, calls, conversation, final_response = await call_patch_with_official_repair(
            client=client,
            generation=generation,
            messages=messages,
            max_continuations=settings.max_continuations,
            max_format_fix_rounds=settings.format_fix_rounds,
        )
    except PatchCallError as exc:
        artifact = {
            **base,
            "status": "skipped",
            "skip_reason": "map_patch_unavailable_after_official_repair",
            "calls": exc.calls,
            "conversation": exc.conversation,
            "final_response": exc.final_response,
            "error": str(exc),
            "patch": None,
        }
        _atomic_json(artifact_path, artifact)
        return artifact

    # The official output contract requests one patch but its parser accepts
    # multiple fenced payloads. Coalescing preserves them while retaining the
    # task-required one-trajectory/one-slot topology.
    patch = coalesce_patches(patches)
    artifact = {
        **base,
        "status": "complete",
        "calls": calls,
        "conversation": conversation,
        "final_response": final_response,
        "parsed_patch_count": len(patches),
        "patch": patch.model_dump(mode="json"),
    }
    _atomic_json(artifact_path, artifact)
    return artifact


def _patches_hash(patches: Sequence[SkillPatch]) -> str:
    return _hash_text(_canonical_json([patch.merger_view() for patch in patches]))


def balanced_chunks_upstream(
    items: Sequence[SkillPatch], batch_size: int
) -> list[list[SkillPatch]]:
    """Port upstream ``chunk_list`` for post-level-one reduction batches."""

    total = len(items)
    if total == 0:
        return []
    if total <= batch_size:
        return [list(items)]
    full_count, remainder = divmod(total, batch_size)
    if remainder == 0:
        return [list(items[index : index + batch_size]) for index in range(0, total, batch_size)]
    batches = [
        list(items[index : index + batch_size])
        for index in range(0, full_count * batch_size, batch_size)
    ]
    tail = items[full_count * batch_size :]
    if remainder < batch_size / 2:
        for index, item in enumerate(tail):
            batches[index % len(batches)].append(item)
    else:
        batches.append(list(tail))
    return batches


def order_combined_map_patches(
    map_slots: Sequence[dict[str, Any]],
    trajectories: Sequence[VisibleTrajectory],
) -> tuple[list[SkillPatch], list[int]]:
    """Match upstream combined-record order and omit unavailable MAP outputs."""

    if len(map_slots) != len(trajectories):
        raise ValueError("MAP slots and trajectories must have identical lengths")
    ordered_indices: list[int] = []
    for outcome in ("FAILURE_TURN_LIMIT", "SUCCESS"):
        matching = [
            index
            for index, trajectory in enumerate(trajectories)
            if trajectory.outcome == outcome and map_slots[index].get("status") == "complete"
        ]
        # Upstream collectors parse each analysis directory in lexicographic
        # report-filename (instance-id) order before concatenating error+success.
        matching.sort(key=lambda index: (str(map_slots[index].get("sample_id", "")), index))
        ordered_indices.extend(matching)
    return (
        [SkillPatch.model_validate(map_slots[index]["patch"]) for index in ordered_indices],
        ordered_indices,
    )


def sanitize_map_patches_for_single_file(
    patches: Sequence[SkillPatch], dataset_indices: Sequence[int]
) -> tuple[list[SkillPatch], list[dict[str, Any]]]:
    """Adapt upstream MAP create/link pairing to a runtime with no auxiliary files."""

    if len(patches) != len(dataset_indices):
        raise ValueError("MAP patches and dataset indices must have identical lengths")
    sanitized: list[SkillPatch] = []
    records: list[dict[str, Any]] = []
    for patch, dataset_index in zip(patches, dataset_indices, strict=True):
        kept: list[PatchEdit] = []
        dropped: list[dict[str, Any]] = []
        for edit_index, source_edit in enumerate(patch.edits):
            edit = source_edit.model_copy(deep=True)
            normalized_file = edit.file.replace("\\", "/").lstrip("./")
            reason: str | None = None
            if normalized_file != "SKILL.md":
                reason = "outside_single_file_skill"
            elif edit.op in {"create", "delete_file"}:
                reason = "unsupported_file_level_operation"
            elif REFERENCE_PATH_PATTERN.search(edit.content):
                reason = "unsupported_reference_link"
            if reason is not None:
                dropped.append(
                    {
                        "edit_index": edit_index,
                        "reason": reason,
                        "edit": source_edit.model_dump(mode="json"),
                    }
                )
                continue
            edit.file = "SKILL.md"
            kept.append(edit)
        sanitized.append(
            SkillPatch(
                reasoning=patch.reasoning,
                edits=kept,
                changelog_entries=patch.changelog_entries,
            )
        )
        if dropped:
            records.append({"dataset_index": dataset_index, "dropped_edits": dropped})
    return sanitized, records


def _read_merge_artifact(
    path: Path,
    *,
    expected_level: int,
    expected_index: int,
    expected_input_hash: str,
) -> list[SkillPatch] | None:
    if not path.exists():
        return None
    value = _read_json(path)
    if (
        value.get("schema_version") != BUILD_SCHEMA_VERSION
        or value.get("artifact_type") != MERGE_ARTIFACT_TYPE
        or value.get("level") != expected_level
        or value.get("group_index") != expected_index
        or value.get("input_patches_sha256") != expected_input_hash
        or value.get("status") not in {"merged", "fallback", "empty"}
    ):
        raise Trace2SkillError(f"incompatible merge artifact: {path}")
    raw = value.get("output_patches")
    if not isinstance(raw, list):
        raise Trace2SkillError(f"merge artifact has no output patch list: {path}")
    return [SkillPatch.model_validate(item) for item in raw]


async def merge_patch_group(
    *,
    artifact_path: Path,
    level: int,
    group_index: int,
    patches: list[SkillPatch],
    client: ChatLLMClient,
    generation: GenerationSettings,
    settings: Trace2SkillRunSettings,
) -> list[SkillPatch]:
    input_hash = _patches_hash(patches)
    existing = _read_merge_artifact(
        artifact_path,
        expected_level=level,
        expected_index=group_index,
        expected_input_hash=input_hash,
    )
    if existing is not None:
        return existing

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "artifact_type": MERGE_ARTIFACT_TYPE,
        "level": level,
        "group_index": group_index,
        "input_patch_count": len(patches),
        "input_patches_sha256": input_hash,
        "initial_skill_sha256": _hash_text(""),
        "system_prompt_sha256": _hash_text(MERGE_SYSTEM_PROMPT),
    }
    if not patches:
        artifact = {
            **base,
            "status": "empty",
            "calls": [],
            "output_patches": [],
        }
        _atomic_json(artifact_path, artifact)
        return []

    messages = build_merge_messages(patches)
    base["request_sha256"] = _messages_hash(messages)
    try:
        merged, calls, conversation, final_response = await call_patch_with_official_repair(
            client=client,
            generation=generation,
            messages=messages,
            max_continuations=settings.max_continuations,
            max_format_fix_rounds=settings.format_fix_rounds,
        )
    except PatchCallError as exc:
        # Official reduce semantics retain this group's original patches so a
        # local merge failure does not cancel the hierarchy.
        artifact = {
            **base,
            "status": "fallback",
            "fallback": "retain_input_patches",
            "calls": exc.calls,
            "conversation": exc.conversation,
            "final_response": exc.final_response,
            "error": str(exc),
            "output_patches": [patch.model_dump(mode="json") for patch in patches],
        }
        _atomic_json(artifact_path, artifact)
        return list(patches)

    artifact = {
        **base,
        "status": "merged",
        "calls": calls,
        "conversation": conversation,
        "final_response": final_response,
        "output_patches": [patch.model_dump(mode="json") for patch in merged],
    }
    _atomic_json(artifact_path, artifact)
    return merged


async def reduce_patches_hierarchically(
    *,
    run_dir: Path,
    source_ordered_patches: Sequence[SkillPatch],
    client: ChatLLMClient,
    generation: GenerationSettings,
    settings: Trace2SkillRunSettings,
) -> tuple[SkillPatch | None, dict[str, Any]]:
    """Run the upstream fallback-preserving REDUCE with the required B32 topology.

    As upstream does, parse/request failures are absent from the input patch list.
    The experiment's explicit B32 adaptation uses fixed chunks instead of upstream's
    small-remainder rebalancing. Subsequent levels retain upstream reduction semantics.
    """

    surviving = list(source_ordered_patches)
    if not surviving:
        summary = {
            "levels": [],
            "forced_merge": False,
            "ultimate_first_patch_fallback": False,
            "result_patch_count": 0,
        }
        _atomic_json(run_dir / "reduce_summary.json", summary)
        return None, summary
    if len(surviving) == 1:
        summary = {
            "levels": [],
            "single_patch_passthrough": True,
            "forced_merge": False,
            "ultimate_first_patch_fallback": False,
            "result_patch_count": 1,
        }
        _atomic_json(run_dir / "reduce_summary.json", summary)
        return surviving[0], summary

    level_root = run_dir / "merge_levels"
    level_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(settings.merge_concurrency)
    level_stats: list[dict[str, Any]] = []

    async def merge_one(
        *,
        level: int,
        group_index: int,
        group: list[SkillPatch],
    ) -> tuple[int, list[SkillPatch]]:
        async with semaphore:
            output = await merge_patch_group(
                artifact_path=(
                    level_root / f"level_{level:02d}" / f"group_{group_index + 1:04d}.json"
                ),
                level=level,
                group_index=group_index,
                patches=group,
                client=client,
                generation=generation,
                settings=settings,
            )
            return group_index, output

    first_inputs = fixed_chunks(surviving, settings.merge_batch_size)
    first_tasks = [
        asyncio.create_task(merge_one(level=1, group_index=index, group=group))
        for index, group in enumerate(first_inputs)
    ]
    first_results = await asyncio.gather(*first_tasks)
    current: list[SkillPatch] = []
    for _, patches in sorted(first_results):
        current.extend(patches)
    level_stats.append(
        _merge_level_stats(
            level_root / "level_01",
            level=1,
            input_patch_count=sum(len(group) for group in first_inputs),
            output_patch_count=len(current),
            source_group_count=len(first_inputs),
        )
    )

    level = 1
    while len(current) > 1 and level < settings.max_merge_levels:
        level += 1
        groups = balanced_chunks_upstream(current, settings.merge_batch_size)
        tasks = [
            asyncio.create_task(merge_one(level=level, group_index=index, group=group))
            for index, group in enumerate(groups)
        ]
        results = await asyncio.gather(*tasks)
        next_level: list[SkillPatch] = []
        for _, patches in sorted(results):
            next_level.extend(patches)
        level_stats.append(
            _merge_level_stats(
                level_root / f"level_{level:02d}",
                level=level,
                input_patch_count=len(current),
                output_patch_count=len(next_level),
                source_group_count=len(groups),
            )
        )
        current = next_level

    forced_merge = False
    ultimate_fallback = False
    if len(current) > 1:
        forced_merge = True
        forced_level = level + 1
        forced_path = level_root / f"level_{forced_level:02d}" / "group_0001.json"
        forced = await merge_patch_group(
            artifact_path=forced_path,
            level=forced_level,
            group_index=0,
            patches=current,
            client=client,
            generation=generation,
            settings=settings,
        )
        forced_artifact = _read_json(forced_path)
        if forced_artifact.get("status") == "fallback":
            # This is the upstream ultimate fallback after a failed forced merge.
            current = current[:1]
            ultimate_fallback = True
        else:
            # Upstream coalesces every patch returned by the final forced call
            # before returning the result.
            current = [coalesce_patches(forced)] if forced else []
        level_stats.append(
            _merge_level_stats(
                forced_path.parent,
                level=forced_level,
                input_patch_count=int(forced_artifact.get("input_patch_count", 0)),
                output_patch_count=len(current),
                source_group_count=1,
            )
        )

    summary = {
        "levels": level_stats,
        "forced_merge": forced_merge,
        "ultimate_first_patch_fallback": ultimate_fallback,
        "result_patch_count": len(current),
    }
    _atomic_json(run_dir / "reduce_summary.json", summary)
    return (current[0] if current else None), summary


def _merge_level_stats(
    level_dir: Path,
    *,
    level: int,
    input_patch_count: int,
    output_patch_count: int,
    source_group_count: int,
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for path in sorted(level_dir.glob("group_*.json")):
        status = str(_read_json(path).get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "level": level,
        "group_count": source_group_count,
        "input_patch_count": input_patch_count,
        "output_patch_count": output_patch_count,
        "statuses": statuses,
    }


def sanitize_translated_edits(
    edits: Sequence[PatchEdit],
) -> tuple[list[PatchEdit], list[dict[str, Any]]]:
    """Apply upstream op/path sanitization to the task's single-file skill layout."""

    kept: list[PatchEdit] = []
    dropped: list[dict[str, Any]] = []
    for index, source_edit in enumerate(edits):
        edit = source_edit.model_copy(deep=True)
        edit.op = PATCH_OP_ALIASES.get(edit.op, edit.op)
        normalized_file = edit.file.replace("\\", "/").lstrip("./")
        reason: str | None = None
        if edit.op not in SUPPORTED_PATCH_OPS:
            reason = "unsupported_operation"
        elif normalized_file != "SKILL.md":
            reason = "outside_single_file_skill"
        elif REFERENCE_PATH_PATTERN.search(edit.content):
            reason = "unsupported_reference_link"
        if reason is not None:
            dropped.append(
                {
                    "edit_index": index,
                    "reason": reason,
                    "edit": source_edit.model_dump(mode="json"),
                }
            )
            continue
        edit.file = "SKILL.md"
        kept.append(edit)
    return kept, dropped


async def translate_final_patch(
    *,
    run_dir: Path,
    patch: SkillPatch,
    client: ChatLLMClient,
    generation: GenerationSettings,
    settings: Trace2SkillRunSettings,
) -> tuple[SkillPatch, dict[str, Any]]:
    """Translate each non-file-level edit independently, as upstream does."""

    translation_dir = run_dir / "translation"
    translation_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(settings.merge_concurrency)

    async def translate_one(index: int, edit: PatchEdit) -> tuple[int, PatchEdit, str]:
        path = translation_dir / f"edit_{index + 1:04d}.json"
        input_hash = _hash_text(_canonical_json(edit.model_dump(mode="json")))
        if path.exists():
            artifact = _read_json(path)
            if (
                artifact.get("schema_version") != BUILD_SCHEMA_VERSION
                or artifact.get("artifact_type") != TRANSLATION_ARTIFACT_TYPE
                or artifact.get("edit_index") != index
                or artifact.get("input_edit_sha256") != input_hash
                or artifact.get("initial_skill_sha256") != _hash_text("")
                or artifact.get("status") not in {"translated", "fallback", "passthrough"}
            ):
                raise Trace2SkillError(f"incompatible translation artifact: {path}")
            return (
                index,
                PatchEdit.model_validate(artifact["output_edit"]),
                cast(str, artifact["status"]),
            )

        base = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "artifact_type": TRANSLATION_ARTIFACT_TYPE,
            "edit_index": index,
            "input_edit_sha256": input_hash,
            "initial_skill_sha256": _hash_text(""),
            "input_edit": edit.model_dump(mode="json"),
        }
        normalized_op = PATCH_OP_ALIASES.get(edit.op, edit.op)
        if normalized_op in {"create", "delete_file"}:
            artifact = {
                **base,
                "status": "passthrough",
                "calls": [],
                "conversation": [],
                "final_response": None,
                "output_edit": edit.model_dump(mode="json"),
            }
            _atomic_json(path, artifact)
            return index, edit, "passthrough"

        messages = build_translation_messages("", edit)
        base.update(
            {
                "system_prompt_sha256": _hash_text(messages[0]["content"]),
                "request_sha256": _messages_hash(messages),
            }
        )
        async with semaphore:
            try:
                (
                    patches,
                    calls,
                    conversation,
                    final_response,
                ) = await call_patch_with_official_repair(
                    client=client,
                    generation=generation,
                    messages=messages,
                    max_continuations=settings.max_continuations,
                    max_format_fix_rounds=settings.format_fix_rounds,
                )
            except PatchCallError as exc:
                artifact = {
                    **base,
                    "status": "fallback",
                    "fallback": "use_original_edit",
                    "calls": exc.calls,
                    "conversation": exc.conversation,
                    "final_response": exc.final_response,
                    "error": str(exc),
                    "output_edit": edit.model_dump(mode="json"),
                }
                _atomic_json(path, artifact)
                return index, edit, "fallback"

        translated = [candidate for patch_item in patches for candidate in patch_item.edits]
        matching = [candidate for candidate in translated if candidate.file == edit.file]
        if matching:
            output = matching[0]
            status = "translated"
            fallback = None
        else:
            output = edit
            status = "fallback"
            fallback = "no_matching_edit_for_source_file"
        artifact = {
            **base,
            "status": status,
            "fallback": fallback,
            "calls": calls,
            "conversation": conversation,
            "final_response": final_response,
            "parsed_patch_count": len(patches),
            "output_edit": output.model_dump(mode="json"),
        }
        _atomic_json(path, artifact)
        return index, output, status

    translated_results = await asyncio.gather(
        *(translate_one(index, edit) for index, edit in enumerate(patch.edits))
    )
    translated_edits = [item[1] for item in sorted(translated_results)]
    sanitized, dropped = sanitize_translated_edits(translated_edits)
    translated_patch = SkillPatch(
        reasoning=patch.reasoning,
        edits=sanitized,
        changelog_entries=patch.changelog_entries,
    )
    summary = {
        "input_edit_count": len(patch.edits),
        "translated_count": sum(item[2] == "translated" for item in translated_results),
        "fallback_count": sum(item[2] == "fallback" for item in translated_results),
        "passthrough_count": sum(item[2] == "passthrough" for item in translated_results),
        "sanitized_edit_count": len(sanitized),
        "dropped_edits": dropped,
    }
    _atomic_json(run_dir / "translation_summary.json", summary)
    _atomic_json(
        run_dir / "translated_final_patch.json",
        {
            "schema_version": BUILD_SCHEMA_VERSION,
            "artifact_type": "trace2skill_translated_final_patch",
            "source_patch_sha256": _patches_hash([patch]),
            "patch": translated_patch.model_dump(mode="json"),
            "summary": summary,
        },
    )
    return translated_patch, summary


def _find_section_bounds(lines: list[str], section_header: str) -> tuple[int, int] | None:
    """Port of the official Markdown section locator."""

    target = section_header.strip()
    level = len(target) - len(target.lstrip("#"))
    for index, line in enumerate(lines):
        if line.strip() != target:
            continue
        end = len(lines)
        for following in range(index + 1, len(lines)):
            if lines[following].startswith("#"):
                following_level = len(lines[following]) - len(lines[following].lstrip("#"))
                if following_level <= level:
                    end = following
                    break
        return index, end
    return None


def apply_patch_edit_to_content(content: str, edit: PatchEdit) -> str:
    """Port of upstream's deterministic single-edit applicator."""

    lines = content.split("\n")
    if edit.op == "append_to_section":
        bounds = _find_section_bounds(lines, edit.target_section)
        if bounds is None:
            return content
        _, end = bounds
        insert_at = end
        while insert_at > bounds[0] + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines = lines[:insert_at] + [""] + edit.content.split("\n") + lines[insert_at:]
        return "\n".join(lines)

    if edit.op == "replace_in_section":
        if edit.target_section:
            bounds = _find_section_bounds(lines, edit.target_section)
            if bounds is None:
                return content
            start, end = bounds
            section = "\n".join(lines[start:end])
            if edit.old_text not in section:
                return content
            replacement = section.replace(edit.old_text, edit.content, 1)
            return "\n".join(lines[:start] + replacement.split("\n") + lines[end:])
        if edit.old_text not in content:
            return content
        return content.replace(edit.old_text, edit.content, 1)

    if edit.op in {"insert_after", "insert_before"}:
        if edit.target_section:
            bounds = _find_section_bounds(lines, edit.target_section)
            if bounds is None:
                return content
            start, end = bounds
            search_text = "\n".join(lines[start:end])
            offset, section_length = start, end - start
        else:
            search_text = "\n".join(lines)
            offset, section_length = 0, len(lines)
        if edit.target_text not in search_text:
            return content
        if edit.op == "insert_after":
            replacement = edit.target_text + "\n" + edit.content
        else:
            replacement = edit.content + "\n" + edit.target_text
        changed = search_text.replace(edit.target_text, replacement, 1)
        return "\n".join(lines[:offset] + changed.split("\n") + lines[offset + section_length :])

    if edit.op == "add_section":
        if edit.after_section:
            bounds = _find_section_bounds(lines, edit.after_section)
            insert_at = bounds[1] if bounds is not None else len(lines)
        else:
            insert_at = len(lines)
        header = edit.target_section or "## New Section"
        new_section = ["", header, ""] + edit.content.split("\n")
        return "\n".join(lines[:insert_at] + new_section + lines[insert_at:])

    if edit.op == "delete_section":
        bounds = _find_section_bounds(lines, edit.target_section)
        if bounds is None:
            return content
        start, end = bounds
        while start > 0 and not lines[start - 1].strip():
            start -= 1
        return "\n".join(lines[:start] + lines[end:])
    return content


def apply_patch_programmatically(
    skill_content: str,
    patch: SkillPatch,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply patch edits in order with zero LLM calls, matching upstream APPLY."""

    current = skill_content
    records: list[dict[str, Any]] = []
    for index, edit in enumerate(patch.edits):
        updated = apply_patch_edit_to_content(current, edit)
        applied = updated != current
        records.append(
            {
                "edit_index": index,
                "status": "applied" if applied else "skipped_no_effect",
                "edit": edit.model_dump(mode="json"),
            }
        )
        current = updated
    return current, records


def validate_runtime_skill(content: str, *, max_lines: int) -> tuple[bool, str]:
    """Task-layout counterpart of upstream quick_validate.py."""

    if not content.strip():
        return False, "SKILL.md is empty"
    line_count = len(content.strip().splitlines())
    if line_count > max_lines:
        return False, f"SKILL.md has {line_count} lines; maximum is {max_lines}"
    return True, "Skill is valid"


async def verify_and_fix_skill(
    *,
    run_dir: Path,
    skill_content: str,
    client: ChatLLMClient,
    generation: GenerationSettings,
    settings: Trace2SkillRunSettings,
) -> tuple[str, list[dict[str, Any]], bool, str]:
    """Run the official validate -> patch -> apply loop for up to three rounds."""

    verification_dir = run_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    current = skill_content
    round_records: list[dict[str, Any]] = []
    for round_number in range(1, settings.max_verification_rounds + 1):
        valid, message = validate_runtime_skill(current, max_lines=settings.max_skill_lines)
        if valid:
            return current, round_records, True, message

        path = verification_dir / f"round_{round_number:02d}.json"
        source_hash = _hash_text(current)
        if path.exists():
            artifact = _read_json(path)
            if (
                artifact.get("schema_version") != BUILD_SCHEMA_VERSION
                or artifact.get("artifact_type") != "trace2skill_verification_round"
                or artifact.get("round") != round_number
                or artifact.get("input_skill_sha256") != source_hash
                or artifact.get("validation_error") != message
                or artifact.get("status") not in {"applied", "stopped"}
            ):
                raise Trace2SkillError(f"incompatible verification artifact: {path}")
            round_records.append(artifact)
            if artifact["status"] == "stopped":
                break
            current = cast(str, artifact["output_skill"])
            continue

        messages = build_verification_messages(current, message)
        base = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "artifact_type": "trace2skill_verification_round",
            "round": round_number,
            "input_skill_sha256": source_hash,
            "validation_error": message,
            "system_prompt_sha256": _hash_text(messages[0]["content"]),
            "request_sha256": _messages_hash(messages),
        }
        try:
            patches, calls, conversation, final_response = await call_patch_with_official_repair(
                client=client,
                generation=generation,
                messages=messages,
                max_continuations=settings.max_continuations,
                max_format_fix_rounds=settings.format_fix_rounds,
            )
        except PatchCallError as exc:
            artifact = {
                **base,
                "status": "stopped",
                "stop_reason": "verification_patch_unavailable",
                "calls": exc.calls,
                "conversation": exc.conversation,
                "final_response": exc.final_response,
                "error": str(exc),
            }
            _atomic_json(path, artifact)
            round_records.append(artifact)
            break

        patch = coalesce_patches(patches)
        sanitized, dropped = sanitize_translated_edits(patch.edits)
        sanitized_patch = SkillPatch(
            reasoning=patch.reasoning,
            edits=sanitized,
            changelog_entries=patch.changelog_entries,
        )
        updated, application = apply_patch_programmatically(current, sanitized_patch)
        if updated == current:
            artifact = {
                **base,
                "status": "stopped",
                "stop_reason": "verification_patch_made_no_effective_change",
                "calls": calls,
                "conversation": conversation,
                "final_response": final_response,
                "patch": sanitized_patch.model_dump(mode="json"),
                "dropped_edits": dropped,
                "application": application,
            }
            _atomic_json(path, artifact)
            round_records.append(artifact)
            break

        artifact = {
            **base,
            "status": "applied",
            "calls": calls,
            "conversation": conversation,
            "final_response": final_response,
            "patch": sanitized_patch.model_dump(mode="json"),
            "dropped_edits": dropped,
            "application": application,
            "output_skill": updated,
            "output_skill_sha256": _hash_text(updated),
        }
        _atomic_json(path, artifact)
        round_records.append(artifact)
        current = updated

    valid, message = validate_runtime_skill(current, max_lines=settings.max_skill_lines)
    return current, round_records, valid, message


def _skill_diff(original: str, updated: str) -> str:
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="a/SKILL.md",
            tofile="b/SKILL.md",
        )
    )


def _prepare_aligned_build_directory(
    *,
    output_root: Path,
    resume_dir: Path | None,
    trace_config: Trace2SkillConfig,
    corpus: Trace2SkillCorpus,
    profile: ModelProfile,
    profile_path: Path,
    run_contract: Trace2SkillRunContract,
    selected_hashes: list[str],
) -> tuple[Path, str]:
    settings = trace_config.run
    profile_source = profile_path.expanduser().resolve()
    source_corpus = {
        "collection_dir": str(corpus.collection_dir),
        "collection_manifest_sha256": _sha256_file(corpus.collection_dir / "manifest.json"),
        "collection_fingerprint": corpus.manifest["immutable_fingerprint"],
        "corpus_seal_sha256": _sha256_file(corpus.collection_dir / "corpus.json"),
        "corpus_sha256": corpus.corpus_sha256,
        "corpus_sample_count": len(corpus.trajectories),
        "selected_trajectory_count": run_contract.selected_sample_count,
        "selected_prefix_sha256": _hash_text(_canonical_json(selected_hashes)),
        "selection": run_contract.selection,
        "assistant_behavior_sha256": corpus.seal["assistant_behavior_sha256"],
    }
    method_settings = {
        "method": settings.method,
        "initial_skill": settings.initial_skill,
        "information_boundary": settings.information_boundary,
        "analysis_mode": settings.analysis_mode,
        "map_batch_size": settings.map_batch_size,
        "merge_batch_size": settings.merge_batch_size,
        "max_continuations": settings.max_continuations,
        "format_fix_rounds": settings.format_fix_rounds,
        "max_merge_levels": settings.max_merge_levels,
        "max_verification_rounds": settings.max_verification_rounds,
        "max_skill_lines": settings.max_skill_lines,
    }
    prompt_hashes = {
        "success_analysis": _hash_text(SUCCESS_ANALYSIS_SYSTEM_PROMPT),
        "failure_analysis": _hash_text(FAILURE_ANALYSIS_SYSTEM_PROMPT),
        "analysis_user": _hash_text(ANALYSIS_USER_TEMPLATE),
        "combined_map": _hash_text(MAP_SYSTEM_PROMPT),
        "combined_merge": _hash_text(MERGE_SYSTEM_PROMPT),
        "translation": _hash_text(TRANSLATION_SYSTEM_PROMPT),
        "verification": _hash_text(VERIFICATION_SYSTEM_PROMPT),
        "continuation": _hash_text(CONTINUE_JSON_PROMPT),
        "format_fix": _hash_text(JSON_FORMAT_FIX_TEMPLATE),
    }
    manifest_basis = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "artifact_type": BUILD_ARTIFACT_TYPE,
        "pipeline": BUILD_PIPELINE,
        "method": settings.method,
        "official_implementation": {
            "repository": "Qwen-Applications/Trace2Skill",
            "commit": OFFICIAL_TRACE2SKILL_COMMIT,
            "parallel_patch_pipeline": "json",
        },
        "implementation_sha256": _sha256_file(Path(__file__)),
        "initial_skill": {"label": NO_SKILL, "content": "", "sha256": _hash_text("")},
        "run_contract": run_contract.to_dict(),
        "source_corpus": source_corpus,
        "development_dataset": corpus.manifest["development_dataset"],
        "assistant_model_profile": {
            "source_path": str(profile_source),
            "source_sha256": _sha256_file(profile_source),
            "settings": profile.model_dump(mode="json"),
            "behavior": _profile_behavior_snapshot(profile),
            "behavior_sha256": _profile_behavior_sha256(profile),
        },
        "profile_skill_binding": {
            "framework": profile.skill.framework if profile.skill is not None else None,
            "path": str(profile.skill.path) if profile.skill is not None else None,
            "publish_policy": "canonical_complete_builds_only",
        },
        "method_settings": method_settings,
        "prompt_sha256": prompt_hashes,
        "git_commit": read_git_commit(),
    }
    fingerprint = _hash_text(_canonical_json(manifest_basis))
    if resume_dir is not None:
        run_dir = resume_dir.expanduser().resolve()
        manifest = _read_json(run_dir / "manifest.json")
        if (
            manifest.get("artifact_type") != BUILD_ARTIFACT_TYPE
            or manifest.get("schema_version") != BUILD_SCHEMA_VERSION
            or manifest.get("pipeline") != BUILD_PIPELINE
        ):
            raise ValueError(
                "resume directory predates the official-aligned Trace2Skill build; "
                "start a new build while reusing the same sealed collection"
            )
        if manifest.get("immutable_fingerprint") != fingerprint:
            raise ValueError("resume directory was created with different immutable build inputs")
        return run_dir, cast(str, manifest["created_at"])

    run_prefix = (
        "trace2skill_build"
        if run_contract.canonical
        else f"trace2skill_build_trial_n{run_contract.selected_sample_count}"
    )
    run_id = run_prefix + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id += "_" + uuid.uuid4().hex[:8]
    run_dir = output_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(UTC).isoformat()
    manifest = {
        **manifest_basis,
        "run_id": run_id,
        "created_at": created_at,
        "immutable_fingerprint": fingerprint,
        "official_alignment": {
            "analysis": "single_call_report_then_official_memory_item_parser",
            "map": "one_parsed_analysis_record_per_independent_json_patch_call",
            "map_structural_sanitization": (
                "official create/link pairing adapted to the single-file runtime"
            ),
            "combined_record_order": (
                "error_records_then_success_records; lexicographic sample id within each source"
            ),
            "response_repair": "up_to_two_continuations_then_two_targeted_format_fixes",
            "reduce": "hierarchical_merge_with_group_fallback_and_forced_final_merge",
            "translation": "one_independent_call_per_non_file_level_edit",
            "apply": "deterministic_programmatic_patch_application",
            "verification": "validate_and_patch_for_up_to_three_rounds",
            "parse_or_request_failures": "exclude_failed_item_and_continue",
        },
        "necessary_task_adaptations": {
            "trajectory_evidence": (
                "visible user/assistant dialogue plus binary completion outcome only"
            ),
            "initial_skill": (
                "empty because collection and peer baselines run without a prompted base"
            ),
            "failure_label": "not naturally completed within twenty interaction turns",
            "skill_layout": (
                "one raw Markdown SKILL.md used verbatim as the only system message; "
                "no references or Agent-Skills frontmatter"
            ),
            "merge_batch_size": (
                "fixed groups of thirty-two surviving MAP patches, yielding thirty-two "
                "first-level groups when all one thousand source records produce patches, "
                "as fixed by the requested baseline protocol"
            ),
            "model_binding": "one authoring/rollout/test profile and skill path per model",
        },
        "method_neutral_wrappers": {
            "separate_collect_and_build": True,
            "immutable_collection_input": True,
            "resume_from_committed_stage_artifacts": True,
            "limit_is_noncanonical_audit_trial": True,
            "artifact_retention": "all prompts, responses, parses, patches, and fallbacks",
        },
        "visibility_contract": {
            "analysis": ["ordered_visible_dialogue", "analyst_role_selected_by_binary_outcome"],
            "map": ["empty_initial_skill", "one_parsed_analysis_record"],
            "merge": ["empty_initial_skill", "patches_only"],
            "raw_episode_directories_read": False,
            "hidden_dataset_or_simulator_fields": False,
        },
        "merge_topology": {
            "planned_source_slots": run_contract.selected_sample_count,
            "map_batch_size": settings.map_batch_size,
            "level1_group_size": settings.merge_batch_size,
            "level1_group_count_if_all_map_succeed": (
                run_contract.selected_sample_count + settings.merge_batch_size - 1
            )
            // settings.merge_batch_size,
            "later_group_size": settings.merge_batch_size,
            "later_grouping_algorithm": "official_balanced_chunk_list",
            "max_merge_levels_before_forced_merge": settings.max_merge_levels,
            "frozen_initial_skill_at_every_merge": True,
        },
        "execution_settings": {
            "analysis_concurrency": settings.trajectory_concurrency,
            "merge_and_translation_concurrency": settings.merge_concurrency,
        },
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    initial_dir = run_dir / "initial_skill"
    initial_dir.mkdir()
    _atomic_text(initial_dir / "SKILL.md", "")
    if not run_contract.canonical:
        _atomic_text(
            run_dir / "SMOKE_ONLY.md",
            "# Non-canonical Trace2Skill build trial\n\n"
            f"This build used the first {run_contract.selected_sample_count} trajectories "
            "from the sealed development corpus.\n\n"
            "It exists only to audit the complete build pipeline. It must not be published "
            "or used for test-set evaluation.\n",
        )
    return run_dir, created_at


async def _materialize_skill(
    *,
    run_dir: Path,
    patch: SkillPatch,
    client: ChatLLMClient,
    generation: GenerationSettings,
    settings: Trace2SkillRunSettings,
    run_contract: Trace2SkillRunContract,
    profile_path: Path,
) -> tuple[Path | None, dict[str, Any]]:
    apply_path = run_dir / "apply.json"
    patch_hash = _patches_hash([patch])
    skill_path = run_dir / ("SKILL.md" if run_contract.canonical else "SMOKE_SKILL.md")
    if apply_path.exists():
        artifact = _read_json(apply_path)
        if (
            artifact.get("schema_version") != BUILD_SCHEMA_VERSION
            or artifact.get("artifact_type") != "trace2skill_programmatic_apply"
            or artifact.get("source_patch_sha256") != patch_hash
            or artifact.get("status") not in {"valid", "invalid"}
            or not isinstance(artifact.get("skill_content"), str)
        ):
            raise Trace2SkillError(f"incompatible APPLY artifact: {apply_path}")
        if artifact["status"] == "valid":
            _atomic_text(skill_path, cast(str, artifact["skill_content"]))
            if run_contract.evaluation_allowed:
                _write_assistant_override(
                    run_dir,
                    skill_path=skill_path,
                    profile_path=str(profile_path),
                )
            return skill_path, artifact
        return None, artifact

    applied, application = apply_patch_programmatically("", patch)
    candidate, verification, valid, validation_message = await verify_and_fix_skill(
        run_dir=run_dir,
        skill_content=applied,
        client=client,
        generation=generation,
        settings=settings,
    )
    committed = candidate.strip() + "\n" if candidate.strip() else ""
    _atomic_text(run_dir / "candidate_skill.md", committed)
    diff = _skill_diff("", committed)
    if diff:
        _atomic_text(run_dir / "applied_diffs.patch", diff)
    artifact = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "artifact_type": "trace2skill_programmatic_apply",
        "source_patch_sha256": patch_hash,
        "status": "valid" if valid else "invalid",
        "application": application,
        "applied_edit_count": sum(item["status"] == "applied" for item in application),
        "skipped_edit_count": sum(item["status"] == "skipped_no_effect" for item in application),
        "verification_round_count": len(verification),
        "verification": [
            {
                "round": item.get("round"),
                "status": item.get("status"),
                "stop_reason": item.get("stop_reason"),
                "output_skill_sha256": item.get("output_skill_sha256"),
            }
            for item in verification
        ],
        "validation_message": validation_message,
        "skill_content": committed,
        "skill_sha256": _hash_text(committed),
        "skill_line_count": len(committed.strip().splitlines()) if committed.strip() else 0,
        "run_mode": run_contract.run_mode,
        "canonical": run_contract.canonical,
        "evaluation_allowed": run_contract.evaluation_allowed,
    }
    _atomic_json(apply_path, artifact)
    if not valid:
        return None, artifact
    _atomic_text(skill_path, committed)
    if run_contract.evaluation_allowed:
        _write_assistant_override(run_dir, skill_path=skill_path, profile_path=str(profile_path))
    return skill_path, artifact


def _artifact_call_count(paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        artifact = _read_json(path)
        calls = artifact.get("calls")
        if isinstance(calls, list):
            count += len(calls)
    return count


def _write_aligned_build_summary(
    run_dir: Path,
    *,
    created_at: str,
    complete: bool,
    corpus: Trace2SkillCorpus,
    run_contract: Trace2SkillRunContract,
    analysis_slots: Sequence[dict[str, Any]],
    map_slots: Sequence[dict[str, Any]],
    stage_error: str | None,
    final_patch: SkillPatch | None,
    final_skill_path: Path | None,
    published_skill_path: Path | None,
) -> None:
    analysis_paths = sorted((run_dir / "trajectory_analysis").glob("*.json"))
    map_paths = sorted((run_dir / "map_patches").glob("*.json"))
    merge_paths = sorted((run_dir / "merge_levels").glob("level_*/group_*.json"))
    translation_paths = sorted((run_dir / "translation").glob("edit_*.json"))
    verification_paths = sorted((run_dir / "verification").glob("round_*.json"))
    all_llm_paths = [
        *analysis_paths,
        *map_paths,
        *merge_paths,
        *translation_paths,
        *verification_paths,
    ]
    analysis_complete = sum(item.get("status") == "complete" for item in analysis_slots)
    map_complete = sum(item.get("status") == "complete" for item in map_slots)

    def count_skip_reasons(slots: Sequence[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in slots:
            if item.get("status") != "skipped":
                continue
            reason = str(item.get("skip_reason", "unspecified"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    final_patch_value = final_patch.model_dump(mode="json") if final_patch is not None else None
    summary = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "artifact_type": BUILD_ARTIFACT_TYPE,
        "pipeline": BUILD_PIPELINE,
        "stage": "build",
        "method": "Trace2Skill-Visible-Combined-Parallel-B32",
        "official_implementation_commit": OFFICIAL_TRACE2SKILL_COMMIT,
        "run_contract": run_contract.to_dict(),
        "source_corpus": {
            "collection_dir": str(corpus.collection_dir),
            "corpus_sha256": corpus.corpus_sha256,
            "sample_count": len(corpus.trajectories),
        },
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "complete" if complete else "incomplete",
        "selected_trajectory_count": run_contract.selected_sample_count,
        "valid_trajectory_count": run_contract.selected_sample_count,
        "analysis_record_count": analysis_complete,
        "analysis_excluded_count": len(analysis_slots) - analysis_complete,
        "map_patch_count": map_complete,
        "map_excluded_count": len(map_slots) - map_complete,
        "map_empty_edit_patch_count": sum(
            item.get("status") == "complete" and not item.get("patch", {}).get("edits")
            for item in map_slots
        ),
        "analysis_skip_reasons": count_skip_reasons(analysis_slots),
        "map_skip_reasons": count_skip_reasons(map_slots),
        "stage_error": stage_error,
        "logical_llm_call_count": _artifact_call_count(all_llm_paths),
        "interaction_call_count": 0,
        "artifact_inventory": {
            "manifest": (run_dir / "manifest.json").exists(),
            "source_corpus_verified": True,
            "analysis_artifact_count": len(analysis_paths),
            "map_artifact_count": len(map_paths),
            "merge_artifact_count": len(merge_paths),
            "translation_artifact_count": len(translation_paths),
            "verification_artifact_count": len(verification_paths),
            "final_patch": (run_dir / "final_patch.json").exists(),
            "translated_final_patch": (run_dir / "translated_final_patch.json").exists(),
            "apply": (run_dir / "apply.json").exists(),
            "candidate_skill": (run_dir / "candidate_skill.md").exists(),
            "run_skill": final_skill_path is not None and final_skill_path.exists(),
            "published_profile_skill": (
                published_skill_path is not None and published_skill_path.exists()
            ),
        },
        "final_patch": final_patch_value,
        "final_patch_sha256": (_patches_hash([final_patch]) if final_patch is not None else None),
        "final_skill_path": str(final_skill_path) if final_skill_path is not None else None,
        "final_skill_sha256": (
            _sha256_file(final_skill_path) if final_skill_path is not None else None
        ),
        "published_skill_path": (
            str(published_skill_path) if published_skill_path is not None else None
        ),
        "test_time_policy": (
            {
                "baseline": "trace2skill",
                "memory": None,
                "retrieval": False,
                "development_access": False,
                "updates": False,
            }
            if run_contract.evaluation_allowed and complete
            else None
        ),
        "intended_use": (
            "formal_baseline_evaluation"
            if run_contract.evaluation_allowed and complete
            else "pipeline_trial_only"
            if not run_contract.canonical
            else "incomplete_build"
        ),
    }
    _atomic_json(run_dir / "summary.json", summary)


async def run_trace2skill_build(
    *,
    collection_dir: str | Path,
    output_root: str | Path,
    trace_config: Trace2SkillConfig,
    profile_path: str | Path,
    assistant_environment: AssistantEnvironmentSettings | None = None,
    analyst_client: ChatLLMClient | None = None,
    resume_dir: str | Path | None = None,
    progress_callback: BuildProgressCallback | None = None,
    limit: int | None = None,
) -> Trace2SkillEvolutionResult:
    """Build with the official analysis -> MAP -> REDUCE -> APPLY workflow."""

    settings = trace_config.run
    run_contract = _resolve_run_contract(
        expected_sample_count=trace_config.dataset.expected_sample_count,
        limit=limit,
    )
    corpus = load_trace2skill_corpus(collection_dir)
    resolved_profile_path = Path(profile_path).expanduser().resolve()
    profile = load_model_profile(str(resolved_profile_path))
    _validate_build_profile(profile, corpus)
    selected_count = run_contract.selected_sample_count
    selected_trajectories = list(corpus.trajectories[:selected_count])
    selected_hashes = list(corpus.trajectory_sha256[:selected_count])
    run_dir, created_at = _prepare_aligned_build_directory(
        output_root=Path(output_root),
        resume_dir=Path(resume_dir) if resume_dir is not None else None,
        trace_config=trace_config,
        corpus=corpus,
        profile=profile,
        profile_path=resolved_profile_path,
        run_contract=run_contract,
        selected_hashes=selected_hashes,
    )
    client = analyst_client or OpenAICompatibleChatClient(profile, assistant_environment)
    generation = profile.generation["assistant"]
    analysis_dir = run_dir / "trajectory_analysis"
    map_dir = run_dir / "map_patches"
    analysis_dir.mkdir(exist_ok=True)
    map_dir.mkdir(exist_ok=True)
    analysis_slots: list[dict[str, Any]] = []
    map_slots: list[dict[str, Any]] = []
    final_patch: SkillPatch | None = None
    final_skill_path: Path | None = None
    published_skill_path: Path | None = None

    try:
        semaphore = asyncio.Semaphore(settings.trajectory_concurrency)

        async def analyze(index: int) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                slot = corpus.slots[index]
                artifact = await process_analysis_slot(
                    artifact_path=analysis_dir / f"{index + 1:04d}.json",
                    dataset_index=index,
                    sample_id=cast(str, slot["sample_id"]),
                    source_trajectory_sha256=selected_hashes[index],
                    trajectory=selected_trajectories[index],
                    client=client,
                    generation=generation,
                )
                return index, artifact

        analysis_results = await asyncio.gather(
            *(analyze(index) for index in range(selected_count))
        )
        analysis_slots = [item for _, item in sorted(analysis_results)]

        async def map_one(index: int) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                slot = corpus.slots[index]
                artifact = await process_map_slot(
                    artifact_path=map_dir / f"{index + 1:04d}.json",
                    dataset_index=index,
                    sample_id=cast(str, slot["sample_id"]),
                    source_analysis=analysis_slots[index],
                    client=client,
                    generation=generation,
                    settings=settings,
                )
                return index, artifact

        map_tasks = [asyncio.create_task(map_one(index)) for index in range(selected_count)]
        ordered_map: list[dict[str, Any] | None] = [None] * selected_count
        try:
            for finished, task in enumerate(asyncio.as_completed(map_tasks), 1):
                index, artifact = await task
                ordered_map[index] = artifact
                if progress_callback is not None:
                    progress_callback(finished, selected_count, artifact)
        except Exception:
            for task in map_tasks:
                task.cancel()
            await asyncio.gather(*map_tasks, return_exceptions=True)
            raise
        map_slots = [cast(dict[str, Any], item) for item in ordered_map]
        # The official combined runner concatenates parsed error records before
        # parsed success records. Failed MAP slots are omitted rather than kept as
        # positional placeholders.
        source_ordered_patches, ordered_map_indices = order_combined_map_patches(
            map_slots, selected_trajectories
        )
        source_ordered_patches, map_layout_drops = sanitize_map_patches_for_single_file(
            source_ordered_patches, ordered_map_indices
        )
        map_patch_count = len(source_ordered_patches)
        _atomic_json(
            run_dir / "map_summary.json",
            {
                "selected_trajectory_count": selected_count,
                "analysis_record_count": sum(
                    item.get("status") == "complete" for item in analysis_slots
                ),
                "analysis_excluded_count": sum(
                    item.get("status") == "skipped" for item in analysis_slots
                ),
                "map_patch_count": map_patch_count,
                "map_excluded_count": sum(item.get("status") == "skipped" for item in map_slots),
                "reduce_record_order": (
                    "error_records_then_success_records; lexicographic sample id within each source"
                ),
                "reduce_dataset_indices": ordered_map_indices,
                "single_file_layout_dropped_edit_count": sum(
                    len(record["dropped_edits"]) for record in map_layout_drops
                ),
                "single_file_layout_drops": map_layout_drops,
            },
        )
        if not map_patch_count:
            raise Trace2SkillError(
                "MAP produced no parseable patches; the official no-op result cannot be used "
                "as intentLLM's non-empty runtime system message"
            )

        final_patch, reduce_summary = await reduce_patches_hierarchically(
            run_dir=run_dir,
            source_ordered_patches=source_ordered_patches,
            client=client,
            generation=generation,
            settings=settings,
        )
        if final_patch is None:
            raise Trace2SkillError("REDUCE produced no final patch")
        _atomic_json(
            run_dir / "final_patch.json",
            {
                "schema_version": BUILD_SCHEMA_VERSION,
                "artifact_type": "trace2skill_final_patch",
                "source_map_patch_count": map_patch_count,
                "source_map_patches_sha256": _hash_text(
                    _canonical_json([patch.merger_view() for patch in source_ordered_patches])
                ),
                "reduce_summary": reduce_summary,
                "patch": final_patch.model_dump(mode="json"),
            },
        )
        translated_patch, _ = await translate_final_patch(
            run_dir=run_dir,
            patch=final_patch,
            client=client,
            generation=generation,
            settings=settings,
        )
        final_skill_path, apply_artifact = await _materialize_skill(
            run_dir=run_dir,
            patch=translated_patch,
            client=client,
            generation=generation,
            settings=settings,
            run_contract=run_contract,
            profile_path=resolved_profile_path,
        )
        if final_skill_path is None:
            raise Trace2SkillError(
                "final skill failed the task-adapted validator after official verification: "
                + str(apply_artifact.get("validation_message"))
            )
        published_skill_path = _publish_profile_bound_skill(
            run_dir=run_dir,
            run_skill_path=final_skill_path,
            profile=profile,
            profile_path=str(resolved_profile_path),
            dataset_info=None,
            run_contract=run_contract,
            development_dataset_sha256=cast(str, corpus.manifest["development_dataset"]["sha256"]),
            source_corpus=corpus,
        )
    except Trace2SkillError as exc:
        stage_error = f"{type(exc).__name__}: {exc}"
        _write_aligned_build_summary(
            run_dir,
            created_at=created_at,
            complete=False,
            corpus=corpus,
            run_contract=run_contract,
            analysis_slots=analysis_slots,
            map_slots=map_slots,
            stage_error=stage_error,
            final_patch=final_patch,
            final_skill_path=final_skill_path,
            published_skill_path=published_skill_path,
        )
        return Trace2SkillEvolutionResult(
            run_dir=run_dir,
            complete=False,
            valid_trajectory_count=selected_count,
            patch_count=sum(item.get("status") == "complete" for item in map_slots),
            final_skill_path=final_skill_path,
            published_skill_path=published_skill_path,
            run_mode=run_contract.run_mode,
            canonical=run_contract.canonical,
            skipped_count=sum(item.get("status") == "skipped" for item in map_slots),
            stage_error=stage_error,
        )

    _write_aligned_build_summary(
        run_dir,
        created_at=created_at,
        complete=True,
        corpus=corpus,
        run_contract=run_contract,
        analysis_slots=analysis_slots,
        map_slots=map_slots,
        stage_error=None,
        final_patch=final_patch,
        final_skill_path=final_skill_path,
        published_skill_path=published_skill_path,
    )
    return Trace2SkillEvolutionResult(
        run_dir=run_dir,
        complete=True,
        valid_trajectory_count=selected_count,
        patch_count=sum(item.get("status") == "complete" for item in map_slots),
        final_skill_path=final_skill_path,
        published_skill_path=published_skill_path,
        run_mode=run_contract.run_mode,
        canonical=run_contract.canonical,
        skipped_count=sum(item.get("status") == "skipped" for item in map_slots),
        stage_error=None,
    )
