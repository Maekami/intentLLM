from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from intent_metrics.config import load_aitr_config
from intent_metrics.models import ConversationMessage
from intent_metrics.prompt import AITRPromptTemplate, load_aitr_prompt


def test_default_prompt_is_loaded_from_yaml() -> None:
    config = load_aitr_config()
    prompt = load_aitr_prompt(config.prompt)
    assert prompt.name == "aitr"
    assert prompt.version == "v1"
    assert prompt.path == config.prompt
    assert len(prompt.hash) == 64
    assert "Initiative calibration" in prompt.system
    assert prompt.user_template.count("{conversation}") == 1


def test_prompt_builds_only_system_and_natural_conversation_messages() -> None:
    prompt = load_aitr_prompt()
    messages = (
        ConversationMessage(0, "user", "Need <help> & advice."),
        ConversationMessage(1, "assistant", "What matters most?"),
    )
    rendered = prompt.build_messages(messages)
    assert [item["role"] for item in rendered] == ["system", "user"]
    assert "Need &lt;help&gt; &amp; advice." in rendered[1]["content"]
    assert rendered[1]["content"].startswith("<conversation>")
    assert rendered[1]["content"].endswith("</conversation>")


def test_prompt_contract_rejects_unknown_placeholder(tmp_path) -> None:
    path = tmp_path / "invalid_prompt.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "invalid",
                "version": "v2",
                "system": "Evaluate.",
                "user_template": "{conversation}\n{hidden_dag}",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="placeholders must be exactly"):
        AITRPromptTemplate.load(path)


def test_prompt_hash_changes_when_yaml_content_changes(tmp_path) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    base = {
        "name": "custom",
        "version": "v2",
        "system": "Evaluate adaptive interaction.",
        "user_template": "<conversation>\n{conversation}\n</conversation>",
    }
    first_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    second_path.write_text(
        yaml.safe_dump({**base, "system": "Evaluate adaptive interaction carefully."}),
        encoding="utf-8",
    )
    assert AITRPromptTemplate.load(first_path).hash != AITRPromptTemplate.load(second_path).hash


def test_custom_judge_config_can_reference_custom_prompt(tmp_path) -> None:
    prompt_path = tmp_path / "custom_prompt.yaml"
    prompt_path.write_text(
        yaml.safe_dump(
            {
                "name": "custom",
                "version": "v2",
                "system": "Evaluate adaptive interaction.",
                "user_template": "{conversation}",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "custom_aitr.yaml"
    config_path.write_text(f"prompt: {prompt_path}\n", encoding="utf-8")
    config = load_aitr_config(config_path)
    prompt = load_aitr_prompt(config.prompt)
    assert Path(config.prompt) == prompt_path
    assert prompt.version == "v2"
