from pathlib import Path

import pytest
import yaml

from assistant.baselines.base import BaseBaseline
from assistant.baselines.prompt_base import PromptBaseBaseline
from assistant.baselines.trace2skill import Trace2SkillBaseline
from assistant.config import load_config, load_model_profile
from assistant.domain.messages import ChatMessage
from assistant.exceptions import ConfigurationError
from assistant.factory import build_assistant_components, configured_components_require_openrouter
from assistant.prompt import SystemPrompt
from assistant.skill import StaticSkill


def test_default_config_uses_base_and_qwen() -> None:
    config = load_config()
    assert config.components.baseline == "base"
    assert config.models["assistant"] == "qwen_3_6_27b"
    profile = load_model_profile(config.models["assistant"])
    assert profile.model_id == "qwen/qwen3.6-27b"
    assert profile.generation["assistant"].temperature == 1.0


def test_luna_non_thinking_profile_disables_reasoning() -> None:
    profile = load_model_profile("gpt_5_6_luna_non_thinking")
    assert profile.model_id == "openai/gpt-5.6-luna"
    assert profile.reasoning.enabled is False
    assert profile.reasoning.effort == "none"
    assert profile.generation["assistant"].temperature is None
    assert profile.generation["assistant"].max_completion_tokens == 32768


def test_gemini_profiles_use_supported_high_and_minimal_thinking_levels() -> None:
    thinking = load_model_profile("gemini_3_6_flash")
    non_thinking = load_model_profile("gemini_3_6_flash_non_thinking")

    for profile in (thinking, non_thinking):
        assert profile.model_id == "google/gemini-3.6-flash"
        assert profile.reasoning.enabled is True
        assert profile.generation["assistant"].temperature is None
        assert profile.generation["assistant"].top_p is None
        assert profile.generation["assistant"].max_completion_tokens == 65536
    assert thinking.reasoning.effort == "high"
    assert non_thinking.reasoning.effort == "minimal"


def test_qwen_profile_is_available() -> None:
    profile = load_model_profile("qwen_3_6_27b")
    assert profile.model_id == "qwen/qwen3.6-27b"
    assert profile.reasoning.enabled is True
    assert profile.reasoning.local_chat_template.enable_thinking is True
    assert profile.generation["assistant"].temperature == 1.0
    assert profile.generation["assistant"].top_p == 0.95


def test_qwen_non_thinking_profile_uses_official_switch_and_sampling() -> None:
    profile = load_model_profile("qwen_3_6_27b_non_thinking")
    assert profile.reasoning.enabled is False
    assert profile.reasoning.local_chat_template.enable_thinking is False
    assert profile.generation["assistant"].temperature == 0.7
    assert profile.generation["assistant"].top_p == 0.8
    assert profile.generation["assistant"].presence_penalty == 1.5


def test_local_vllm_qwen_profiles_match_served_alias_and_thinking_modes() -> None:
    thinking = load_model_profile("qwen_3_6_27b_vllm")
    non_thinking = load_model_profile("qwen_3_6_27b_vllm_non_thinking")

    for profile in (thinking, non_thinking):
        assert profile.provider == "vllm"
        assert profile.model_id == "qwen3.6-27b"
        assert profile.base_url == "http://127.0.0.1:8001/v1"
        assert profile.routing == {}
    assert thinking.reasoning.enabled is True
    assert thinking.reasoning.local_chat_template.enable_thinking is True
    assert thinking.reasoning.local_chat_template.preserve_thinking is True
    assert non_thinking.reasoning.enabled is False
    assert non_thinking.reasoning.local_chat_template.enable_thinking is False
    assert non_thinking.reasoning.local_chat_template.preserve_thinking is False


def test_openrouter_requirement_follows_selected_model_provider() -> None:
    openrouter = load_config(cli_overrides={"models": {"assistant": "qwen_3_6_27b"}})
    local = load_config(cli_overrides={"models": {"assistant": "qwen_3_6_27b_vllm"}})

    assert configured_components_require_openrouter(openrouter) is True
    assert configured_components_require_openrouter(local) is False


def test_deepseek_v4_flash_0731_profile_is_available() -> None:
    profile = load_model_profile("deepseek_v4_flash_0731")
    assert profile.model_id == "deepseek/deepseek-v4-flash-0731"
    assert profile.reasoning.enabled is False
    assert profile.reasoning.effort == "high"
    assert profile.generation["assistant"].temperature == 1.0
    assert profile.generation["assistant"].max_completion_tokens == 8192


def test_base_adds_no_system_prompt() -> None:
    history = [ChatMessage(role="user", content="hello")]
    assert BaseBaseline().build_messages(history) == [{"role": "user", "content": "hello"}]


def test_prompt_base_prepends_configured_system_prompt() -> None:
    prompt = SystemPrompt.load(Path("configs/prompts/prompt_base.yaml"))
    history = [ChatMessage(role="user", content="hello")]
    messages = PromptBaseBaseline(prompt).build_messages(history)
    assert messages[0] == {"role": "system", "content": prompt.system}
    assert messages[1] == {"role": "user", "content": "hello"}
    assert prompt.system.startswith("The assistant is designed to be helpful")


def test_trace2skill_loads_markdown_and_prepends_it_verbatim(tmp_path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("# Dialogue skill\n\nAsk concise follow-up questions.\n", encoding="utf-8")
    skill = StaticSkill.load(path)
    history = [ChatMessage(role="user", content="hello")]

    messages = Trace2SkillBaseline(skill).build_messages(history)

    assert messages == [
        {
            "role": "system",
            "content": "# Dialogue skill\n\nAsk concise follow-up questions.",
        },
        {"role": "user", "content": "hello"},
    ]
    assert len(skill.hash) == 64


def test_trace2skill_rejects_an_empty_skill(tmp_path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty Trace2Skill"):
        StaticSkill.load(path)


def test_trace2skill_is_available_through_the_factory(tmp_path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("Use the visible dialogue to answer directly.\n", encoding="utf-8")
    config = load_config(
        cli_overrides={
            "components": {"baseline": "trace2skill"},
            "prompts": {"trace2skill": str(path)},
            "models": {"assistant": "qwen_3_6_27b_non_thinking"},
        }
    )
    fake_client = type("FakeClient", (), {"last_call_metadata": {}})()

    components = build_assistant_components(config=config, client=fake_client)

    assert components.baseline.name == "trace2skill"
    assert components.memory_framework is None
    assert components.prompt is not None
    assert components.prompt.system == "Use the visible dialogue to answer directly."


def test_trace2skill_rejects_a_memory_enabled_profile(tmp_path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text("Use the visible dialogue only.\n", encoding="utf-8")
    config = load_config(
        cli_overrides={
            "components": {"baseline": "trace2skill"},
            "prompts": {"trace2skill": str(path)},
            "models": {"assistant": "qwen_3_6_27b_exprag"},
        }
    )

    with pytest.raises(ConfigurationError, match="memory-free"):
        configured_components_require_openrouter(config)


def _write_profile_bound_trace2skill(tmp_path: Path, *, skill_path: Path) -> Path:
    source = load_model_profile("qwen_3_6_27b_vllm_non_thinking")
    raw = source.model_dump(mode="json")
    raw["profile_name"] = "fixture_qwen_trace2skill"
    raw["skill"] = {"framework": "trace2skill", "path": str(skill_path)}
    profile_path = tmp_path / "fixture_trace2skill.yaml"
    profile_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return profile_path


def test_profile_bound_skill_activates_on_explicit_unprompted_base(tmp_path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("Use visible dialogue carefully.\n", encoding="utf-8")
    profile_path = _write_profile_bound_trace2skill(tmp_path, skill_path=skill_path)
    config = load_config(
        cli_overrides={
            "components": {"baseline": "base"},
            "models": {"assistant": str(profile_path)},
        }
    )
    fake_client = type("FakeClient", (), {"last_call_metadata": {}})()

    components = build_assistant_components(config=config, client=fake_client)

    assert config.components.baseline == "base"
    assert components.baseline.name == "trace2skill"
    assert components.baseline_source == "model_profile"
    assert components.skill_framework == "trace2skill"
    assert components.prompt is not None
    assert components.prompt.path == str(skill_path)


def test_profile_bound_skill_can_be_disabled_for_no_skill_authoring(tmp_path) -> None:
    missing_skill = tmp_path / "not-created-yet.md"
    profile_path = _write_profile_bound_trace2skill(tmp_path, skill_path=missing_skill)
    config = load_config(
        cli_overrides={
            "components": {"baseline": "base", "profile_skill_enabled": False},
            "models": {"assistant": str(profile_path)},
        }
    )
    fake_client = type("FakeClient", (), {"last_call_metadata": {}})()

    components = build_assistant_components(config=config, client=fake_client)

    assert components.baseline.name == "base"
    assert components.baseline_source == "assistant_config"
    assert components.skill_framework is None
    assert components.prompt is None


def test_profile_bound_skill_rejects_prompted_base_stacking(tmp_path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("Use visible dialogue carefully.\n", encoding="utf-8")
    profile_path = _write_profile_bound_trace2skill(tmp_path, skill_path=skill_path)
    config = load_config(
        cli_overrides={
            "components": {"baseline": "prompt_base"},
            "models": {"assistant": str(profile_path)},
        }
    )

    with pytest.raises(ConfigurationError, match="cannot be combined with prompted base"):
        configured_components_require_openrouter(config)


@pytest.mark.parametrize(
    ("trace_profile_name", "memory_profile_name"),
    (
        ("qwen_3_6_27b_trace2skill", "qwen_3_6_27b_exprag"),
        ("gpt_5_6_luna_trace2skill", "gpt_5_6_luna_exprag"),
        ("gemini_3_6_flash_trace2skill", "gemini_3_6_flash_exprag"),
    ),
)
def test_trace2skill_profiles_align_model_reasoning_and_generation(
    trace_profile_name,
    memory_profile_name,
) -> None:
    trace_profile = load_model_profile(trace_profile_name)
    memory_profile = load_model_profile(memory_profile_name)

    assert trace_profile.provider == memory_profile.provider
    assert trace_profile.model_id == memory_profile.model_id
    assert trace_profile.base_url == memory_profile.base_url
    assert trace_profile.routing == memory_profile.routing
    assert trace_profile.reasoning == memory_profile.reasoning
    assert trace_profile.generation == memory_profile.generation
    assert trace_profile.retry == memory_profile.retry
    assert trace_profile.memory is None
    assert trace_profile.skill is not None
    assert trace_profile.skill.framework == "trace2skill"
    assert trace_profile.skill.path.name == "SKILL.md"
    assert trace_profile.skill.path.parent.name == trace_profile.profile_name


def test_trace2skill_profiles_bind_three_distinct_skill_paths() -> None:
    profiles = [
        load_model_profile(name)
        for name in (
            "qwen_3_6_27b_trace2skill",
            "gpt_5_6_luna_trace2skill",
            "gemini_3_6_flash_trace2skill",
        )
    ]

    paths = [profile.skill.path for profile in profiles if profile.skill is not None]
    assert len(paths) == 3
    assert len(set(paths)) == 3
