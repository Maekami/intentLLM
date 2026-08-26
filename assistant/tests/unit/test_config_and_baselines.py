from pathlib import Path

from assistant.baselines.base import BaseBaseline
from assistant.baselines.prompt_base import PromptBaseBaseline
from assistant.config import load_config, load_model_profile
from assistant.domain.messages import ChatMessage
from assistant.factory import configured_components_require_openrouter
from assistant.prompt import SystemPrompt


def test_default_config_uses_base_and_luna() -> None:
    config = load_config()
    assert config.components.baseline == "base"
    assert config.models["assistant"] == "gpt_5_6_luna"
    profile = load_model_profile(config.models["assistant"])
    assert profile.model_id == "openai/gpt-5.6-luna"
    assert profile.generation["assistant"].temperature is None


def test_luna_non_thinking_profile_disables_reasoning() -> None:
    profile = load_model_profile("gpt_5_6_luna_non_thinking")
    assert profile.model_id == "openai/gpt-5.6-luna"
    assert profile.reasoning.enabled is False
    assert profile.reasoning.effort == "none"
    assert profile.generation["assistant"].temperature is None
    assert profile.generation["assistant"].max_completion_tokens == 32768


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
    openrouter = load_config(
        cli_overrides={"models": {"assistant": "qwen_3_6_27b"}}
    )
    local = load_config(
        cli_overrides={"models": {"assistant": "qwen_3_6_27b_vllm"}}
    )

    assert configured_components_require_openrouter(openrouter) is True
    assert configured_components_require_openrouter(local) is False


def test_deepseek_v4_flash_0731_profile_is_available() -> None:
    profile = load_model_profile("deepseek_v4_flash_0731")
    assert profile.model_id == "deepseek/deepseek-v4-flash-0731"
    assert profile.reasoning.enabled is True
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
