import pytest

from user_simulator.config import load_model_profile


def test_default_profile_loader_is_pinned_deepseek_v4_flash_0731() -> None:
    profile = load_model_profile()
    assert profile.profile_name == "deepseek_v4_flash_0731"
    assert profile.model_id == "deepseek/deepseek-v4-flash-0731"
    assert profile.reasoning.effort == "low"
    assert profile.structured_output.response_healing is True
    assert profile.retry.max_attempts == 4


@pytest.mark.parametrize(
    "profile_name",
    ["deepseek_v4_flash_0731", "deepseek_v4_pro", "gpt_5_6_luna"],
)
def test_model_profiles_use_component_level_reasoning(profile_name: str) -> None:
    profile = load_model_profile(profile_name)
    for key in ("controller", "satisfaction"):
        reasoning = profile.generation[key].reasoning
        assert reasoning is not None
        assert reasoning.enabled is True
        assert reasoning.effort == "low"
        assert reasoning.exclude_from_response is True
    for key in ("realizer_clear", "realizer_abstract"):
        reasoning = profile.generation[key].reasoning
        assert reasoning is not None


def test_qwen_3_8_vllm_profile_matches_local_service_and_component_modes() -> None:
    profile = load_model_profile("qwen_3_8_27b_vllm")
    assert profile.provider == "vllm"
    assert profile.model_id == "Qwen/Qwen3.8-27B"
    assert profile.base_url == "http://127.0.0.1:8005/v1"
    for key in ("controller", "satisfaction"):
        generation = profile.generation[key]
        reasoning = generation.reasoning
        assert generation.temperature == 1.0
        assert generation.top_p == 0.95
        assert generation.top_k == 20
        assert reasoning is not None
        assert reasoning.enabled is True
        assert reasoning.effort == "low"
        assert reasoning.local_chat_template is not None
        assert reasoning.local_chat_template.model_dump(exclude_none=True) == {
            "enable_thinking": True,
            "preserve_thinking": False,
            "reasoning_effort": "low",
        }
    for key in ("realizer_clear", "realizer_abstract"):
        reasoning = profile.generation[key].reasoning
        assert reasoning is not None
        assert reasoning.enabled is False
        assert reasoning.local_chat_template is not None
        assert reasoning.local_chat_template.model_dump(exclude_none=True) == {
            "enable_thinking": False,
            "preserve_thinking": False,
        }
