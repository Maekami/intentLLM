import pytest

from user_simulator.config import load_model_profile


def test_default_profile_loader_restores_audited_fallbacks_and_rate_limit_retry() -> None:
    profile = load_model_profile()
    expected_providers = ["baidu", "siliconflow", "nextbit", "deepinfra"]
    assert profile.profile_name == "deepseek_v4_flash_0731"
    assert profile.model_id == "deepseek/deepseek-v4-flash-0731"
    assert profile.routing.order == expected_providers
    assert profile.routing.only == expected_providers
    assert profile.routing.quantizations == ["fp8"]
    assert profile.routing.require_parameters is True
    assert profile.routing.allow_fallbacks is True
    assert profile.reasoning.effort == "low"
    assert profile.structured_output.response_healing is True
    assert profile.retry.max_attempts == 4
    assert profile.retry.rate_limit is not None
    assert profile.retry.rate_limit.max_attempts == 6
    assert profile.retry.rate_limit.rpm_initial_backoff_seconds == 15.0
    assert profile.retry.rate_limit.tpm_initial_backoff_seconds == 60.0
    assert profile.retry.rate_limit.generic_initial_backoff_seconds == 30.0
    assert profile.retry.rate_limit.maximum_backoff_seconds == 60.0
    assert profile.retry.rate_limit.jitter_ratio == 0.25
    assert profile.retry.rate_limit.honor_retry_after is True


def test_official_deepseek_profile_is_isolated_and_has_no_quantization_filter() -> None:
    profile = load_model_profile("deepseek_v4_flash_0731_official")
    baidu_profile = load_model_profile("deepseek_v4_flash_0731")
    expected_providers = ["deepseek"]
    assert profile.profile_name == "deepseek_v4_flash_0731_official"
    assert profile.model_id == "deepseek/deepseek-v4-flash-0731"
    assert profile.routing.order == expected_providers
    assert profile.routing.only == expected_providers
    assert profile.routing.quantizations is None
    assert profile.routing.require_parameters is True
    assert profile.routing.allow_fallbacks is False
    assert profile.structured_output.type == "json_schema"
    assert profile.structured_output.transport == "json_object"
    assert profile.structured_output.strict is True
    assert profile.structured_output.require_parameters is True
    assert profile.structured_output.response_healing is True
    assert profile.retry.rate_limit is not None
    assert profile.reasoning == baidu_profile.reasoning
    assert profile.generation == baidu_profile.generation
    assert baidu_profile.structured_output.transport == "json_schema"
    assert profile.retry == baidu_profile.retry


@pytest.mark.parametrize(
    "profile_name",
    [
        "deepseek_v4_flash_0731",
        "deepseek_v4_flash_0731_official",
        "deepseek_v4_pro",
        "gpt_5_6_luna",
    ],
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
