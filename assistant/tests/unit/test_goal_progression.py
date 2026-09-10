"""GP v2 end-to-end local contracts, using synthetic model calls only."""

import json
from collections import Counter

import pytest
from gp_helpers import ROOT, VARIANTS, Client, profile, session

from assistant.config import load_config
from assistant.factory import build_assistant_components
from assistant.goal_progression.prompts import snapshot


@pytest.mark.parametrize("variant", VARIANTS)
async def test_six_variants_share_the_existing_session_and_profile_entrypoints(variant):
    client = Client()
    agent = session(variant, client=client)
    events = []
    reply = await agent.respond(
        "Create a synthetic card.", audit_sink=lambda k, p: events.append((k, p))
    )
    assert reply == "A complete synthetic card."
    assert len(client.calls) == (4 if variant in {"full", "no_anticipate"} else 3)
    assert agent.history[-1].content == reply
    assert agent.last_call_metadata["visible_response_tokens"] == len(reply.split())
    assert events[-1][0] == "assistant_gp_state_committed"
    assert all(e[1]["architecture_version"] == "v2_contracts" for e in events)
    assert "SECRET_REASONING" not in json.dumps(events)
    assert agent.last_call_metadata["goal_progression"]["committed"]
    if variant == "no_tracker":
        assert agent.state is None
    else:
        assert agent.state.goals[0].ref == "g0001"
    if variant == "full":
        assert client.max_parallel_planners == 2
    for call in client.calls:
        if call["role"] not in {"tracker"} and variant != "no_tracker":
            assert "visible_history" not in call["context"]
        if call["role"] == "generator":
            assert "previous_snapshot" not in call["context"]
            assert "previous_execution_bridge" not in call["context"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_profile_and_prompt_snapshot_are_reproducible(variant):
    p = profile(variant)
    snap = snapshot(p)
    assert snap["architecture_version"] == "v2_contracts"
    assert p.retry.max_attempts == 1
    assert snap["prompts"]["generator"]["hash"]
    assert all(value == 3 for value in snap["effective_retry_limits"]["generator"].values())
    assert "review_operation" not in json.dumps(snap)


def test_factory_builds_existing_baseline_name():
    config = load_config(ROOT / "configs/assistant.yaml")
    config.components.baseline = "goal_progression"
    config.models["assistant"] = ROOT / "configs/models/qwen_3_6_27b_gp.yaml"
    built = build_assistant_components(config=config, client=Client())
    assert built.session.baseline.name == "goal_progression"
    assert built.model_profile.goal_progression.architecture_version == "v2_contracts"


async def test_repeated_need_does_not_become_fresh_on_the_next_turn():
    client = Client(ask=True)
    agent = session(client=client)
    first = await agent.respond("Please create a synthetic card.")
    assert "Which edge shape" in first
    assert agent.interaction_state["n0001"]["issued_count"] == 1
    second = await agent.respond("The value has not been provided.")
    assert "Which edge shape" not in second
    assert agent.interaction_state["n0001"]["issued_count"] == 1
    assert Counter(c["role"] for c in client.calls) == {
        "tracker": 2,
        "intra": 2,
        "inter": 2,
        "generator": 2,
    }
