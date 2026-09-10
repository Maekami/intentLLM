"""Citation pairs use supplied evidence, never guessed event/block combinations."""

import pytest
from test_gp_projection_v2 import projection

from assistant.goal_progression.context import ContextBuilder
from assistant.goal_progression.contracts import output_schema, validate_wire_required
from assistant.goal_progression.errors import ContractError
from assistant.goal_progression.events import IdentityIndex
from assistant.goal_progression.schemas import SemanticProjection

jsonschema = pytest.importorskip("jsonschema")


@pytest.mark.parametrize(
    "event,block,valid",
    [
        ("m1", None, True),
        ("m2", "draft", True),
        ("m2", None, True),
        ("m1", "draft", False),
        ("m3", None, False),
        ("m2", "other", False),
    ],
)
def test_event_block_pairs_agree_between_schema_and_prompt_mode_validation(event, block, valid):
    schema = output_schema(
        SemanticProjection,
        registry=IdentityIndex(),
        evidence_domain={"m1": [None], "m2": [None, "draft"]},
    )
    value = projection()
    value["current_goals"][0]["material_refs"] = [{"event_id": event, "block_id": block}]
    jsonschema.Draft202012Validator.check_schema(schema)
    assert jsonschema.Draft202012Validator(schema).is_valid(value) is valid
    if valid:
        validate_wire_required(value, schema, "tracker")
    else:
        with pytest.raises(ContractError) as error:
            validate_wire_required(value, schema, "tracker")
        assert error.value.owner == "tracker"
        assert error.value.field_path.startswith("$.current_goals[0].material_refs[0].")


def test_scoped_source_domain_does_not_expand_block_to_unseen_whole_message():
    context = {
        "latest_user": {"event_id": "m3", "content": "Refine the draft."},
        "evidence": [{"event_id": "m2", "block_id": "draft", "content": "Card draft."}],
        "irrelevant_metadata": {"event_id": "m1", "block_id": "unseen"},
    }
    assert ContextBuilder.evidence_domain(context) == {"m3": [None], "m2": ["draft"]}


def test_tracker_history_domain_contains_actual_blocks_on_their_original_events():
    context = {
        "visible_history": [
            {"event_id": "m1", "role": "user", "content": "Make a card."},
            {
                "event_id": "m2",
                "role": "assistant",
                "content": "Card.",
                "blocks": {"card": {"start": 0, "end": 5}},
            },
        ]
    }
    assert ContextBuilder.evidence_domain(context) == {"m1": [None], "m2": [None, "card"]}
