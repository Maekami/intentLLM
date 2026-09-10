"""Structural constraints only: no semantic regexes, latent evaluators or patch rules."""

import copy

from .action_schema import required_fields_branch, seal_actions
from .errors import ContractError
from .identity import reference_schema
from .schemas import CurrentRealization, InterPlan
from .state_schema import seal_states


def validate_wire_required(value, schema, owner):
    """Enforce wire ownership/presence and the same source domain used by decoding."""
    evidence = schema.get("$defs", {}).get("EvidenceRef", {})
    source_pairs = {
        branch["properties"]["event_id"]["const"]: branch["properties"]["block_id"]["enum"]
        for branch in evidence.get("anyOf", [])
    }

    def visit(data, node, path):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[1]
            node = schema["$defs"][name]
            if name == "EvidenceRef" and source_pairs and isinstance(data, dict):
                event, block = data.get("event_id"), data.get("block_id")
                if (
                    isinstance(event, str)
                    and "block_id" in data
                    and (block is None or isinstance(block, str))
                    and (event not in source_pairs or block not in source_pairs[event])
                ):
                    raise ContractError(
                        "contract.reference",
                        owner,
                        "The event/block pair is not a source visible to this role. "
                        f"Allowed pairs by event: {source_pairs}. Do not guess IDs or "
                        "attach another event's block. null means the whole visible event.",
                        path + (".event_id" if event not in source_pairs else ".block_id"),
                    )
        if "anyOf" in node:
            if data is None:
                return
            candidates = [n for n in node["anyOf"] if n.get("type") != "null"]
            if len(candidates) == 1:
                return visit(data, candidates[0], path)
            branch = required_fields_branch(node, data)
            if branch is not None:
                return visit(data, branch, path)
        if isinstance(data, dict) and node.get("type") == "object":
            missing = set(node.get("required", [])) - data.keys()
            if missing:
                raise ContractError(
                    "output.schema",
                    owner,
                    "Missing required wire fields",
                    path + "." + min(missing),
                )
            extra = data.keys() - node.get("properties", {}).keys()
            if extra and node.get("additionalProperties") is False:
                raise ContractError(
                    "output.schema",
                    owner,
                    "Field is not part of this producer's wire contract; omit it",
                    path + "." + min(extra),
                )
            for key, item in data.items():
                if key in node.get("properties", {}):
                    visit(item, node["properties"][key], path + "." + key)
        elif isinstance(data, list) and node.get("type") == "array":
            for index, item in enumerate(data):
                visit(item, node.get("items", {}), f"{path}[{index}]")

    visit(value, schema, "$")


def output_schema(
    model,
    *,
    snapshot=None,
    role=None,
    variant="full",
    unit_ids=None,
    registry=None,
    anchor_ids=(),
    input_reference_ids=None,
    request_eligibility=None,
    evidence_domain=None,
):
    schema = copy.deepcopy(model.model_json_schema())

    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    definitions = schema.get("$defs", {})
    if evidence_domain and "EvidenceRef" in definitions:
        # A block belongs to one event; independently selecting two strings
        # is not a valid citation contract. Do not expose unrelated source IDs.
        template = definitions["EvidenceRef"]
        branches = []
        for event_id, blocks in evidence_domain.items():
            branch = copy.deepcopy(template)
            branch["properties"]["event_id"] = {"const": event_id}
            branch["properties"]["block_id"] = {"enum": blocks}
            branches.append(branch)
        definitions["EvidenceRef"] = {"anyOf": branches}
    if registry is not None:
        goal_ref = reference_schema(registry, "goal")
        need_ref = reference_schema(registry, "need")
        constraint_ref = reference_schema(registry, "constraint")
        bindings = {
            "Goal": {
                "ref": goal_ref,
                "anchor_goal_ids": reference_schema(registry, "goal", extra_ids=anchor_ids),
                "constraint_ids": constraint_ref,
            },
            "Constraint": {"ref": constraint_ref, "applies_to": goal_ref},
            "InformationNeed": {"ref": need_ref, "goal_id": goal_ref},
            "SemanticSnapshot": {"changed_goal_ids": goal_ref},
            "IntraItem": {"goal_id": goal_ref, "blocked_by": need_ref},
            "DeliveryProposal": {"goal_id": goal_ref, "required_need_ids": need_ref},
            "RequestProposal": {"need_id": need_ref},
            "InterPlan": {"goal_id": {"anyOf": [{"type": "null"}, goal_ref]}},
        }
        for name, fields in bindings.items():
            root = model.__name__ == name or (
                name == "SemanticSnapshot" and model.__name__ == "SemanticProjection"
            )
            definition = schema if root else definitions.get(name)
            if not definition:
                continue
            if name in {"Goal", "Constraint", "InformationNeed"}:
                definition["properties"]["evidence_refs"]["minItems"] = 1
            for field, ref_schema in fields.items():
                prop = definition["properties"][field]
                if prop.get("type") == "array":
                    prop["items"] = copy.deepcopy(ref_schema)
                else:
                    definition["properties"][field] = copy.deepcopy(ref_schema)
    if variant == "no_anticipate":
        inter = schema if model is InterPlan else definitions.get("InterPlan")
        if inter:
            inter["properties"]["action"]["enum"] = ["none", "elicit"]
    if snapshot is not None:
        current = [g.ref for g in snapshot.goals_for("current")]
        adjacent = [g.ref for g in snapshot.goals_for("adjacent")]
        # Every planner reference uses a state-derived domain, not a free-text
        # string. This is the same typed contract used by local validators;
        # it does not infer semantics or repair malformed IDs by editing them.
        visible = (
            adjacent
            if model is InterPlan or role == "inter"
            else current
            if model.__name__ in {"IntraPlan", "CurrentRealization"}
            or role in {"intra", "generator"}
            else current + adjacent
        )
        needs = [n for n in snapshot.needs if n.goal_id in visible]
        available = [n.ref for n in needs if n.status == "available"]
        pending = [n.ref for n in needs if n.status != "available"]
        requestable = [
            n.ref
            for n in needs
            if n.status == "unknown"
            and (
                request_eligibility is None
                or request_eligibility.get(n.ref, {}).get("request_eligible", False)
            )
        ]

        def bind_array(prop, domain):
            if domain:
                prop["items"] = {"type": "string", "enum": list(domain)}
            else:
                prop["maxItems"] = 0

        delivery = definitions.get("DeliveryProposal")
        if delivery:
            if visible:
                delivery["properties"]["goal_id"] = {"type": "string", "enum": visible}
            bind_array(delivery["properties"]["required_need_ids"], available)
        request = definitions.get("RequestProposal")
        if request and requestable:
            request["properties"]["need_id"] = {"type": "string", "enum": requestable}
        item = definitions.get("IntraItem")
        if item and current:
            item["properties"]["goal_id"] = {"type": "string", "enum": current}
            bind_array(item["properties"]["blocked_by"], pending)
            if not requestable:
                item["properties"]["request"] = {"type": "null"}
                item["properties"]["action"] = {"enum": ["advance", "revise", None]}
        intra = definitions.get("IntraPlan")
        if model.__name__ == "IntraPlan":
            intra = schema
        if intra:
            intra["properties"]["items"].update(minItems=len(current), maxItems=len(current))
        inter = schema if model is InterPlan else definitions.get("InterPlan")
        if inter:
            inter["properties"]["goal_id"] = {"enum": [None, *adjacent]}
            if not requestable:
                inter["properties"]["request"] = {"type": "null"}
                inter["properties"]["action"] = {
                    "type": "string",
                    "enum": ["none"] if variant == "no_anticipate" else ["none", "anticipate"],
                }
            if not adjacent:
                inter["properties"]["action"] = {"type": "string", "enum": ["none"]}
                inter["properties"]["request"] = {"type": "null"}
                inter["properties"]["deliveries"]["maxItems"] = 0
    if unit_ids is not None and model is not CurrentRealization:
        body = definitions.get("BodyUnit")
        issue = definitions.get("RealizationIssue")
        for definition in (body, issue):
            if definition and unit_ids:
                definition["properties"]["unit_id"] = {"type": "string", "enum": list(unit_ids)}
        for field in ("units", "issues"):
            schema["properties"][field]["maxItems"] = len(unit_ids)
        if input_reference_ids is not None and issue:
            if input_reference_ids:
                issue["properties"]["input_refs"]["items"] = {
                    "type": "string",
                    "enum": list(input_reference_ids),
                }
            else:
                issue["properties"]["input_refs"]["maxItems"] = 0
    return seal_states(seal_actions(schema, model.__name__), model.__name__, registry)


def _request(request, goal_id, needs, eligibility, owner):
    if request is None:
        return
    need = needs.get(request.need_id)
    if need is None or need.goal_id != goal_id:
        raise ContractError(
            "contract.request", owner, "Request must reference this goal's Need", "request.need_id"
        )
    if not eligibility[need.ref]["request_eligible"]:
        raise ContractError(
            "contract.request",
            owner,
            f"Need {need.ref!r} is not eligible for another request. Eligible IDs: "
            f"{[ref for ref, view in eligibility.items() if view['request_eligible']]}. "
            "Choose useful independent delivery without a request; if genuinely blocked, "
            "use action=null and exact Need IDs in blocked_by. Do not restate a prior question.",
            "request.need_id",
        )
    if not request.question_text.strip():
        raise ContractError(
            "contract.request", owner, "Request text is blank", "request.question_text"
        )


def _delivery(delivery, goal_id, needs, store, owner):
    if delivery.goal_id != goal_id or not delivery.target.strip():
        raise ContractError(
            "contract.reference", owner, "Delivery goal/target is invalid", "deliveries"
        )
    for ref in delivery.material_refs:
        store.resolve(ref, owner)
    for ref in delivery.required_need_ids:
        need = needs.get(ref)
        if need is None:
            raise ContractError("contract.reference", owner, "Unknown required Need", ref)
        if need.status != "available":
            raise ContractError(
                "contract.dependency",
                owner,
                "A direct delivery requires unavailable information",
                ref,
            )


def validate_intra(plan, snapshot, eligibility, store, owner="intra"):
    expected = {g.ref for g in snapshot.goals_for("current")}
    received = [item.goal_id for item in plan.items]
    if len(received) != len(set(received)) or set(received) != expected:
        raise ContractError(
            "contract.coverage", owner, "Cover all active current goals exactly once", "items"
        )
    needs = {n.ref: n for n in snapshot.needs if n.goal_id in expected}
    for item_index, item in enumerate(plan.items):
        if item.action is None:
            if item.deliveries or item.request or not item.blocked_by:
                raise ContractError(
                    "contract.action",
                    owner,
                    "Null action is only a blocked item without delivery/request",
                    item.goal_id,
                )
        elif item.action == "clarify":
            if item.request is None:
                raise ContractError(
                    "contract.request", owner, "Clarify requires a request", item.goal_id
                )
        elif item.request is not None or not item.deliveries:
            raise ContractError(
                "contract.action",
                owner,
                "Advance/revise need delivery and no request",
                item.goal_id,
            )
        _request(item.request, item.goal_id, needs, eligibility, owner)
        for delivery in item.deliveries:
            _delivery(delivery, item.goal_id, needs, store, owner)
        for offset, ref in enumerate(item.blocked_by):
            need = needs.get(ref)
            if need is None or need.goal_id != item.goal_id or need.status == "available":
                valid = [
                    n.ref
                    for n in needs.values()
                    if n.goal_id == item.goal_id and n.status != "available"
                ]
                raise ContractError(
                    "contract.dependency",
                    owner,
                    f"Invalid blocked Need {ref!r}. blocked_by contains exact Need IDs only: "
                    f"{valid}. Do not append explanations or put prose in reference fields. "
                    "A blocker must belong to this goal and must not be available.",
                    f"$.items[{item_index}].blocked_by[{offset}]",
                )


def validate_inter(plan, snapshot, eligibility, store, policy, variant="full", owner="inter"):
    if plan.action == "none":
        if plan.goal_id is not None or plan.deliveries or plan.request is not None:
            raise ContractError("contract.action", owner, "none must have empty payload", "inter")
        return
    if plan.goal_id not in {g.ref for g in snapshot.goals_for("adjacent")}:
        raise ContractError(
            "contract.reference", owner, "Inter must reference an active adjacent goal", "goal_id"
        )
    visible_goals = {g.ref for g in snapshot.goals_for("adjacent")}
    needs = {n.ref: n for n in snapshot.needs if n.goal_id in visible_goals}
    if plan.action == "elicit":
        if plan.request is None or plan.deliveries:
            raise ContractError("contract.action", owner, "elicit requires only a request", "inter")
    else:
        if variant == "no_anticipate":
            raise ContractError("contract.action", owner, "anticipate is disabled", "action")
        if not plan.deliveries or plan.request is not None:
            raise ContractError(
                "contract.action", owner, "anticipate requires only deliveries", "inter"
            )
        if len(plan.deliveries) > policy.max_adjacent_deliveries:
            raise ContractError(
                "contract.coverage", owner, "Too many adjacent delivery units", "deliveries"
            )
    _request(plan.request, plan.goal_id, needs, eligibility, owner)
    for delivery in plan.deliveries:
        _delivery(delivery, plan.goal_id, needs, store, owner)
