"""State-shape invariants shared by all semantic producers' wire contracts."""

import copy


def seal_states(schema, model_name, registry=None):
    definitions = schema.get("$defs", {})
    snapshot = (
        schema
        if model_name in {"SemanticSnapshot", "SemanticProjection"}
        else definitions.get("SemanticSnapshot")
    )
    if snapshot:
        snapshot["properties"].pop("changed_goal_ids", None)
        snapshot["required"] = [name for name in snapshot["required"] if name != "changed_goal_ids"]
    constraint = definitions.get("Constraint")
    if constraint:
        constraint["properties"]["applies_to"]["minItems"] = 1
    goal = definitions.get("Goal")
    if goal:
        # Only Constraint.applies_to owns this relation. The inverse index is
        # runtime state, not a second semantic decision for the model to copy.
        goal["properties"].pop("constraint_ids", None)
        goal["required"] = [name for name in goal["required"] if name != "constraint_ids"]
        scopes = (
            ("current",)
            if model_name == "IntraPacket"
            else (("adjacent",) if model_name == "InterPacket" else ("current", "adjacent"))
        )
        branches = []
        for scope in scopes:
            for active in (True, False):
                branch = copy.deepcopy(goal)
                props = branch["properties"]
                props["scope"] = {"const": scope}
                props["status"] = {"enum": ["active"] if active else ["paused", "addressed"]}
                if active:
                    props["remaining_work"]["minLength"] = 1
                if scope == "current":
                    props["anchor_goal_ids"]["maxItems"] = 0
                elif active:
                    props["anchor_goal_ids"]["minItems"] = 1
                branches.append(branch)
        definitions["Goal"] = {"anyOf": branches}
        if model_name == "SemanticProjection":
            for scope in ("current", "adjacent"):
                name = scope.title() + "Goal"
                definitions[name] = {
                    "anyOf": [b for b in branches if b["properties"]["scope"]["const"] == scope]
                }
                schema["properties"][scope + "_goals"]["items"] = {"$ref": f"#/$defs/{name}"}
            schema["properties"]["current_goals"]["minItems"] = 1
            del definitions["Goal"]
    need = definitions.get("InformationNeed")
    if need:
        # ref owns identity/history; goal_id owns current applicability. Their
        # typed ID domains were sealed by contracts.py; do not couple the latter
        # to the original goal and force obsolete task state into this projection.
        branches = []
        for status in ("unknown", "available", "unavailable"):
            branch = copy.deepcopy(need)
            branch["properties"]["status"] = {"const": status}
            if status == "available":
                branch["properties"]["answer_refs"]["minItems"] = 1
            branches.append(branch)
        definitions["InformationNeed"] = {"anyOf": branches}
    return schema
