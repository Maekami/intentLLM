"""Compile finite action contracts into disjoint, structurally legal wire choices.

The local validators remain authoritative, including in prompt-only decoding.
No free-text semantic decisions or silent repairs are performed here.
"""

import copy


def permits(node, value):
    if "const" in node:
        return node["const"] == value
    if "enum" in node:
        return value in node["enum"]
    if "anyOf" in node:
        return any(permits(branch, value) for branch in node["anyOf"])
    return (value is None) if node.get("type") == "null" else value is not None


def nonnull(node):
    result = copy.deepcopy(node)
    if "enum" in result:
        result["enum"] = [v for v in result["enum"] if v is not None]
    if "anyOf" in result:
        result["anyOf"] = [v for v in result["anyOf"] if v.get("type") != "null"]
        if len(result["anyOf"]) == 1:
            return result["anyOf"][0]
    return result


def seal_action(node, *, inter=False):
    definitions = node.get("$defs")
    template = {k: copy.deepcopy(v) for k, v in node.items() if k != "$defs"}
    props = template["properties"]
    choices = []

    def branch(action):
        value = copy.deepcopy(template)
        value["properties"]["action"] = {"const": action}
        choices.append(value)
        return value["properties"]

    if inter:
        if permits(props["action"], "none"):
            p = branch("none")
            p["goal_id"] = p["request"] = {"type": "null"}
            p["deliveries"]["maxItems"] = 0
        if permits(props["action"], "elicit") and props["request"].get("type") != "null":
            p = branch("elicit")
            p["goal_id"] = nonnull(p["goal_id"])
            p["request"] = nonnull(p["request"])
            p["deliveries"]["maxItems"] = 0
        if permits(props["action"], "anticipate") and props["deliveries"].get("maxItems") != 0:
            p = branch("anticipate")
            p["goal_id"] = nonnull(p["goal_id"])
            p["request"] = {"type": "null"}
            p["deliveries"]["minItems"] = 1
    else:
        for action in ("advance", "revise"):
            if permits(props["action"], action):
                p = branch(action)
                p["request"] = {"type": "null"}
                p["deliveries"]["minItems"] = 1
        if permits(props["action"], "clarify") and props["request"].get("type") != "null":
            p = branch("clarify")
            p["request"] = nonnull(p["request"])
        if permits(props["action"], None) and props["blocked_by"].get("maxItems") != 0:
            p = branch(None)
            p["request"] = {"type": "null"}
            p["deliveries"]["maxItems"] = 0
            p["blocked_by"]["minItems"] = 1
    if not choices:
        raise ValueError("No legal GP action branch")
    node.clear()
    node["anyOf"] = choices
    if definitions is not None:
        node["$defs"] = definitions


def seal_actions(schema, model_name):
    definitions = schema.get("$defs", {})
    if "IntraItem" in definitions:
        seal_action(definitions["IntraItem"])
    if model_name == "InterPlan":
        seal_action(schema, inter=True)
    elif "InterPlan" in definitions:
        seal_action(definitions["InterPlan"], inter=True)
    return schema


def required_fields_branch(node, data):
    """Select a tagged object for wire-field presence checks, not value coercion.

    All tagged alternatives preserve the same required field set. An unknown
    tag still gets presence checked; model/local validation reports its
    value or contract violation using the existing error taxonomy.
    """
    objects = [branch for branch in node["anyOf"] if branch.get("type") == "object"]
    if not isinstance(data, dict) or not objects:
        return None
    matching = [
        branch
        for branch in objects
        if all(
            permits(prop, data.get(key))
            for key, prop in branch.get("properties", {}).items()
            if "const" in prop or "enum" in prop
        )
    ]
    return (matching or objects)[0]
