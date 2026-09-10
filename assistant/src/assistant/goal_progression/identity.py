"""One typed identity contract for decoding, validation and repair feedback."""

import re

from .errors import ContractError

# Machine-local names need no Unicode. Use a positive ASCII class: XGrammar can
# expand \S to a negated class containing U+00A0 and clamp that class to ASCII.
# Keep this one pattern shared by the wire schema and local validation.
NEW_REF_PATTERN = r"^new:[A-Za-z0-9_./-]+$"
GROUP_KINDS = {"goals": "goal", "constraints": "constraint", "needs": "need"}


def registered_ids(registry):
    return {
        kind: sorted(ref for ref, entry in registry.entries.items() if entry["kind"] == kind)
        for kind in GROUP_KINDS.values()
    }


def reference_schema(registry, kind, *, extra_ids=()):
    existing = sorted(set(registered_ids(registry)[kind]) | set(extra_ids))
    new = {"type": "string", "pattern": NEW_REF_PATTERN}
    return {"anyOf": [{"type": "string", "enum": existing}, new]} if existing else new


def validate_reference(ref, kind, registry, owner, path, *, extra_ids=()):
    entry = registry.entries.get(ref)
    if (entry is not None and entry["kind"] == kind) or ref in extra_ids:
        return
    if entry is None and re.fullmatch(NEW_REF_PATTERN, ref):
        return
    existing = registered_ids(registry)[kind]
    raise ContractError(
        "contract.reference",
        owner,
        f"Invalid {kind} reference {ref!r}. Registered {kind} IDs: {existing}. "
        "Reuse a registered ID only for the same entity. For a genuinely new entity, "
        "declare a unique new:<local_name> using only ASCII letters, digits, _, ., /, - "
        "in local_name, and use it consistently in "
        "every linked field in this output. Only runtime assigns canonical IDs; "
        "do not invent or increment them.",
        path,
    )


def validate_semantic_identities(snapshot, registry, owner, *, owners=None, anchor_ids=()):
    """Check against the INPUT registry, before new IDs are allocated into a clone.

    Otherwise a guessed canonical ID can accidentally become valid when an earlier
    local entity in the same output gets allocated that number.
    """
    owners = owners or {}
    for group, kind in GROUP_KINDS.items():
        for index, item in enumerate(getattr(snapshot, group)):
            who = owners.get(item.ref, owner)
            path = f"$.{group}[{index}]"
            validate_reference(item.ref, kind, registry, who, path + ".ref")
            fields = {
                "goals": (("anchor_goal_ids", "goal"),),
                "constraints": (("applies_to", "goal"),),
                "needs": (),
            }[group]
            for name, target_kind in fields:
                for offset, ref in enumerate(getattr(item, name)):
                    validate_reference(
                        ref,
                        target_kind,
                        registry,
                        who,
                        f"{path}.{name}[{offset}]",
                        extra_ids=anchor_ids if name == "anchor_goal_ids" else (),
                    )
            if group == "needs":
                validate_reference(item.goal_id, "goal", registry, who, path + ".goal_id")
    for index, ref in enumerate(snapshot.changed_goal_ids):
        validate_reference(ref, "goal", registry, owner, f"$.changed_goal_ids[{index}]")
