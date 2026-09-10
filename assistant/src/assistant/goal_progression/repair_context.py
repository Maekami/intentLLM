"""Show a rejected field's local object, not an unrelated prefix of a long output."""

import json
import re


def repair_excerpt(raw, correction, limit):
    errors = (correction or {}).get("errors", [])
    path = errors[0].get("field_path", "") if errors else ""
    try:
        value = json.loads(raw)
        if not re.fullmatch(r"\$(?:\.[A-Za-z_][A-Za-z_0-9]*|\[[0-9]+\])+", path):
            return raw[:limit]
        tokens = re.findall(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[([0-9]+)\]", path)
        selected = value
        for key, index in tokens:
            if isinstance(value, dict):
                selected = value
            value = value[key] if key else value[int(index)]
        if isinstance(value, (dict, list)):
            selected = value
        excerpt = json.dumps(
            {"field_path": path, "rejected_value": value, "rejected_fragment": selected},
            ensure_ascii=False,
        )
        return excerpt[:limit]
    except (ValueError, TypeError, KeyError, IndexError):
        return raw[:limit]
