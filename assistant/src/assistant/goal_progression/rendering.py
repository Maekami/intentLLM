"""Pure, deterministic delivery composition and receipt creation."""

from .errors import ContractError
from .events import text_digest


def validate_receipt(receipt, content, event_id):
    """Validate imported facts against their actual visible text, not against a plan."""
    if (
        receipt["reply_event_id"] != event_id
        or receipt["reply_hash"] != text_digest(content)
        or receipt["span_unit"] != "unicode_codepoints"
    ):
        raise ValueError("receipt does not match visible reply")
    for group, key in (("delivered_units", "unit_id"), ("issued_requests", "need_id")):
        ids = [part[key] for part in receipt[group]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate receipt entries")
        for part in receipt[group]:
            span = part["text_span"]
            start, end = span["start"], span["end"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(content)
                or text_digest(content[start:end]) != part["text_hash"]
            ):
                raise ValueError("invalid receipt text span/hash")


def render(contract, bodies: dict[str, str], reply_event_id: str, separator="\n\n"):
    expected = {unit.unit_id for unit in contract.deliveries}
    if set(bodies) != expected:
        raise ContractError(
            "contract.coverage",
            "generator",
            "Body unit coverage mismatch",
            affected_units=sorted(expected - set(bodies)),
        )
    fragments, blocks, delivered, requests = [], {}, [], []
    position = 0

    def append(key, text):
        nonlocal position
        if not text.strip():
            raise ContractError("contract.coverage", "generator", "Blank body", key, [key])
        if fragments:
            position += len(separator)
        start = position
        fragments.append(text)
        position += len(text)
        span = {"start": start, "end": position}
        blocks[key] = {**span, "text_hash": text_digest(text)}
        return span

    for unit in contract.deliveries:
        text = bodies[unit.unit_id]
        span = append(unit.unit_id, text)
        delivered.append(
            {"unit_id": unit.unit_id, "text_span": span, "text_hash": text_digest(text)}
        )
    for index, request in enumerate(contract.requests):
        text = request.question_text
        span = append(f"request:{index}:{request.need_id}", text)
        requests.append(
            {"need_id": request.need_id, "text_span": span, "text_hash": text_digest(text)}
        )
    content = separator.join(fragments)
    if not content.strip():
        raise ContractError(
            "contract.coverage", "intra", "Contract has no visible work", "turn_contract"
        )
    receipt = {
        "receipt_id": f"{contract.contract_id}:receipt",
        "contract_id": contract.contract_id,
        "reply_event_id": reply_event_id,
        "delivered_units": delivered,
        "issued_requests": requests,
        "span_unit": "unicode_codepoints",
        "reply_hash": text_digest(content),
    }
    return content, receipt, blocks
