import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

from user_simulator.domain.dag import Sample
from user_simulator.exceptions import DatasetValidationError

NODE_RE = re.compile(r"^N([1-9]\d*)$")


def node_index(node_id: str) -> int:
    match = NODE_RE.fullmatch(node_id)
    if not match:
        raise ValueError(f"invalid intent node ID: {node_id}")
    return int(match.group(1))


@dataclass(frozen=True)
class ValidationIssue:
    sample_id: str
    message: str
    severity: str = "error"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [item for item in self.issues if item.severity == "warning"]

    def raise_for_errors(self) -> None:
        if self.errors:
            detail = "\n".join(f"{x.sample_id}: {x.message}" for x in self.errors)
            raise DatasetValidationError(detail)


class DatasetValidator:
    def __init__(self, *, strict_prefix_closure: bool = False) -> None:
        self.strict_prefix_closure = strict_prefix_closure

    def validate_sample(self, sample: Sample) -> ValidationReport:
        report = ValidationReport()
        sid = sample.sample_id
        nodes = sample.reason_dag.nodes
        edges = sample.reason_dag.edges
        node_ids = [n.node_id for n in nodes]
        intent_ids = [n.node_id for n in nodes if n.node_type == "intent"]

        def error(message: str) -> None:
            report.issues.append(ValidationIssue(sid, message))

        if len(node_ids) != len(set(node_ids)):
            error("duplicate node IDs")
        if node_ids.count("END") != 1:
            error("must contain exactly one END node")
        end_nodes = [n for n in nodes if n.node_id == "END"]
        if end_nodes and end_nodes[0].node_type != "terminal":
            error("END must have terminal node_type")
        if any(n.node_type == "terminal" and n.node_id != "END" for n in nodes):
            error("END must be the only terminal node")

        expected = [f"N{i}" for i in range(1, len(intent_ids) + 1)]
        if sorted(intent_ids, key=_safe_index) != expected:
            error(f"intent IDs must be contiguous N1...Nk; found {intent_ids}")

        edge_ids = [e.edge_id for e in edges]
        endpoint_pairs = [(e.source, e.target) for e in edges]
        if len(edge_ids) != len(set(edge_ids)):
            error("duplicate edge IDs")
        if len(endpoint_pairs) != len(set(endpoint_pairs)):
            error("duplicate directed edges")

        known = set(node_ids)
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.source not in known or edge.target not in known:
                error(f"edge {edge.edge_id} has unknown endpoint")
                continue
            if edge.source == edge.target:
                error(f"edge {edge.edge_id} is a self-loop")
            if edge.source == "END":
                error("END has an outgoing edge")
            if edge.target != "END":
                try:
                    if edge.source == "END" or node_index(edge.target) <= node_index(edge.source):
                        error(f"edge {edge.edge_id} does not point forward")
                except ValueError:
                    error(f"edge {edge.edge_id} uses malformed node ID")
            outgoing[edge.source].append(edge.target)

        for left, right in pairwise(expected):
            if (left, right) not in endpoint_pairs:
                error(f"missing mandatory backbone edge {left}->{right}")
        if expected and (expected[-1], "END") not in endpoint_pairs:
            error(f"missing mandatory backbone edge {expected[-1]}->END")

        for node_id in expected:
            if not _can_reach_end(node_id, outgoing):
                error(f"{node_id} cannot reach END")

        for source, targets in outgoing.items():
            if source == "END":
                continue
            intent_targets = sorted(
                (target for target in targets if target != "END"), key=_safe_index
            )
            if intent_targets:
                start = node_index(source) + 1
                highest = node_index(intent_targets[-1])
                required = {f"N{i}" for i in range(start, highest + 1)}
                missing = sorted(required - set(intent_targets), key=_safe_index)
                if missing:
                    severity = "error" if self.strict_prefix_closure else "warning"
                    report.issues.append(
                        ValidationIssue(
                            sid,
                            f"outgoing targets from {source} violate prefix closure; "
                            f"missing {missing}",
                            severity,
                        )
                    )
        return report

    def validate_samples(self, samples: list[Sample]) -> ValidationReport:
        combined = ValidationReport()
        ids = [sample.sample_id for sample in samples]
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        if duplicates:
            combined.issues.append(
                ValidationIssue("<dataset>", f"duplicate sample IDs: {duplicates}")
            )
        for sample in samples:
            combined.issues.extend(self.validate_sample(sample).issues)
        return combined


def _safe_index(node_id: str) -> int:
    try:
        return node_index(node_id)
    except ValueError:
        return 10**12


def _can_reach_end(start: str, outgoing: dict[str, list[str]]) -> bool:
    stack = [start]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == "END":
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(outgoing.get(current, []))
    return False
