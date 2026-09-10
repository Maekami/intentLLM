"""Runtime-owned structural deltas, not a second semantic change classifier."""


def updated_goal_ids(snapshot, previous):
    def footprints(value):
        if value is None:
            return {}
        return {
            goal.ref: {
                "goal": goal.model_dump(),
                "constraints": sorted(
                    [c.model_dump() for c in value.constraints if goal.ref in c.applies_to],
                    key=lambda item: item["ref"],
                ),
                "needs": sorted(
                    [n.model_dump() for n in value.needs if n.goal_id == goal.ref],
                    key=lambda item: item["ref"],
                ),
            }
            for goal in value.goals
        }

    old, new = footprints(previous), footprints(snapshot)
    # Rephrasing may count as an update: this records changed declared inputs,
    # not semantic equivalence or hidden goal completion. Retired IDs are not
    # supplied as actionable references in a snapshot where they no longer exist.
    return [ref for ref, value in new.items() if old.get(ref) != value]
