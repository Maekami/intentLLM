"""One eligibility rule for grounded progress anchors, not execution dependencies."""


def is_progress_anchor(goal):
    return goal.status == "addressed" or (goal.scope == "current" and goal.status == "active")


def progress_anchor_ids(snapshot):
    return {goal.ref for goal in snapshot.goals if is_progress_anchor(goal)}
