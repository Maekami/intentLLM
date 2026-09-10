"""Synthetic GP fixtures; never replay benchmark examples into prompts."""

import asyncio
import copy
import json
from pathlib import Path

from assistant.config import GoalProgressionSettings, load_model_profile
from assistant.goal_progression import GoalProgressionBaseline, GoalProgressionSession
from assistant.goal_progression.schemas import none_inter
from assistant.llm.base import GeneratedResponse

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_tracker", "no_intra", "no_inter", "joint", "no_anticipate")


def profile(variant="full", **settings):
    suffix = "" if variant == "full" else "_" + variant
    p = load_model_profile(str(ROOT / "configs/models" / f"qwen_3_6_27b_gp{suffix}.yaml"))
    raw = p.goal_progression.model_dump()
    raw["recovery"].update(
        initial_backoff_seconds=0, maximum_backoff_seconds=0, call_timeout_seconds=1
    )
    raw.update(settings)
    p.goal_progression = GoalProgressionSettings.model_validate(raw)
    return p


def ref(event="m1"):
    return {"event_id": event, "block_id": None}


def goal(gid="new:current", scope="current", event="m1"):
    return {
        "ref": gid,
        "scope": scope,
        "basis": "explicit" if scope == "current" else "inferred",
        "status": "active",
        "description": "Create a synthetic card"
        if scope == "current"
        else "Provide a distinct storage label",
        "remaining_work": "Provide the specified artifact",
        "anchor_goal_ids": [] if scope == "current" else ["new:current"],
        "material_refs": [ref(event)],
        "evidence_refs": [ref(event)],
    }


def need(nid="new:shape", gid="new:current", event="m1"):
    return {
        "ref": nid,
        "goal_id": gid,
        "description": "Card edge shape",
        "status": "unknown",
        "answer_refs": [],
        "renewed_authorization_ref": None,
        "evidence_refs": [ref(event)],
    }


def semantic(*, adjacent=True, needs=False, event="m1", current="new:current"):
    current_goal = goal(current, event=event)
    goals = [current_goal]
    if adjacent:
        other = goal("new:adjacent", "adjacent", event)
        other["anchor_goal_ids"] = [current]
        goals.append(other)
    return {
        "goals": goals,
        "constraints": [],
        "needs": [need(gid=current, event=event)] if needs else [],
    }


def delivery(gid, target="Produce the supported card body"):
    return {"goal_id": gid, "target": target, "required_need_ids": [], "material_refs": []}


def current_plan(context, *, ask=False):
    items = []
    elig = context.get("request_eligibility", {})
    for g in context["goals"]:
        if g["scope"] != "current" or g["status"] != "active":
            continue
        candidate = next((n for n in context["needs"] if n["goal_id"] == g["ref"]), None)
        request = None
        blocked = []
        action, deliveries = "advance", [delivery(g["ref"])]
        if ask and candidate:
            if elig.get(candidate["ref"], {}).get("request_eligible", True):
                action = "clarify"
                request = {
                    "need_id": candidate["ref"],
                    "question_text": "Which edge shape should the card use?",
                }
            else:
                action, deliveries, blocked = None, [], [candidate["ref"]]
        items.append(
            {
                "goal_id": g["ref"],
                "action": action,
                "deliveries": deliveries,
                "request": request,
                "blocked_by": blocked,
            }
        )
    return {"items": items}


class Client:
    def __init__(self, script=None, *, ask=False, anticipate=False):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.ask, self.anticipate = ask, anticipate
        self.calls, self.tokenized = [], []
        self.active_planners = 0
        self.max_parallel_planners = 0

    async def generate(self, *, messages, generation, response_schema=None):
        payload = json.loads(messages[1]["content"])
        role, mode, context = payload["role"], payload["mode"], payload["context"]
        call = {
            "role": role,
            "mode": mode,
            "context": copy.deepcopy(context),
            "messages": copy.deepcopy(messages),
            "schema": response_schema,
            "generation": generation.model_dump(),
            "repair": payload.get("repair"),
        }
        self.calls.append(call)
        if role in {"intra", "inter"}:
            self.active_planners += 1
            self.max_parallel_planners = max(self.max_parallel_planners, self.active_planners)
        try:
            await asyncio.sleep(0.001)
            scripted = self.script.get(role, [])
            if scripted:
                value = scripted.pop(0)
                if isinstance(value, BaseException):
                    raise value
                value = value(call) if callable(value) else value
            else:
                value = self.default(call)
            # Existing fixtures describe canonical snapshots. Convert their
            # layout to the Tracker wire view without fixing values or links.
            if role == "tracker" and isinstance(value, dict) and "goals" in value:
                value = copy.deepcopy(value)
                goals = value.pop("goals")
                value.pop("changed_goal_ids", None)
                value["current_goals"] = [g for g in goals if g.get("scope") != "adjacent"]
                value["adjacent_goals"] = [g for g in goals if g.get("scope") == "adjacent"]
            raw = value if isinstance(value, str) else json.dumps(value)
            return GeneratedResponse(
                raw,
                reasoning_content="SECRET_REASONING",
                metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "transport_retry_count": 0,
                    "reasoning_content": "SECRET_REASONING",
                },
            )
        finally:
            if role in {"intra", "inter"}:
                self.active_planners -= 1

    def default(self, call):
        role, mode, context = call["role"], call["mode"], call["context"]
        if role == "tracker":
            previous = context["previous_snapshot"]
            if previous:
                return previous
            return semantic(needs=self.ask)
        if mode == "no_tracker":
            value = semantic(adjacent=role == "inter", needs=self.ask and role == "intra")
            if role == "inter":
                value["goals"] = [g for g in value["goals"] if g["scope"] == "adjacent"]
                for g in value["goals"]:
                    g["anchor_goal_ids"] = ["m1"]
            plan = (
                current_plan(value, ask=self.ask) if role == "intra" else none_inter().model_dump()
            )
            return {"semantic": value, "plan": plan}
        if role == "intra":
            return current_plan(context, ask=self.ask)
        if role == "inter":
            adjacent = [g for g in context["goals"] if g["scope"] == "adjacent"]
            if self.anticipate and adjacent:
                gid = adjacent[0]["ref"]
                return {
                    "action": "anticipate",
                    "goal_id": gid,
                    "deliveries": [delivery(gid, "Provide the storage label")],
                    "request": None,
                }
            return none_inter().model_dump()
        if role == "joint":
            return {
                "intra": current_plan(context, ask=self.ask),
                "inter": none_inter().model_dump(),
            }
        if mode == "no_intra":
            plan = current_plan(context, ask=self.ask)
            units = [
                {"unit_id": f"intra:{item['goal_id']}:{i}", "text": "A complete synthetic card."}
                for item in plan["items"]
                for i, _ in enumerate(item["deliveries"])
            ]
            units += [
                {"unit_id": f"boundary:{item['goal_id']}", "text": "The input is unavailable."}
                for item in plan["items"]
                if not item["deliveries"]
            ]
            units += [
                {"unit_id": u["unit_id"], "text": "A distinct storage label."}
                for u in context["inter_work"]["deliveries"]
            ]
            return {"intra": plan, "units": units, "issues": []}
        return {
            "units": [
                {"unit_id": u["unit_id"], "text": "A complete synthetic card."}
                for u in context["turn_contract"]["deliveries"]
            ],
            "issues": [],
        }

    async def count_visible_tokens(self, content):
        self.tokenized.append(content)
        return {"visible_response_tokens": len(content.split()), "visible_token_source": "fixture"}


def session(variant="full", *, client=None, **settings):
    p = profile(variant, **settings)
    return GoalProgressionSession(
        client or Client(), p.generation["assistant"], GoalProgressionBaseline(), profile=p
    )
