"""Three-stage execution graph with producer repair and dependent artifact invalidation."""

import asyncio
import json

from .assembly import assemble, without_optional
from .context import ContextBuilder
from .contracts import output_schema
from .errors import ContractError, RecoveryExhausted
from .generator import validate_units
from .identity import validate_semantic_identities
from .planners import check_plan
from .rendering import render
from .repair_context import repair_excerpt
from .schemas import (
    CurrentRealization,
    GeneratorOutput,
    InterPacket,
    InterPlan,
    IntraPacket,
    IntraPlan,
    JointPlan,
    SemanticProjection,
    SemanticSnapshot,
    none_inter,
)
from .state_delta import updated_goal_ids
from .tracker import merge_packets, normalize_projection


class TurnEngine:
    def __init__(self, runtime, store, registry, interactions, previous, bridge, turn_id, version):
        self.runtime, self.store, self.registry = runtime, store, registry.clone()
        self.interactions, self.previous, self.bridge = interactions, previous, bridge
        self.turn_id, self.version = turn_id, version
        self.settings, self.variant = runtime.settings, runtime.settings.variant
        self.snapshot = None
        self.cache, self.corrections, self.frozen = {}, {}, {}
        self.backups = {}
        self.inter_disabled = self.variant == "no_inter"
        self.dirty_tracker = False
        self.last_contract = None
        self.repair_optional = False

    @property
    def builder(self):
        return ContextBuilder(self.store, self.registry, self.interactions)

    def emit(self, kind, payload):
        self.runtime.recorder.emit(kind, payload)

    def policy_context(self, context):
        return {**context, "policy": self.settings.policy.model_dump()}

    def degrade_inter(self, error):
        self.inter_disabled = True
        self.cache.pop("inter_packet", None)
        self.cache["inter"] = none_inter()
        if "joint" in self.cache:
            self.cache["joint"].inter = none_inter()
        self.runtime.audit["degradations"].append(error.payload())
        self.emit("assistant_gp_optional_pruned", error.payload())

    async def track(self):
        if self.snapshot is not None and not self.dirty_tracker:
            return
        builder = self.builder
        old_snapshot = self.snapshot
        context = self.policy_context(builder.tracker(self.snapshot or self.previous, self.bridge))
        try:
            value = await self.runtime.call(
                "tracker",
                SemanticProjection,
                context,
                output_schema(
                    SemanticProjection,
                    registry=self.registry,
                    evidence_domain=builder.evidence_domain(context),
                ),
                correction=self.corrections.pop("tracker", None),
                validate=lambda raw: normalize_projection(
                    raw, self.registry, self.store, self.turn_id, self.settings.policy
                ),
            )
        except RecoveryExhausted as exc:
            if old_snapshot is not None and self.repair_optional:
                self.degrade_inter(exc.error)
                self.dirty_tracker = False
                return
            raise
        self.snapshot, self.registry, _ = value
        self.snapshot.changed_goal_ids = updated_goal_ids(
            self.snapshot, old_snapshot or self.previous
        )
        self.version += 1
        self.dirty_tracker = False
        self.cache.clear()
        self.backups.clear()
        self.emit(
            "assistant_gp_snapshot_projected",
            {
                "snapshot_version": self.version,
                "snapshot": self.snapshot.model_dump(),
                "interaction_view": self.builder.interactions.view(
                    self.snapshot, self.registry, self.store
                ),
            },
        )

    async def planner(self, role):
        if role in self.cache:
            return self.cache[role]
        if role == "inter" and self.inter_disabled:
            self.cache[role] = none_inter()
            return self.cache[role]
        builder = self.builder
        context = (
            builder.joint(self.snapshot)
            if role == "joint"
            else builder.planner(self.snapshot, role)
        )
        model = {"intra": IntraPlan, "inter": InterPlan, "joint": JointPlan}[role]
        try:
            result = await self.runtime.call(
                role,
                model,
                self.policy_context(context),
                output_schema(
                    model,
                    snapshot=self.snapshot,
                    role=role,
                    variant=self.variant,
                    request_eligibility=builder.interactions.view(
                        self.snapshot, self.registry, self.store
                    ),
                    evidence_domain=builder.evidence_domain(context),
                ),
                correction=self.corrections.pop(role, None),
                validate=lambda plan: check_plan(
                    role, plan, self.snapshot, builder, self.settings.policy, self.variant
                ),
            )
        except RecoveryExhausted as exc:
            if role == "inter":
                self.degrade_inter(exc.error)
                return self.cache["inter"]
            if role == "joint" and self.repair_optional and role in self.backups:
                result = self.backups[role].model_copy(deep=True)
                result.inter = none_inter()
                self.degrade_inter(exc.error)
                self.cache[role] = result
                return result
            raise
        self.cache[role] = result
        return result

    async def parallel(self, *awaitables):
        tasks = [asyncio.create_task(value) for value in awaitables]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def packet(self, role):
        key = role + "_packet"
        if role == "inter" and self.inter_disabled:
            return InterPacket(
                semantic=SemanticSnapshot(goals=[], constraints=[], needs=[]), plan=none_inter()
            )
        if key in self.cache:
            return self.cache[key]
        model = IntraPacket if role == "intra" else InterPacket
        context = self.policy_context(self.builder.packet(role))

        def check(packet):
            validate_semantic_identities(
                packet.semantic,
                self.registry,
                role,
                anchor_ids=self.store.by_id if role == "inter" else (),
            )
            scope = "current" if role == "intra" else "adjacent"
            if any(g.scope != scope for g in packet.semantic.goals):
                raise ContractError("contract.reference", role, "Wrong packet scope", "semantic")
            for group in (
                packet.semantic.goals,
                packet.semantic.constraints,
                packet.semantic.needs,
            ):
                for item in group:
                    for ref in item.evidence_refs:
                        self.store.resolve(ref, role)

        try:
            result = await self.runtime.call(
                role,
                model,
                context,
                output_schema(
                    model,
                    registry=self.registry,
                    anchor_ids=self.store.by_id if role == "inter" else (),
                    evidence_domain=self.builder.evidence_domain(context),
                ),
                validate=check,
                mode="no_tracker",
                correction=self.corrections.pop(role, None),
            )
        except RecoveryExhausted as exc:
            if role == "inter":
                self.degrade_inter(exc.error)
                return await self.packet(role)
            raise
        self.cache[key] = result
        return result

    async def plans(self):
        if self.variant == "no_tracker":
            left, right = await self.parallel(self.packet("intra"), self.packet("inter"))
            snapshot, registry, intra, inter = merge_packets(
                left, right, self.registry, self.store, self.turn_id, self.settings.policy
            )
            snapshot.changed_goal_ids = updated_goal_ids(snapshot, self.snapshot or self.previous)
            if self.snapshot != snapshot:
                self.version += 1
            self.snapshot, self.registry = snapshot, registry
            check_plan("intra", intra, snapshot, self.builder, self.settings.policy, self.variant)
            check_plan("inter", inter, snapshot, self.builder, self.settings.policy, self.variant)
            self.emit(
                "assistant_gp_packets_merged",
                {
                    "snapshot": snapshot.model_dump(),
                    "intra_plan": intra.model_dump(),
                    "inter_plan": inter.model_dump(),
                    "shared_tracker": False,
                },
            )
            return intra, inter
        await self.track()
        if self.variant == "joint":
            result = await self.planner("joint")
            return result.intra, none_inter() if self.inter_disabled else result.inter
        if self.variant == "no_intra":
            return None, await self.planner("inter")
        if self.variant == "no_inter":
            return await self.planner("intra"), none_inter()
        return await self.parallel(self.planner("intra"), self.planner("inter"))

    def frozen_for(self, contract):
        return {
            unit.unit_id: self.frozen[unit.unit_id][1]
            for unit in contract.deliveries
            if unit.unit_id in self.frozen
            and self.frozen[unit.unit_id][0] == self.builder.unit_signature(self.snapshot, unit)
        }

    def remember(self, result, contract):
        by_id = {unit.unit_id: unit for unit in contract.deliveries}
        counts = {}
        for unit in result.units:
            counts[unit.unit_id] = counts.get(unit.unit_id, 0) + 1
        issue_ids = {issue.unit_id for issue in result.issues}
        for body in result.units:
            if (
                body.unit_id in by_id
                and body.text.strip()
                and counts[body.unit_id] == 1
                and body.unit_id not in issue_ids
            ):
                signature = self.builder.unit_signature(self.snapshot, by_id[body.unit_id])
                if self.frozen.get(body.unit_id, (None,))[0] != signature:
                    self.frozen[body.unit_id] = (signature, body.text)

    def issue_error(self, issue, contract):
        unit = next(u for u in contract.deliveries if u.unit_id == issue.unit_id)
        owner = (
            "joint"
            if self.variant == "joint"
            else "generator"
            if self.variant == "no_intra" and unit.source != "inter"
            else "inter"
            if unit.source == "inter"
            else "intra"
        )
        # Reference provenance determines ownership; no extra semantic routing model.
        valid_semantic = {f"goal:{g.ref}" for g in self.snapshot.goals}
        valid_semantic |= {f"need:{n.ref}" for n in self.snapshot.needs}
        valid_semantic |= {f"constraint:{c.ref}" for c in self.snapshot.constraints}
        allowed_goals = {unit.goal_id}
        scoped = self.builder.semantic_view(
            self.snapshot, allowed_goals, need_ids=unit.required_need_ids
        )
        semantic_ids = {f"goal:{g['ref']}" for g in scoped["goals"]}
        semantic_ids |= {f"need:{n['ref']}" for n in scoped["needs"]}
        semantic_ids |= {f"constraint:{c['ref']}" for c in scoped["constraints"]}
        if not issue.input_refs:
            return ContractError(
                "contract.reference",
                "generator",
                "Issue must cite an input owner reference",
                affected_units=[unit.unit_id],
            )
        for ref in issue.input_refs:
            if ref.startswith("unit:") and ref != f"unit:{unit.unit_id}":
                return ContractError(
                    "contract.reference",
                    "generator",
                    "Issue cites another unit",
                    affected_units=[unit.unit_id],
                )
            if ref in valid_semantic and ref not in semantic_ids:
                return ContractError(
                    "contract.reference",
                    "generator",
                    "Issue cites unrelated state",
                    affected_units=[unit.unit_id],
                )
            if ref not in semantic_ids and ref != f"unit:{unit.unit_id}":
                return ContractError(
                    "contract.reference",
                    "generator",
                    "Unknown issue input reference",
                    affected_units=[unit.unit_id],
                )
        if self.variant != "no_tracker" and any(ref in semantic_ids for ref in issue.input_refs):
            owner = "tracker"
        return ContractError(
            issue.code,
            owner,
            issue.detail,
            field_path=issue.input_refs[0],
            affected_units=[unit.unit_id],
        )

    async def realize(self, contract):
        builder = self.builder
        self.last_contract = contract

        def context_factory():
            frozen = self.frozen_for(contract)
            pending = [u for u in contract.deliveries if u.unit_id not in frozen]
            context = builder.generator(
                self.snapshot,
                contract,
                pending,
                frozen=[{"unit_id": key, "text": text} for key, text in frozen.items()],
            )
            context["input_reference_ids"] = [
                *[f"unit:{u.unit_id}" for u in pending],
                *[f"goal:{g['ref']}" for g in context["goals"]],
                *[f"need:{n['ref']}" for n in context["needs"]],
                *[f"constraint:{c['ref']}" for c in context["constraints"]],
            ]
            return context, output_schema(
                GeneratorOutput,
                unit_ids=[u.unit_id for u in pending],
                input_reference_ids=context["input_reference_ids"],
            )

        def check(result):
            pending = {u.unit_id for u in contract.deliveries} - self.frozen_for(contract).keys()
            try:
                validate_units(result, contract, pending)
            finally:
                # Only individually well-formed authorized bodies can be frozen.
                self.remember(result, contract)

        result = await self.runtime.call(
            "generator",
            GeneratorOutput,
            {},
            {},
            validate=check,
            correction=self.corrections.pop("generator", None),
            context_factory=context_factory,
        )
        if result.issues:
            raise self.issue_error(result.issues[0], contract)
        return contract, self.frozen_for(contract)

    async def realize_current(self, inter):
        # Explicit no_intra owner map: only this mode lets Generator choose current actions.
        prepared, _ = assemble(
            self.snapshot,
            None,
            inter,
            self.settings.policy,
            version=self.version,
            turn_id=self.turn_id,
        )
        ids = {g.ref for g in self.snapshot.goals_for("current")}
        ids.update(u.goal_id for u in prepared.deliveries)
        needs = {n.ref: n for n in self.snapshot.needs}
        ids.update(needs[r.need_id].goal_id for r in prepared.requests)
        dependencies = {ref for unit in prepared.deliveries for ref in unit.required_need_ids}
        context = self.builder.semantic_view(self.snapshot, ids, need_ids=dependencies)
        context.update(
            current_goal_ids=[g.ref for g in self.snapshot.goals_for("current")],
            request_eligibility=self.interactions.view(self.snapshot, self.registry, self.store),
            inter_work=prepared.model_dump(),
            policy=self.settings.policy.model_dump(),
        )
        extra = [ref for unit in prepared.deliveries for ref in unit.material_refs]
        context["inter_materials"] = self.store.project(extra)
        accepted = {}

        def check(result):
            check_plan(
                "generator",
                result.intra,
                self.snapshot,
                self.builder,
                self.settings.policy,
                self.variant,
            )
            contract, deferred = assemble(
                self.snapshot,
                result.intra,
                inter,
                self.settings.policy,
                version=self.version,
                turn_id=self.turn_id,
            )
            self.last_contract = contract
            allowed_extra = {
                f"boundary:{item.goal_id}" for item in result.intra.items if not item.deliveries
            }
            chosen_ids = {u.unit_id for u in contract.deliveries}
            extras = {u.unit_id for u in [*result.units, *result.issues]} - chosen_ids
            if not extras <= allowed_extra:
                raise ContractError(
                    "contract.reference",
                    "generator",
                    "Unapproved no_intra body IDs",
                    affected_units=sorted(extras),
                )
            filtered = GeneratorOutput(
                units=[u for u in result.units if u.unit_id in chosen_ids],
                issues=[u for u in result.issues if u.unit_id in chosen_ids],
            )
            try:
                validate_units(filtered, contract, chosen_ids)
            finally:
                self.remember(filtered, contract)
            accepted.update(
                contract=contract,
                result=filtered,
                deferred=deferred,
                discarded_body_unit_ids=sorted(extras),
            )
            self.cache["current"] = result.intra.model_copy(deep=True)

        await self.runtime.call(
            "generator",
            CurrentRealization,
            context,
            output_schema(
                CurrentRealization,
                snapshot=self.snapshot,
                request_eligibility=context["request_eligibility"],
                evidence_domain=self.builder.evidence_domain(context),
            ),
            validate=check,
            correction=self.corrections.pop("generator", None),
            mode="no_intra",
        )
        self.emit(
            "assistant_gp_contract_compiled",
            {
                "contract": accepted["contract"].model_dump(),
                "deferred": accepted["deferred"],
                "compilation_phase": "after_current_realization",
                "discarded_body_unit_ids": accepted["discarded_body_unit_ids"],
            },
        )
        result, contract = accepted["result"], accepted["contract"]
        if result.issues:
            raise self.issue_error(result.issues[0], contract)
        return contract, self.frozen_for(contract)

    def optional_error(self, error):
        if error.owner == "inter":
            return True
        if self.last_contract is None:
            return False
        optional = {u.unit_id for u in self.last_contract.deliveries if u.criticality == "optional"}
        if error.affected_units and set(error.affected_units) <= optional:
            return True
        if error.owner == "generator" and optional:
            if error.code == "context.capacity":
                return True
            required = {
                u.unit_id for u in self.last_contract.deliveries if u.criticality == "required"
            }
            return required <= self.frozen_for(self.last_contract).keys()
        return False

    def invalidate(self, error):
        role = error.owner
        artifact = (
            self.snapshot
            if role == "tracker"
            else self.cache.get(role + "_packet" if self.variant == "no_tracker" else role)
        )
        if artifact is not None:
            self.backups[role] = artifact.model_copy(deep=True)
            correction = self.corrections.get(role)
            if correction is not None:
                correction["previous_owned_output_excerpt"] = repair_excerpt(
                    json.dumps(artifact.model_dump(), ensure_ascii=False),
                    correction,
                    self.settings.recovery.correction_max_characters,
                )
        if role == "tracker":
            self.dirty_tracker = True
        elif role == "generator":
            self.cache.pop("current", None)
        elif role in {"intra", "inter", "joint"}:
            self.cache.pop(role, None)
            self.cache.pop(role + "_packet", None)
        self.emit(
            "assistant_gp_dependencies_invalidated",
            {
                **error.payload(),
                "preserved_producers": sorted(self.cache),
                "snapshot_version": self.version,
            },
        )

    async def execute(self, reply_event_id):
        while True:
            try:
                intra, inter = await self.plans()
                if self.variant == "no_intra" and "current" not in self.cache:
                    contract, bodies = await self.realize_current(inter)
                else:
                    if self.variant == "no_intra":
                        intra = self.cache["current"]
                    contract, deferred = assemble(
                        self.snapshot,
                        intra,
                        inter,
                        self.settings.policy,
                        version=self.version,
                        turn_id=self.turn_id,
                    )
                    if self.inter_disabled:
                        contract = without_optional(contract)
                    self.last_contract = contract
                    self.emit(
                        "assistant_gp_contract_compiled",
                        {
                            "snapshot_version": self.version,
                            "contract": contract.model_dump(),
                            "deferred": deferred,
                            "compilation_phase": "before_realization",
                        },
                    )
                    # Realization is work-driven: a request-only contract is already
                    # complete, as are unchanged bodies frozen during local repair.
                    # Do not ask a model to invent output for an empty work order.
                    ready = self.frozen_for(contract)
                    if len(ready) == len(contract.deliveries):
                        bodies = ready
                        self.emit(
                            "assistant_generator_skipped",
                            {
                                "contract_id": contract.contract_id,
                                "reason": "no_pending_body_units",
                                "frozen_unit_ids": sorted(ready),
                                "request_ids": [r.need_id for r in contract.requests],
                                "model_called": False,
                            },
                        )
                    else:
                        contract, bodies = await self.realize(contract)
                content, receipt, blocks = render(
                    contract, bodies, reply_event_id, self.settings.realization.separator
                )
                self.emit(
                    "assistant_gp_reply_composed",
                    {
                        "contract_id": contract.contract_id,
                        "final_reply": content,
                        "body_units": bodies,
                        "prepared_receipt": receipt,
                        "committed": False,
                    },
                )
                return content, self.snapshot, self.registry, receipt, blocks
            except RecoveryExhausted as exc:
                if self.optional_error(exc.error) and not self.inter_disabled:
                    self.degrade_inter(exc.error)
                    continue
                raise
            except ContractError as error:
                error.artifact_id = (
                    self.last_contract.contract_id
                    if self.last_contract is not None
                    else f"{self.turn_id}:{error.owner}:candidate"
                )
                error.input_version = f"snapshot:{self.version}"
                optional = self.optional_error(error)
                try:
                    self.corrections[error.owner] = self.runtime.repair(error)
                except RecoveryExhausted:
                    if optional and not self.inter_disabled:
                        self.degrade_inter(error)
                        continue
                    raise
                self.repair_optional = optional
                self.invalidate(error)
