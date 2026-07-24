import random
from typing import Any

from user_simulator.audit.logger import AuditLogger
from user_simulator.controller.base import Controller
from user_simulator.domain.dag import Sample
from user_simulator.domain.enums import Difficulty, SatisfactionLevel
from user_simulator.domain.messages import ChatMessage
from user_simulator.domain.results import TurnResult
from user_simulator.domain.state import EpisodeState
from user_simulator.engine.termination import should_terminate
from user_simulator.engine.transitions import (
    all_exposed_satisfied,
    apply_satisfaction_updates,
    normalize_controller_result,
    unresolved_queue,
)
from user_simulator.exceptions import (
    EpisodeTurnLimitError,
    InvalidTerminalStateError,
)
from user_simulator.graph.navigator import GraphNavigator
from user_simulator.policy.realization import RealizationPolicy
from user_simulator.policy.selection import NodeSelectionPolicy
from user_simulator.realizer.base import UserRealizer
from user_simulator.satisfaction.base import SatisfactionUpdater


class Episode:
    """Orchestrates one episode while delegating every replaceable decision."""

    def __init__(
        self,
        *,
        sample: Sample,
        difficulty: Difficulty,
        seed: int,
        controller: Controller,
        satisfaction_updater: SatisfactionUpdater,
        selection_policy: NodeSelectionPolicy,
        realization_policy: RealizationPolicy,
        user_realizer: UserRealizer,
        audit_logger: AuditLogger | None = None,
        max_turns: int = 20,
        monotonic_satisfaction: bool = True,
        auto_expose_backbone_on_empty_queue: bool = True,
    ) -> None:
        self.sample = sample
        self.navigator = GraphNavigator(sample)
        self.controller = controller
        self.satisfaction_updater = satisfaction_updater
        self.selection_policy = selection_policy
        self.realization_policy = realization_policy
        self.user_realizer = user_realizer
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.monotonic_satisfaction = monotonic_satisfaction
        self.auto_expose_backbone_on_empty_queue = auto_expose_backbone_on_empty_queue
        self.state = EpisodeState(
            sample_id=sample.sample_id,
            difficulty=difficulty,
            current_frontier="N1",
            exposed_nodes=["N1"],
            satisfaction={"N1": SatisfactionLevel.UNSATISFIED},
            random_seed=seed,
        )
        self.audit = audit_logger or AuditLogger(sample.sample_id, enabled=False)
        self.started = False

    async def start(self) -> TurnResult:
        if self.started:
            raise RuntimeError("episode has already started")
        self.started = True
        self.audit.log(
            "episode_started",
            0,
            {
                "difficulty": self.state.difficulty.value,
                "seed": self.state.random_seed,
                "initial_frontier": "N1",
            },
        )
        mode = self.realization_policy.choose_mode(self.state.difficulty, self.rng)
        self.audit.log("realization_mode_selected", 0, {"mode": mode.value})
        self.audit.log("user_generation_requested", 0, {"selected_nodes": ["N1"], "initial": True})
        try:
            generation = await self.user_realizer.generate(
                selected_nodes=[self.navigator.node("N1")],
                unselected_unresolved_nodes=[],
                satisfaction=self.state.satisfaction,
                history=[],
                latest_assistant_response=None,
                mode=mode,
                sample=self.sample,
            )
            self.audit.log(
                "user_generation_raw_result",
                0,
                {
                    "raw_response": generation.model_dump(mode="json"),
                    "llm_call": self._llm_metadata(self.user_realizer),
                },
            )
            message = ChatMessage(role="user", content=generation.user_message)
            self.state.conversation_history.append(message)
            self.audit.log_message(message, 0)
            self.audit.log(
                "initial_user_generated",
                0,
                {"user_message": generation.user_message, "mode": mode.value},
            )
            return TurnResult(
                terminal=False,
                user_message=generation.user_message,
                state=self.state.model_copy(deep=True),
            )
        except Exception as exc:
            self._fail(exc)
            raise

    async def submit_assistant(self, response: str) -> TurnResult:
        if not self.started:
            raise RuntimeError("call start() before submitting an assistant response")
        if self.state.terminated:
            raise RuntimeError("episode is already terminated")
        if not response.strip():
            raise ValueError("assistant response must be non-empty")
        if self.state.turn_index >= self.max_turns:
            exc = EpisodeTurnLimitError(
                f"episode exceeded maximum of {self.max_turns} assistant turns"
            )
            self._fail(exc)
            raise exc

        self.state.turn_index += 1
        turn = self.state.turn_index
        latest = ChatMessage(role="assistant", content=response)
        self.state.conversation_history.append(latest)
        self.audit.log_message(latest, turn)
        self.audit.log("assistant_message_received", turn, {"assistant_message": response})
        frontier_before = self.state.current_frontier
        satisfaction_before = dict(self.state.satisfaction)
        try:
            candidate_ids, has_end_edge = self.navigator.outgoing(frontier_before)
            candidates = [self.navigator.node(item) for item in candidate_ids]
            self.audit.log(
                "controller_requested",
                turn,
                {
                    "frontier": frontier_before,
                    "candidates": candidate_ids,
                    "has_end_edge": has_end_edge,
                },
            )
            raw_controller = await self.controller.decide(
                history=self.state.conversation_history,
                latest_assistant_response=response,
                state=self.state,
                candidates=candidates,
                has_end_edge=has_end_edge,
            )
            self.audit.log(
                "controller_raw_result",
                turn,
                {
                    "raw_response": raw_controller.model_dump(mode="json"),
                    "llm_call": self._llm_metadata(self.controller),
                },
            )
            normalized = normalize_controller_result(raw_controller, candidate_ids, has_end_edge)
            self.audit.log(
                "controller_prefix_normalized",
                turn,
                normalized.model_dump(mode="json"),
            )

            for node_id in normalized.newly_exposed:
                if node_id not in self.state.exposed_nodes:
                    self.state.exposed_nodes.append(node_id)
                    self.state.satisfaction[node_id] = SatisfactionLevel.UNSATISFIED
            if normalized.newly_exposed:
                self.state.current_frontier = normalized.newly_exposed[-1]
            self.state.end_reachable = normalized.end_reachable
            self.audit.log(
                "nodes_exposed",
                turn,
                {
                    "newly_exposed": normalized.newly_exposed,
                    "frontier_before": frontier_before,
                    "frontier_after": self.state.current_frontier,
                    "end_reachable": self.state.end_reachable,
                },
            )

            self.audit.log(
                "satisfaction_requested",
                turn,
                {"exposed_nodes": self.state.exposed_nodes},
            )
            raw_satisfaction = await self.satisfaction_updater.update(
                history=self.state.conversation_history,
                latest_assistant_response=response,
                state=self.state,
                exposed_nodes=[self.navigator.node(item) for item in self.state.exposed_nodes],
            )
            self.audit.log(
                "satisfaction_raw_result",
                turn,
                {
                    "raw_response": raw_satisfaction.model_dump(mode="json"),
                    "llm_call": self._llm_metadata(self.satisfaction_updater),
                },
            )
            applied, satisfaction_violations = apply_satisfaction_updates(
                self.state.satisfaction,
                raw_satisfaction,
                self.state.exposed_nodes,
                monotonic=self.monotonic_satisfaction,
            )
            self.state.satisfaction = applied
            self.audit.log(
                "satisfaction_applied",
                turn,
                {
                    "before": {key: value.value for key, value in satisfaction_before.items()},
                    "after": {key: value.value for key, value in applied.items()},
                    "violations": satisfaction_violations,
                },
            )

            all_satisfied = all_exposed_satisfied(self.state.exposed_nodes, self.state.satisfaction)
            terminating = should_terminate(self.state)
            self.audit.log(
                "termination_checked",
                turn,
                {
                    "end_reachable": self.state.end_reachable,
                    "all_exposed_satisfied": all_satisfied,
                    "will_terminate": terminating,
                },
            )
            if terminating:
                self.state.terminated = True
                self.audit.log(
                    "episode_terminated",
                    turn,
                    {"final_frontier": self.state.current_frontier},
                )
                self.audit.save_state(self.state)
                return TurnResult(
                    terminal=True, state=self.state.model_copy(deep=True), audit=self._turn_audit()
                )

            if not self.state.end_reachable and all_satisfied:
                if not self.auto_expose_backbone_on_empty_queue:
                    raise InvalidTerminalStateError(
                        "all exposed nodes satisfied while END is unreachable"
                    )
                next_node = self.navigator.next_backbone(self.state.current_frontier)
                if next_node is None:
                    raise InvalidTerminalStateError(
                        "final intent node is satisfied but END is not reachable"
                    )
                self.state.exposed_nodes.append(next_node)
                self.state.satisfaction[next_node] = SatisfactionLevel.UNSATISFIED
                self.state.current_frontier = next_node
                self.audit.log("backbone_node_auto_exposed", turn, {"node_id": next_node})

            queue = unresolved_queue(self.state.exposed_nodes, self.state.satisfaction)
            self.audit.log("unresolved_queue_built", turn, {"queue": queue})
            if not queue:
                raise InvalidTerminalStateError("unresolved queue is empty after natural exposure")
            selection = self.selection_policy.select(queue, self.state.difficulty, self.rng)
            selected_ids = selection.selected_nodes
            self.audit.log(
                "nodes_selected",
                turn,
                {
                    "difficulty": self.state.difficulty.value,
                    "selected_nodes": selected_ids,
                },
            )
            mode = self.realization_policy.choose_mode(self.state.difficulty, self.rng)
            self.audit.log("realization_mode_selected", turn, {"mode": mode.value})
            unselected = [item for item in queue if item not in selected_ids]
            self.audit.log(
                "user_generation_requested",
                turn,
                {
                    "selected_nodes": selected_ids,
                    "unselected_unresolved_nodes": unselected,
                    "mode": mode.value,
                },
            )
            generation = await self.user_realizer.generate(
                selected_nodes=[self.navigator.node(item) for item in selected_ids],
                unselected_unresolved_nodes=[self.navigator.node(item) for item in unselected],
                satisfaction=self.state.satisfaction,
                history=self.state.conversation_history,
                latest_assistant_response=response,
                mode=mode,
                sample=self.sample,
            )
            self.audit.log(
                "user_generation_raw_result",
                turn,
                {
                    "raw_response": generation.model_dump(mode="json"),
                    "llm_call": self._llm_metadata(self.user_realizer),
                },
            )
            message = ChatMessage(role="user", content=generation.user_message)
            self.state.conversation_history.append(message)
            self.audit.log_message(message, turn)
            self.audit.log(
                "user_message_generated",
                turn,
                {
                    "user_message": generation.user_message,
                    "selected_nodes": selected_ids,
                },
            )
            return TurnResult(
                terminal=False,
                user_message=generation.user_message,
                state=self.state.model_copy(deep=True),
                audit=self._turn_audit(),
            )
        except Exception as exc:
            self._fail(exc)
            raise

    def _turn_audit(self) -> dict[str, Any]:
        return {
            "events": [event.model_dump(mode="json") for event in self.audit.latest_turn_events]
        }

    @staticmethod
    def _llm_metadata(component: Any) -> dict[str, Any]:
        direct = getattr(component, "last_call_metadata", None)
        if isinstance(direct, dict) and direct:
            return dict(direct)
        client = getattr(component, "client", None)
        metadata = getattr(client, "last_call_metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}

    def _fail(self, exc: Exception) -> None:
        self.audit.log(
            "episode_failed",
            self.state.turn_index,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )
        self.audit.save_state(self.state)
