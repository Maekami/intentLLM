"""Project session interface for the frozen R1 agent."""
from assistant.baselines.base import BaseBaseline
from assistant.config import EnvironmentSettings
from assistant.domain.messages import ChatMessage
from assistant.session import AssistantSession

from .agent import StaticAgent
from .client import ProfileClient
from .token_count import configured_counter


class R1Baseline(BaseBaseline):
    name = 'static_r1'


class R1Session(AssistantSession):
    def __init__(self, *, profile, environment=None, client=None, trace_dir=None):
        self.client = client or ProfileClient(
            profile, environment or EnvironmentSettings(), token_counter=configured_counter())
        self.generation = profile.generation['assistant']
        self.baseline = R1Baseline()
        self.engine = StaticAgent(client=self.client, trace_dir=trace_dir)

    @property
    def history(self):
        return tuple(ChatMessage(**message) for message in self.engine.history)

    @property
    def last_call_metadata(self):
        metadata = dict(self.engine.last_call_metadata)
        # Explicit absence prevents metrics from substituting private role tokens.
        metadata.setdefault("visible_response_tokens", None)
        return metadata

    @property
    def last_audit(self):
        return getattr(self.engine, 'last_audit', {})

    async def respond(self, user_message, *, audit_sink=None):
        try:
            return await self.engine.respond(user_message)
        finally:
            if audit_sink is not None:
                audit_sink('r1_turn', self.last_audit)

    def reset(self):
        self.engine = StaticAgent(client=self.client, trace_dir=self.engine.trace_dir)

    def replace_history(self, messages):
        self.reset()
        self.engine.history = [dict(role=m.role, content=m.content) for m in messages]
        self.engine.turn = sum(m.role == 'assistant' for m in messages)

    async def close(self):
        await self.client.close()
