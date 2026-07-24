from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from user_simulator.audit.logger import AuditLogger
from user_simulator.domain.state import EpisodeState


class ConsoleAuditRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def latest(self, logger: AuditLogger) -> None:
        payload = [
            {"event": event.event_type, **event.payload} for event in logger.latest_turn_events
        ]
        self.console.print(Panel(Pretty(payload), title="Turn audit"))

    def state(self, state: EpisodeState) -> None:
        self.console.print(Panel(Pretty(state.model_dump(mode="json")), title="State"))
