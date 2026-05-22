# output/event_bus.py
# ─────────────────────────────────────────────────────────────
# Lightweight publish/subscribe event bus.
# Any application module registers a callback here.
# The main loop calls bus.emit() — handlers do the rest.
# ─────────────────────────────────────────────────────────────

from typing import Callable, Dict, List, Optional
from interaction.gestures import GestureEvent, GestureType


# Type alias for handler functions
Handler = Callable[[int, GestureEvent], None]


class EventBus:
    """
    Decouples gesture output from any downstream application.

    Usage
    -----
    bus = EventBus()

    # Register a handler for all events
    bus.subscribe(my_handler)

    # Emit from the main loop
    bus.emit(locked_marker_id, gesture_event)

    Handler signature:
        def my_handler(marker_id: int, event: GestureEvent) -> None
    """

    def __init__(self):
        self._handlers: List[Handler] = []

    def subscribe(self, handler: Handler):
        """Register a callback to receive all gesture events."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: Handler):
        self._handlers = [h for h in self._handlers if h is not handler]

    def emit(self, marker_id: int, event: GestureEvent):
        """Dispatch event to all registered handlers."""
        for handler in self._handlers:
            try:
                handler(marker_id, event)
            except Exception as exc:
                print(f"[EventBus] Handler error: {exc}")


# ── Default console handler (always active for debugging) ─────

def console_handler(marker_id: int, event: GestureEvent):
    """Prints every event to stdout — replace or extend for real devices."""
    print(
        f"[EVENT] marker={marker_id:2d} | "
        f"{event.gesture_type.name:<14} | "
        f"delta={event.delta:+6.1f} | "
        f"value={event.value:5.1f}"
    )
