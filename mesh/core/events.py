"""Event system for streaming execution updates.

Provider-agnostic event format compatible with Vel and Vercel AI SDK.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List, Callable, Awaitable
from enum import Enum


class EventType(str, Enum):
    """Standard event types for execution streaming."""

    # Execution lifecycle
    EXECUTION_START = "execution_start"
    EXECUTION_COMPLETE = "execution_complete"
    EXECUTION_ERROR = "execution_error"

    # Node lifecycle
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_ERROR = "node_error"

    # Streaming content
    TOKEN = "token"
    MESSAGE_START = "message_start"
    MESSAGE_COMPLETE = "message_complete"

    # Tool execution
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"

    # State updates
    STATE_UPDATE = "state_update"


@dataclass
class ExecutionEvent:
    """Provider-agnostic execution event.

    This event format is compatible with Vel's event system and can be
    translated to/from OpenAI, Anthropic, and other provider formats.
    """

    type: EventType
    node_id: Optional[str] = None
    content: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "node_id": self.node_id,
            "content": self.content,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class EventEmitter:
    """Event emitter for publishing execution events.

    This class manages event listeners and provides methods for emitting
    events during graph execution. It can integrate with Vel's event system
    for provider-agnostic event translation.
    """

    def __init__(self):
        self._listeners: List[Callable[[ExecutionEvent], Awaitable[None]]] = []

    def on(self, listener: Callable[[ExecutionEvent], Awaitable[None]]) -> None:
        """Register an event listener.

        Args:
            listener: Async function that receives ExecutionEvent objects
        """
        self._listeners.append(listener)

    def off(self, listener: Callable[[ExecutionEvent], Awaitable[None]]) -> None:
        """Remove an event listener.

        Args:
            listener: The listener function to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def emit(self, event: ExecutionEvent) -> None:
        """Emit an event to all listeners.

        Args:
            event: The event to emit
        """
        for listener in self._listeners:
            try:
                await listener(event)
            except Exception as e:
                # Log error but don't fail execution
                print(f"Error in event listener: {e}")

    def clear(self) -> None:
        """Remove all event listeners."""
        self._listeners.clear()


class VelEventTranslator:
    """Translator for Vel SDK events.

    This class provides methods to translate between Vel events and
    Mesh ExecutionEvent format. It will use Vel's event translation
    when the Vel SDK is available.
    """

    def __init__(self):
        self._vel_available = False
        try:
            # Try to import Vel SDK
            import vel  # noqa: F401

            self._vel_available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        """Check if Vel SDK is available."""
        return self._vel_available

    def from_vel_event(self, vel_event: Any) -> ExecutionEvent:
        """Convert Vel event to Mesh ExecutionEvent.

        Args:
            vel_event: Event from Vel SDK

        Returns:
            ExecutionEvent in Mesh format
        """
        if not self._vel_available:
            raise RuntimeError("Vel SDK is not installed")

        # Vel event translation logic
        # This will be implemented based on actual Vel SDK event format
        event_type = self._map_vel_event_type(vel_event)
        return ExecutionEvent(
            type=event_type,
            content=getattr(vel_event, "content", None),
            metadata=getattr(vel_event, "metadata", {}),
        )

    def _map_vel_event_type(self, vel_event: Any) -> EventType:
        """Map Vel event type to Mesh EventType."""
        # This mapping will be based on Vel SDK's event types
        vel_type = getattr(vel_event, "type", "unknown")

        mapping = {
            "token": EventType.TOKEN,
            "message_start": EventType.MESSAGE_START,
            "message_complete": EventType.MESSAGE_COMPLETE,
            "tool_call_start": EventType.TOOL_CALL_START,
            "tool_call_complete": EventType.TOOL_CALL_COMPLETE,
            "error": EventType.EXECUTION_ERROR,
        }

        return mapping.get(vel_type, EventType.NODE_COMPLETE)


class OpenAIEventTranslator:
    """Translator for OpenAI SDK events."""

    @staticmethod
    def from_openai_chunk(chunk: Any) -> ExecutionEvent:
        """Convert OpenAI streaming chunk to Mesh ExecutionEvent.

        Args:
            chunk: Chunk from OpenAI streaming response

        Returns:
            ExecutionEvent in Mesh format
        """
        content = ""
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                content = delta.content

        return ExecutionEvent(
            type=EventType.TOKEN,
            content=content,
            metadata={"provider": "openai"},
        )

    @staticmethod
    def from_openai_delta(delta: Any) -> ExecutionEvent:
        """Convert OpenAI Assistants SDK delta to Mesh ExecutionEvent.

        Args:
            delta: Delta from OpenAI Assistants streaming

        Returns:
            ExecutionEvent in Mesh format
        """
        content = getattr(delta, "value", "")

        return ExecutionEvent(
            type=EventType.TOKEN,
            content=content,
            metadata={"provider": "openai_assistants"},
        )
