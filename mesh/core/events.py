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

    # Step execution (for multi-step agents)
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"

    # State updates
    STATE_UPDATE = "state_update"

    # Reasoning (o1/o3/Claude Extended Thinking)
    REASONING_START = "reasoning_start"
    REASONING_TOKEN = "reasoning_token"
    REASONING_END = "reasoning_end"

    # Metadata (usage, timing, model info)
    RESPONSE_METADATA = "response_metadata"

    # Sources and citations (Gemini grounding, RAG)
    SOURCE = "source"

    # File attachments (multi-modal)
    FILE = "file"

    # Custom data events (extensibility)
    CUSTOM_DATA = "custom_data"


@dataclass
class ExecutionEvent:
    """Provider-agnostic execution event.

    This event format is compatible with Vel's event system and can be
    translated to/from OpenAI, Anthropic, and other provider formats.

    The raw_event field preserves the original event (Vel event dict or
    native provider event) for debugging and accessing provider-specific fields.
    """

    type: EventType
    node_id: Optional[str] = None
    content: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_event: Optional[Any] = None  # Original vanilla event (Vel dict or native object)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "node_id": self.node_id,
            "content": self.content,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

        # Include raw_event if present (for dict/serializable types)
        if self.raw_event is not None:
            if isinstance(self.raw_event, dict):
                result["raw_event"] = self.raw_event
            else:
                # For non-dict objects, include type info
                result["raw_event_type"] = type(self.raw_event).__name__

        return result


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
