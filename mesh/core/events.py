"""Event system for streaming execution updates.

Provider-agnostic event format compatible with Vel and Vercel AI SDK.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List, Callable, Awaitable
from enum import Enum


class EventType(str, Enum):
    """AI SDK compatible event types for execution streaming.

    Based on Vercel AI SDK V5 UI Stream Protocol.
    Matches Vel event format for seamless frontend integration.

    Mesh-specific events are prefixed with 'data-' for AI SDK compatibility.
    These are handled via the onData callback in useChat.
    """

    # Execution lifecycle (Mesh-specific, prefixed with data- for AI SDK)
    EXECUTION_START = "data-execution-start"
    EXECUTION_COMPLETE = "data-execution-complete"
    EXECUTION_ERROR = "data-execution-error"

    # Node lifecycle (Mesh-specific, prefixed with data- for AI SDK)
    NODE_START = "data-node-start"
    NODE_COMPLETE = "data-node-complete"
    NODE_ERROR = "data-node-error"

    # AI SDK V5 - Generation lifecycle
    START = "start"  # Message/generation start
    FINISH = "finish"  # Generation complete
    FINISH_MESSAGE = "finish-message"  # Message complete
    ERROR = "error"  # Error event

    # AI SDK V5 - Text streaming
    TEXT_START = "text-start"  # Text block start
    TEXT_DELTA = "text-delta"  # Token streaming (replaces TOKEN)
    TEXT_END = "text-end"  # Text block end

    # AI SDK V5 - Tool execution
    TOOL_INPUT_START = "tool-input-start"  # Tool call start
    TOOL_INPUT_DELTA = "tool-input-delta"  # Tool args streaming
    TOOL_INPUT_AVAILABLE = "tool-input-available"  # Tool ready to execute
    TOOL_OUTPUT_AVAILABLE = "tool-output-available"  # Tool result ready

    # AI SDK V5 - Multi-step agents
    START_STEP = "start-step"  # Step start
    FINISH_STEP = "finish-step"  # Step complete

    # AI SDK V5 - Reasoning (o1, o3, Claude Extended Thinking)
    REASONING_START = "reasoning-start"
    REASONING_DELTA = "reasoning-delta"  # Reasoning token (replaces REASONING_TOKEN)
    REASONING_END = "reasoning-end"

    # AI SDK V5 - Metadata
    RESPONSE_METADATA = "response-metadata"

    # AI SDK V5 - Multimodal
    SOURCE = "source"  # Citations, grounding
    FILE = "file"  # File attachments

    # State updates (Mesh-specific, prefixed with data- for AI SDK)
    STATE_UPDATE = "data-state-update"

    # Approval workflow events (Mesh-specific, prefixed with data- for AI SDK)
    # Used for human-in-the-loop workflows where execution pauses for approval
    APPROVAL_PENDING = "data-approval-pending"  # Execution paused, awaiting approval
    APPROVAL_RECEIVED = "data-approval-received"  # Approval granted, resuming
    APPROVAL_REJECTED = "data-approval-rejected"  # Approval denied
    APPROVAL_TIMEOUT = "data-approval-timeout"  # Approval timed out

    # Custom data events (AI SDK V5 data-* pattern)
    CUSTOM_DATA = "data-custom"

    # Legacy aliases for backwards compatibility (deprecated)
    TOKEN = "text-delta"  # Alias for TEXT_DELTA
    MESSAGE_START = "text-start"  # Alias for TEXT_START
    MESSAGE_COMPLETE = "text-end"  # Alias for TEXT_END
    TOOL_CALL_START = "tool-input-start"  # Alias for TOOL_INPUT_START
    TOOL_CALL_COMPLETE = "tool-output-available"  # Alias for TOOL_OUTPUT_AVAILABLE
    STEP_START = "start-step"  # Alias for START_STEP
    STEP_COMPLETE = "finish-step"  # Alias for FINISH_STEP
    REASONING_TOKEN = "reasoning-delta"  # Alias for REASONING_DELTA


# Custom data-* events for multi-agent orchestration
# Following Vercel AI SDK / Vel pattern for FE rendering hints

@dataclass
class MeshNodeStartData:
    """Data payload for mesh-node-start event.

    Emitted when an intermediate agent/LLM node begins execution.
    Provides context for UI to show progress indicators.
    """
    node_id: str
    node_type: str  # "agent" or "llm"
    is_final: bool
    is_intermediate: bool


@dataclass
class MeshNodeCompleteData:
    """Data payload for mesh-node-complete event.

    Emitted when an intermediate agent/LLM node finishes execution.
    """
    node_id: str
    node_type: str
    is_final: bool
    is_intermediate: bool
    output_preview: Optional[str] = None  # First 100 chars of output


@dataclass
class MeshGraphProgressData:
    """Data payload for mesh-graph-progress event.

    Emitted periodically to show overall graph execution progress.
    """
    nodes_completed: int
    nodes_total: int
    current_node: str
    progress_percent: float


@dataclass
class ExecutionEvent:
    """AI SDK compatible execution event.

    This event format matches Vercel AI SDK V5 UI Stream Protocol and Vel's event system.
    Can be translated to/from OpenAI, Anthropic, and other provider formats.

    The raw_event field preserves the original event (Vel event dict or
    native provider event) for debugging and accessing provider-specific fields.

    Node metadata (node_id, node_type, node_name) is included in ALL events
    for comprehensive execution tracking.
    """

    type: EventType
    node_id: Optional[str] = None
    delta: Optional[str] = None  # AI SDK field for streaming content (text-delta, reasoning-delta)
    content: Optional[str] = None  # Alternative to delta for backwards compatibility
    output: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_event: Optional[Any] = None  # Original vanilla event (Vel dict or native object)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for AI SDK serialization.

        For data-* events (custom Mesh events), wraps payload in 'data' field
        per AI SDK V5 convention (like data-artifact-* events).

        For standard AI SDK events, includes only AI SDK fields at top level,
        with Mesh-specific metadata in '_mesh' field (underscore prefix to avoid validation).
        """
        event_type = self.type.value if isinstance(self.type, EventType) else self.type

        # For data-* events, wrap everything in 'data' field
        if event_type.startswith("data-"):
            payload = {
                "timestamp": self.timestamp.isoformat(),
            }

            if self.node_id:
                payload["node_id"] = self.node_id
            if self.delta is not None:
                payload["delta"] = self.delta
            if self.content is not None:
                payload["content"] = self.content
            if self.output is not None:
                payload["output"] = self.output
            if self.error is not None:
                payload["errorText"] = self.error
            if self.metadata:
                payload["metadata"] = self.metadata
            if self.raw_event is not None:
                if isinstance(self.raw_event, dict):
                    payload["raw_event"] = self.raw_event
                else:
                    payload["raw_event_type"] = type(self.raw_event).__name__

            return {
                "type": event_type,
                "data": payload
            }

        # Standard AI SDK events - pass through ALL fields from raw_event if available
        # This preserves required AI SDK fields like 'id', 'toolCallId', etc.
        else:
            # If we have a raw_event dict, use it as the base (it has all AI SDK fields)
            if self.raw_event is not None and isinstance(self.raw_event, dict):
                result = dict(self.raw_event)  # Copy all fields from original event
                # Ensure type is set correctly (in case it was modified)
                result["type"] = event_type

                # Remove legacy 'error' field if present (Vel uses this, but AI SDK expects 'errorText')
                result.pop("error", None)

                # AI SDK finish-step and finish events should only have type field
                # Remove extra fields that Vel includes but AI SDK doesn't accept
                if event_type == "finish-step":
                    result = {"type": "finish-step"}
                elif event_type == "finish":
                    result = {"type": "finish"}

                # Override with any fields explicitly set on ExecutionEvent
                # This allows Mesh to update delta, content, etc. while preserving other fields
                if self.delta is not None:
                    result["delta"] = self.delta
                if self.content is not None:
                    result["content"] = self.content
                if self.output is not None:
                    result["output"] = self.output
                if self.error is not None:
                    result["errorText"] = self.error
            else:
                # No raw_event - build from scratch with available fields
                result = {"type": event_type}

                # Add AI SDK standard fields only
                if self.delta is not None:
                    result["delta"] = self.delta
                if self.content is not None:
                    result["content"] = self.content
                if self.output is not None:
                    result["output"] = self.output
                if self.error is not None:
                    result["errorText"] = self.error

            # Note: Mesh metadata (timestamp, node_id, metadata) are NOT included
            # in standard AI SDK events because they fail validation.
            # Use data-* events for Mesh orchestration metadata instead.

            return result


# Helper functions for creating custom data-* events

def create_mesh_node_start_event(
    node_id: str,
    node_type: str,
    is_final: bool,
    is_intermediate: bool
) -> ExecutionEvent:
    """Create a data-mesh-node-start custom event.

    Args:
        node_id: ID of the node starting
        node_type: "agent" or "llm"
        is_final: Whether this is a final node in the graph
        is_intermediate: Whether this is an intermediate node

    Returns:
        ExecutionEvent with custom data payload
    """
    data = MeshNodeStartData(
        node_id=node_id,
        node_type=node_type,
        is_final=is_final,
        is_intermediate=is_intermediate
    )

    return ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id=node_id,
        metadata={
            "data_event_type": "data-mesh-node-start",
            "data": {
                "node_id": data.node_id,
                "node_type": data.node_type,
                "is_final": data.is_final,
                "is_intermediate": data.is_intermediate
            },
            "transient": True  # UI indicator only, don't save to history
        }
    )


def create_mesh_node_complete_event(
    node_id: str,
    node_type: str,
    is_final: bool,
    is_intermediate: bool,
    output_preview: Optional[str] = None
) -> ExecutionEvent:
    """Create a data-mesh-node-complete custom event.

    Args:
        node_id: ID of the node completing
        node_type: "agent" or "llm"
        is_final: Whether this is a final node in the graph
        is_intermediate: Whether this is an intermediate node
        output_preview: First 100 chars of output (optional)

    Returns:
        ExecutionEvent with custom data payload
    """
    data = MeshNodeCompleteData(
        node_id=node_id,
        node_type=node_type,
        is_final=is_final,
        is_intermediate=is_intermediate,
        output_preview=output_preview
    )

    return ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id=node_id,
        metadata={
            "data_event_type": "data-mesh-node-complete",
            "data": {
                "node_id": data.node_id,
                "node_type": data.node_type,
                "is_final": data.is_final,
                "is_intermediate": data.is_intermediate,
                "output_preview": data.output_preview
            },
            "transient": True
        }
    )


def create_mesh_graph_progress_event(
    nodes_completed: int,
    nodes_total: int,
    current_node: str,
    progress_percent: float
) -> ExecutionEvent:
    """Create a data-mesh-graph-progress custom event.

    Args:
        nodes_completed: Number of nodes completed so far
        nodes_total: Total number of nodes in graph
        current_node: ID of currently executing node
        progress_percent: Progress percentage (0-100)

    Returns:
        ExecutionEvent with custom data payload
    """
    data = MeshGraphProgressData(
        nodes_completed=nodes_completed,
        nodes_total=nodes_total,
        current_node=current_node,
        progress_percent=progress_percent
    )

    return ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id=current_node,
        metadata={
            "data_event_type": "data-mesh-graph-progress",
            "data": {
                "nodes_completed": data.nodes_completed,
                "nodes_total": data.nodes_total,
                "current_node": data.current_node,
                "progress_percent": data.progress_percent
            },
            "transient": True
        }
    )


def transform_event_for_transient_mode(
    event: ExecutionEvent,
    node_type: str  # "agent", "llm", "tool", etc.
) -> ExecutionEvent:
    """Transform AI SDK event to data-{node_type}-node-* prefixed transient event.

    This is used for intermediate agents/LLMs in multi-agent workflows where
    events should be rendered differently (e.g., in a progress panel instead of chat).

    Args:
        event: Original execution event (already in AI SDK format)
        node_type: Type of node ("agent", "llm", "tool", "start", "foreach", "loop", "condition")

    Returns:
        Transformed event with data-* prefix and transient flag

    Example:
        >>> # Agent emits text-delta
        >>> original = ExecutionEvent(type=EventType.TEXT_DELTA, delta="Hello")
        >>> transformed = transform_event_for_transient_mode(original, "agent")
        >>> # FE receives: {type: "custom_data", metadata: {data_event_type: "data-agent-node-text-delta", ...}}
    """
    # Get event type as string
    if isinstance(event.type, EventType):
        event_type_str = event.type.value
    else:
        event_type_str = str(event.type)

    # Event types are already in AI SDK format (text-delta, text-start, etc.)
    # No mapping needed - just prefix with data-{node_type}-node-
    data_event_type = f"data-{node_type}-node-{event_type_str}"

    # Build data payload from original event
    data_payload = {}

    # AI SDK delta field (for text-delta, reasoning-delta, tool-input-delta)
    if event.delta is not None:
        data_payload["delta"] = event.delta

    # Backwards compatibility content field
    if event.content is not None:
        data_payload["content"] = event.content

    if event.output is not None:
        data_payload["output"] = event.output

    if event.error is not None:
        data_payload["errorText"] = event.error

    # Include original metadata (contains node_type, node_name, etc.)
    if event.metadata:
        data_payload["metadata"] = event.metadata

    # Always include node_id for tracking
    if event.node_id:
        data_payload["node_id"] = event.node_id

    return ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        node_id=event.node_id,
        metadata={
            "data_event_type": data_event_type,
            "data": data_payload,
            "transient": True,  # Not saved to history
            "original_event_type": event_type_str,  # For debugging
        }
    )


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
