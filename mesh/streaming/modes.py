"""Streaming modes for Mesh execution.

This module provides multiple streaming modes for different use cases:
- VALUES: Emit full state after each node
- UPDATES: Emit only state changes (deltas)
- MESSAGES: Emit chat messages only
- EVENTS: Emit all execution events (default behavior)
- DEBUG: Emit everything including internal state

Example:
    >>> from mesh import Executor, StreamMode
    >>> executor = Executor(graph, backend)
    >>>
    >>> # Different streaming modes
    >>> async for state in executor.stream(input, context, mode=StreamMode.VALUES):
    ...     print(f"Full state: {state}")
    >>>
    >>> async for delta in executor.stream(input, context, mode=StreamMode.UPDATES):
    ...     print(f"State changes: {delta}")
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from mesh.core.events import ExecutionEvent
    from mesh.core.state import ExecutionContext


class StreamMode(Enum):
    """Available streaming modes for graph execution."""

    VALUES = "values"      # Full state after each node
    UPDATES = "updates"    # State deltas only
    MESSAGES = "messages"  # Chat messages only
    EVENTS = "events"      # All execution events (default)
    DEBUG = "debug"        # Everything including internals


@dataclass
class StateValue:
    """Full state snapshot emitted in VALUES mode.

    Attributes:
        node_id: ID of the node that just completed
        state: Complete state dictionary
        metadata: Additional metadata about the state
    """
    node_id: str
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StateUpdate:
    """State delta emitted in UPDATES mode.

    Attributes:
        node_id: ID of the node that caused the update
        added: New keys added to state
        modified: Existing keys that were modified
        removed: Keys that were removed from state
    """
    node_id: str
    added: Dict[str, Any] = field(default_factory=dict)
    modified: Dict[str, Any] = field(default_factory=dict)
    removed: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def has_changes(self) -> bool:
        """Check if this update contains any changes."""
        return bool(self.added or self.modified or self.removed)


@dataclass
class StreamMessage:
    """Chat message emitted in MESSAGES mode.

    Attributes:
        role: Message role (user, assistant, tool, system)
        content: Message content
        node_id: ID of the node that generated this message
        metadata: Additional message metadata
    """
    role: str  # user, assistant, tool, system
    content: str
    node_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DebugInfo:
    """Debug information emitted in DEBUG mode.

    Attributes:
        event: The underlying execution event
        internal_state: Internal execution state
        queue: Current execution queue
        timing: Timing information
    """
    event: "ExecutionEvent"
    internal_state: Dict[str, Any] = field(default_factory=dict)
    queue: List[str] = field(default_factory=list)
    timing: Dict[str, Any] = field(default_factory=dict)
    visited_nodes: List[str] = field(default_factory=list)


class StreamModeAdapter:
    """Adapts execution events to different streaming modes.

    This adapter transforms the raw ExecutionEvent stream into
    different output formats based on the selected streaming mode.

    Example:
        >>> adapter = StreamModeAdapter(StreamMode.VALUES)
        >>> async for item in adapter.adapt(events, context):
        ...     print(item)
    """

    def __init__(self, mode: StreamMode):
        """Initialize adapter with specified mode.

        Args:
            mode: The streaming mode to use
        """
        self.mode = mode
        self._previous_state: Dict[str, Any] = {}
        self._message_accumulator: Dict[str, str] = {}  # node_id -> accumulated content
        self._current_message_node: Optional[str] = None

    async def adapt(
        self,
        events: AsyncIterator["ExecutionEvent"],
        context: "ExecutionContext",
    ) -> AsyncIterator[Any]:
        """Convert events to specified streaming mode.

        Args:
            events: Async iterator of ExecutionEvents
            context: Execution context for state access

        Yields:
            Items appropriate to the streaming mode
        """
        from mesh.core.events import EventType

        if self.mode == StreamMode.EVENTS:
            # Pass through all events unchanged
            async for event in events:
                yield event

        elif self.mode == StreamMode.VALUES:
            # Track accumulated state from event outputs
            accumulated_state: Dict[str, Any] = {}
            async for event in events:
                # Update accumulated state from node outputs
                if event.type == EventType.NODE_COMPLETE and event.output:
                    if isinstance(event.output, dict):
                        accumulated_state.update(event.output)
                        accumulated_state[f"{event.node_id}_output"] = event.output

                # Emit full state on node/subgraph completion
                if event.type in (EventType.NODE_COMPLETE, EventType.SUBGRAPH_COMPLETE):
                    # Merge with context state for complete picture
                    full_state = {**context.state, **accumulated_state}
                    yield StateValue(
                        node_id=event.node_id or "",
                        state=full_state,
                        metadata=event.metadata or {},
                    )

        elif self.mode == StreamMode.UPDATES:
            async for event in events:
                # Emit state delta on node/subgraph completion
                if event.type in (EventType.NODE_COMPLETE, EventType.SUBGRAPH_COMPLETE):
                    # Use event.output for the delta since it contains the node's changes
                    if event.output and isinstance(event.output, dict):
                        # Compute delta from this node's output
                        delta = self._compute_delta_from_output(
                            event.output, event.node_id or ""
                        )
                        if delta.has_changes:
                            yield delta

        elif self.mode == StreamMode.MESSAGES:
            async for event in events:
                # Handle token events - accumulate into messages
                if event.type == EventType.TOKEN:
                    node_id = event.node_id or "unknown"
                    if node_id not in self._message_accumulator:
                        self._message_accumulator[node_id] = ""
                    self._message_accumulator[node_id] += event.content or ""

                # Emit accumulated message on node completion
                elif event.type == EventType.NODE_COMPLETE:
                    node_id = event.node_id or "unknown"
                    if node_id in self._message_accumulator:
                        content = self._message_accumulator.pop(node_id)
                        if content:
                            yield StreamMessage(
                                role=event.metadata.get("role", "assistant") if event.metadata else "assistant",
                                content=content,
                                node_id=node_id,
                                metadata=event.metadata or {},
                            )

                # Handle explicit message events
                elif event.type == EventType.CUSTOM_DATA:
                    if event.metadata and event.metadata.get("data_type") == "message":
                        yield StreamMessage(
                            role=event.metadata.get("role", "assistant"),
                            content=event.content or "",
                            node_id=event.node_id or "",
                            metadata=event.metadata,
                        )

        elif self.mode == StreamMode.DEBUG:
            async for event in events:
                yield DebugInfo(
                    event=event,
                    internal_state={
                        "_current_node": context.state.get("_current_node"),
                        "_visited": context.state.get("_visited_nodes", []),
                        "_pending_queue": context.state.get("_pending_queue", []),
                    },
                    queue=context.state.get("_pending_queue", []),
                    visited_nodes=context.state.get("_visited_nodes", []),
                    timing={
                        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    },
                )

    def _compute_delta(self, current_state: Dict[str, Any], node_id: str) -> StateUpdate:
        """Compute state delta from previous state.

        Args:
            current_state: Current state dictionary
            node_id: ID of the node that caused the change

        Returns:
            StateUpdate with added, modified, and removed keys
        """
        added: Dict[str, Any] = {}
        modified: Dict[str, Any] = {}
        removed: List[str] = []

        # Find added and modified keys
        for key, value in current_state.items():
            if key.startswith("_"):
                continue  # Skip internal keys
            if key not in self._previous_state:
                added[key] = value
            elif self._previous_state[key] != value:
                modified[key] = value

        # Find removed keys
        for key in self._previous_state:
            if key not in current_state and not key.startswith("_"):
                removed.append(key)

        # Update previous state for next comparison
        self._previous_state = {
            k: v for k, v in current_state.items()
            if not k.startswith("_")
        }

        return StateUpdate(
            node_id=node_id,
            added=added,
            modified=modified,
            removed=removed,
        )

    def _compute_delta_from_output(
        self, output: Dict[str, Any], node_id: str
    ) -> StateUpdate:
        """Compute state delta from a node's output.

        This is used for UPDATES mode to compute changes from node outputs
        rather than full context state.

        Args:
            output: Node output dictionary
            node_id: ID of the node that produced the output

        Returns:
            StateUpdate representing changes from this node
        """
        added: Dict[str, Any] = {}
        modified: Dict[str, Any] = {}

        # All keys in output are either added or modified
        for key, value in output.items():
            if key.startswith("_"):
                continue  # Skip internal keys
            if key not in self._previous_state:
                added[key] = value
            elif self._previous_state[key] != value:
                modified[key] = value

        # Update previous state
        for key, value in output.items():
            if not key.startswith("_"):
                self._previous_state[key] = value

        return StateUpdate(
            node_id=node_id,
            added=added,
            modified=modified,
            removed=[],  # Outputs don't typically remove keys
        )

    def reset(self) -> None:
        """Reset adapter state for new execution."""
        self._previous_state = {}
        self._message_accumulator = {}
        self._current_message_node = None
