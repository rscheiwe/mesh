"""Event system for graph execution streaming."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel


class EventType(str, Enum):
    """Types of events that can occur during graph execution."""

    # Graph-level events
    GRAPH_START = "graph_start"
    GRAPH_END = "graph_end"
    GRAPH_ERROR = "graph_error"

    # Node-level events
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    NODE_SKIPPED = "node_skipped"

    # Streaming events
    STREAM_CHUNK = "stream_chunk"

    # Tool events
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"

    # State events
    STATE_UPDATE = "state_update"

    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Base event class for graph execution."""

    type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "node_name": self.node_name,
            "data": self.data,
            "metadata": self.metadata,
        }


@dataclass
class GraphStartEvent(Event):
    """Event emitted when graph execution starts."""

    def __init__(self, graph_id: str, graph_name: str, **kwargs):
        super().__init__(
            type=EventType.GRAPH_START,
            data={
                "graph_id": graph_id,
                "graph_name": graph_name,
            },
            **kwargs,
        )


@dataclass
class GraphEndEvent(Event):
    """Event emitted when graph execution ends."""

    def __init__(self, graph_id: str, success: bool, execution_time: float, **kwargs):
        super().__init__(
            type=EventType.GRAPH_END,
            data={
                "graph_id": graph_id,
                "success": success,
                "execution_time": execution_time,
            },
            **kwargs,
        )


@dataclass
class NodeStartEvent(Event):
    """Event emitted when a node starts executing."""

    def __init__(self, node_id: str, node_name: str, node_type: str, **kwargs):
        super().__init__(
            type=EventType.NODE_START,
            node_id=node_id,
            node_name=node_name,
            data={
                "node_type": node_type,
            },
            **kwargs,
        )


@dataclass
class NodeEndEvent(Event):
    """Event emitted when a node finishes executing."""

    def __init__(
        self,
        node_id: str,
        node_name: str,
        success: bool,
        execution_time: float,
        **kwargs,
    ):
        super().__init__(
            type=EventType.NODE_END,
            node_id=node_id,
            node_name=node_name,
            data={
                "success": success,
                "execution_time": execution_time,
            },
            **kwargs,
        )


@dataclass
class StreamChunkEvent(Event):
    """Event emitted for streaming content chunks."""

    def __init__(self, node_id: str, node_name: str, content: str, **kwargs):
        super().__init__(
            type=EventType.STREAM_CHUNK,
            node_id=node_id,
            node_name=node_name,
            data={
                "content": content,
            },
            **kwargs,
        )


@dataclass
class ToolStartEvent(Event):
    """Event emitted when a tool starts executing."""

    def __init__(
        self,
        node_id: str,
        node_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            type=EventType.TOOL_START,
            node_id=node_id,
            node_name=node_name,
            data={
                "tool_name": tool_name,
                "tool_args": tool_args,
            },
            **kwargs,
        )


@dataclass
class ToolEndEvent(Event):
    """Event emitted when a tool finishes executing."""

    def __init__(
        self,
        node_id: str,
        node_name: str,
        tool_name: str,
        success: bool,
        result: Any,
        **kwargs,
    ):
        super().__init__(
            type=EventType.TOOL_END,
            node_id=node_id,
            node_name=node_name,
            data={
                "tool_name": tool_name,
                "success": success,
                "result": result,
            },
            **kwargs,
        )


@dataclass
class ErrorEvent(Event):
    """Event emitted when an error occurs."""

    def __init__(
        self,
        error: str,
        error_type: str,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        **kwargs,
    ):
        event_type = EventType.NODE_ERROR if node_id else EventType.GRAPH_ERROR
        super().__init__(
            type=event_type,
            node_id=node_id,
            node_name=node_name,
            data={
                "error": error,
                "error_type": error_type,
            },
            **kwargs,
        )


@dataclass
class StateUpdateEvent(Event):
    """Event emitted when state is updated."""

    def __init__(self, key: str, value: Any, node_id: Optional[str] = None, **kwargs):
        super().__init__(
            type=EventType.STATE_UPDATE,
            node_id=node_id,
            data={
                "key": key,
                "value": value,
            },
            **kwargs,
        )
