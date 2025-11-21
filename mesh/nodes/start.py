"""Start node implementation.

The start node is the entry point to a graph. It passes the input through
to downstream nodes without modification.
"""

from typing import Any, Dict

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, transform_event_for_transient_mode


class StartNode(BaseNode):
    """Start node for graph execution.

    This node marks the entry point(s) of a graph. It simply passes
    its input through to the next nodes in the workflow.

    The start node can accept any input type and forwards it downstream.
    """

    def __init__(self, id: str = "START", event_mode: str = "full", config: Dict[str, Any] = None):
        """Initialize start node.

        Args:
            id: Node identifier (defaults to "START")
            event_mode: Event emission mode (default: "full")
                - "full": All events - streams to chat
                - "status_only": Only progress indicators
                - "transient_events": All events prefixed with data-start-node-*
                - "silent": No events
            config: Configuration dictionary
        """
        super().__init__(id, config)
        self.event_mode = event_mode

    async def _emit_event_if_enabled(self, context: ExecutionContext, event: "ExecutionEvent") -> None:
        """Emit event based on event_mode.

        Args:
            context: Execution context
            event: Event to emit
        """
        # Silent mode - no events
        if self.event_mode == "silent":
            return

        # Status only - skip regular events, only custom events emitted separately
        if self.event_mode == "status_only":
            # Only emit if it's a custom data event
            if event.type == EventType.CUSTOM_DATA:
                await context.emit_event(event)
            return

        # Transient events - transform all events with data-start-node-* prefix
        if self.event_mode == "transient_events":
            transformed_event = transform_event_for_transient_mode(event, "start")
            await context.emit_event(transformed_event)
        else:
            # Full mode - emit events normally
            await context.emit_event(event)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Pass input through to downstream nodes.

        Args:
            input: Input data to the graph
            context: Execution context

        Returns:
            NodeResult with input as output
        """
        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.EXECUTION_START,
                node_id=self.id,
                metadata={"node_type": "start"},
            )
        )

        return NodeResult(
            output=input,
            metadata={"node_type": "start"},
        )
