"""Subgraph node for embedding graphs within graphs.

This module provides SubgraphNode which wraps a Subgraph as a standard Node
that can be added to a StateGraph.

Example:
    >>> from mesh import StateGraph, Subgraph
    >>> from mesh.nodes.subgraph import SubgraphNode
    >>>
    >>> # Create subgraph
    >>> sub = StateGraph()
    >>> sub.add_node("tool", my_tool, node_type="tool")
    >>> sub.set_entry_point("tool")
    >>> compiled = sub.compile()
    >>>
    >>> # Use in parent graph
    >>> parent = StateGraph()
    >>> parent.add_node("research", SubgraphNode("research", Subgraph(compiled)))
"""

from typing import Any, TYPE_CHECKING
from mesh.nodes.base import BaseNode, NodeResult
from mesh.subgraph import Subgraph

if TYPE_CHECKING:
    from mesh.core.state import ExecutionContext


class SubgraphNode(BaseNode):
    """Node type for executing subgraphs.

    A SubgraphNode wraps a Subgraph instance, allowing it to be added to
    a parent StateGraph like any other node. When executed, it runs the
    entire embedded graph and returns the aggregated results.

    Attributes:
        subgraph: The Subgraph instance to execute
        node_type: Always "subgraph"

    Example:
        >>> subgraph = Subgraph(compiled_graph, config)
        >>> node = SubgraphNode("research_phase", subgraph)
        >>> graph.add_node("research", node)
    """

    node_type = "subgraph"

    def __init__(
        self,
        id: str,
        subgraph: Subgraph,
        config: dict = None,
    ):
        """Initialize subgraph node.

        Args:
            id: Unique node identifier
            subgraph: Subgraph instance to execute
            config: Optional additional configuration
        """
        super().__init__(id=id, config=config or {})
        self.subgraph = subgraph

    async def _execute_impl(
        self,
        input: Any,
        context: "ExecutionContext",
    ) -> NodeResult:
        """Execute the subgraph.

        Runs the embedded graph with proper context isolation and
        collects all events and final output.

        Args:
            input: Input data from parent graph
            context: Parent execution context

        Returns:
            NodeResult with subgraph output and state updates
        """
        from mesh.core.events import ExecutionEvent, EventType
        from datetime import datetime

        # Track events and final output
        events_collected = []
        final_output = None
        final_state = {}

        # Execute subgraph and collect events
        async for event in self.subgraph.execute(input, context):
            events_collected.append(event)

            # Emit event through parent context
            if context._event_emitter:
                await context.emit_event(event)

            # Capture final output from subgraph complete event
            if event.type == EventType.SUBGRAPH_COMPLETE:
                final_output = event.output
                if event.metadata and "final_state" in event.metadata:
                    final_state = event.metadata["final_state"]

        # Build result
        return NodeResult(
            output=final_output or {},
            state=final_state,
            metadata={
                "subgraph_name": self.subgraph.name,
                "events_count": len(events_collected),
                "isolated": self.subgraph.config.isolated,
            },
        )

    def __repr__(self) -> str:
        return f"SubgraphNode(id='{self.id}', subgraph={self.subgraph})"
