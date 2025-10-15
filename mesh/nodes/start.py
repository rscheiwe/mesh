"""Start node implementation.

The start node is the entry point to a graph. It passes the input through
to downstream nodes without modification.
"""

from typing import Any, Dict

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext


class StartNode(BaseNode):
    """Start node for graph execution.

    This node marks the entry point(s) of a graph. It simply passes
    its input through to the next nodes in the workflow.

    The start node can accept any input type and forwards it downstream.
    """

    def __init__(self, id: str = "START", config: Dict[str, Any] = None):
        """Initialize start node.

        Args:
            id: Node identifier (defaults to "START")
            config: Configuration dictionary
        """
        super().__init__(id, config)

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
        return NodeResult(
            output=input,
            metadata={"node_type": "start"},
        )
