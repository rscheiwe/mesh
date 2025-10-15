"""End node implementation.

The end node marks the termination point(s) of a graph. It collects
the final output and can perform final transformations.
"""

from typing import Any, Dict

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext


class EndNode(BaseNode):
    """End node for graph execution.

    This node marks the exit point(s) of a graph. It can optionally
    transform the final output or just pass it through.

    The end node's output becomes the final result of the graph execution.
    """

    def __init__(self, id: str = "END", config: Dict[str, Any] = None):
        """Initialize end node.

        Args:
            id: Node identifier (defaults to "END")
            config: Configuration dictionary
        """
        super().__init__(id, config)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Process final input and return as output.

        Args:
            input: Input data from parent nodes
            context: Execution context

        Returns:
            NodeResult with processed output
        """
        # Can add final formatting/transformation here if needed
        output = input

        # If input is a dict with 'content', extract it
        if isinstance(input, dict) and "content" in input:
            output = input["content"]

        return NodeResult(
            output=output,
            metadata={"node_type": "end"},
        )
