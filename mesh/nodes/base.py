"""Base node class with common functionality."""

from typing import Any, Dict, Optional

from mesh.core.node import Node, NodeConfig
from mesh.state.state import GraphState


class BaseNode(Node[Dict[str, Any], Dict[str, Any], GraphState]):
    """Base node with common functionality for all specialized nodes."""

    def __init__(self, config: Optional[NodeConfig] = None):
        super().__init__(config)

    async def pre_execute(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Hook called before execute.

        Args:
            input_data: Input data for the node
            state: Optional shared state

        Returns:
            Processed input data
        """
        return input_data

    async def post_execute(
        self, result: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Hook called after execute.

        Args:
            result: Result from execute
            state: Optional shared state

        Returns:
            Processed result
        """
        return result

    async def execute(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute the node logic with pre/post hooks.

        Args:
            input_data: Input data for the node
            state: Optional shared state

        Returns:
            Output from the node execution
        """
        # Pre-processing
        processed_input = await self.pre_execute(input_data, state)

        # Main execution
        result = await self._execute_impl(processed_input, state)

        # Post-processing
        final_result = await self.post_execute(result, state)

        # Store result in state if available
        if state:
            await state.add_node_output(self.id, final_result)

        return final_result

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Implementation of the node's main logic.

        Subclasses should override this method.

        Args:
            input_data: Input data for the node
            state: Optional shared state

        Returns:
            Output from the node execution
        """
        return input_data
