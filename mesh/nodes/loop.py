"""Loop node for iteration over arrays.

This node enables iteration over arrays, executing downstream nodes
for each item. It sets iteration context variables that can be accessed
via {{$iteration}} in templates.
"""

from typing import Any, Dict, List, Optional

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from jsonpath_ng import parse as jsonpath_parse


class LoopNode(BaseNode):
    """Iterate over an array and execute downstream nodes for each item.

    This node extracts an array from the input and signals the executor
    to iterate over it. For each iteration, it sets the {{$iteration}}
    variable in the context.

    The actual iteration happens in the executor via queue manipulation.
    This node provides the array and metadata.

    Example:
        >>> loop = LoopNode(
        ...     id="process_items",
        ...     array_path="$.items",  # JSONPath to array
        ...     max_iterations=100,
        ... )
    """

    def __init__(
        self,
        id: str,
        array_path: str = "$.items",
        max_iterations: int = 100,
        config: Dict[str, Any] = None,
    ):
        """Initialize loop node.

        Args:
            id: Node identifier
            array_path: JSONPath expression to extract array from input
            max_iterations: Maximum iterations (safety limit)
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.array_path = array_path
        self.max_iterations = max_iterations

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Extract array and prepare for iteration.

        Args:
            input: Input data containing array
            context: Execution context

        Returns:
            NodeResult with array data and loop metadata
        """
        # Extract array from input
        array_data = self._extract_array(input)

        if not isinstance(array_data, list):
            raise ValueError(
                f"Loop node '{self.id}': Path '{self.array_path}' "
                f"does not point to an array, got {type(array_data)}"
            )

        # Limit iterations
        original_length = len(array_data)
        array_data = array_data[: self.max_iterations]

        if len(array_data) < original_length:
            print(
                f"Warning: Loop node '{self.id}' limited to "
                f"{self.max_iterations} iterations (original: {original_length})"
            )

        return NodeResult(
            output={
                "array": array_data,
                "length": len(array_data),
                "original_length": original_length,
            },
            metadata={
                "loop": True,
                "array_path": self.array_path,
                "iterations": len(array_data),
                "node_type": "loop",
            },
        )

    def _extract_array(self, input: Any) -> List[Any]:
        """Extract array from input using JSONPath or direct access.

        Args:
            input: Input data

        Returns:
            Extracted array

        Raises:
            ValueError: If array cannot be extracted
        """
        # If input is already a list, return it
        if isinstance(input, list):
            return input

        # If input is dict with common array keys
        if isinstance(input, dict):
            # Try common keys first
            for key in ["array", "items", "data", "results"]:
                if key in input and isinstance(input[key], list):
                    return input[key]

            # Try JSONPath if specified
            if self.array_path.startswith("$"):
                try:
                    jsonpath_expr = jsonpath_parse(self.array_path)
                    matches = jsonpath_expr.find(input)
                    if matches:
                        value = matches[0].value
                        if isinstance(value, list):
                            return value
                except Exception as e:
                    raise ValueError(
                        f"Failed to extract array using JSONPath '{self.array_path}': {e}"
                    )

        raise ValueError(
            f"Could not extract array from input. "
            f"Input type: {type(input)}, Array path: {self.array_path}"
        )

    def __repr__(self) -> str:
        return f"LoopNode(id='{self.id}', path='{self.array_path}')"


class IterationHelper:
    """Helper utilities for working with iteration context."""

    @staticmethod
    def execute_for_each(
        context: ExecutionContext,
        array: List[Any],
        node_id: str,
    ) -> List[Dict[str, Any]]:
        """Execute iterations and return results.

        This is a helper for manually managing iterations outside
        the executor's automatic loop handling.

        Args:
            context: Execution context
            array: Array to iterate over
            node_id: Node being iterated

        Returns:
            List of iteration results
        """
        results = []
        total = len(array)

        for index, item in enumerate(array):
            # Set iteration context
            context.set_iteration_context(index, item, total)

            # Collect result (actual execution happens in executor)
            results.append(
                {
                    "index": index,
                    "value": item,
                    "is_first": index == 0,
                    "is_last": index == total - 1,
                }
            )

        # Clear iteration context
        context.clear_iteration_context()

        return results
