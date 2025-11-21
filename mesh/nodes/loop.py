"""Loop nodes for iteration and backward jumps.

This module provides:
1. ForEachNode - Iteration over arrays (for-each loops)
2. LoopNode - Backward jumps to previously executed nodes (Flowise-compatible)
"""

from typing import Any, Dict, List, Optional

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType, transform_event_for_transient_mode
from jsonpath_ng import parse as jsonpath_parse


class ForEachNode(BaseNode):
    """Iterate over an array and execute downstream nodes for each item.

    This node extracts an array from the input and signals the executor
    to iterate over it. For each iteration, it sets the {{$iteration}}
    variable in the context.

    The actual iteration happens in the executor via queue manipulation.
    This node provides the array and metadata.

    Example:
        >>> loop = ForEachNode(
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
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize foreach node.

        Args:
            id: Node identifier
            array_path: JSONPath expression to extract array from input
            max_iterations: Maximum iterations (safety limit)
            event_mode: Event emission mode (default: "full")
                - "full": All events - streams to chat
                - "status_only": Only progress indicators
                - "transient_events": All events prefixed with data-foreach-node-*
                - "silent": No events
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.array_path = array_path
        self.max_iterations = max_iterations
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

        # Transient events - transform all events with data-foreach-node-* prefix
        if self.event_mode == "transient_events":
            transformed_event = transform_event_for_transient_mode(event, "foreach")
            await context.emit_event(transformed_event)
        else:
            # Full mode - emit events normally
            await context.emit_event(event)

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
        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "array_path": self.array_path,
                    "node_type": "foreach",
                },
            )
        )

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

        # Emit complete event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_COMPLETE,
                node_id=self.id,
                metadata={
                    "array_length": len(array_data),
                    "iterations": len(array_data),
                    "node_type": "foreach",
                },
            )
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
        return f"ForEachNode(id='{self.id}', path='{self.array_path}')"


class LoopNode(BaseNode):
    """Loop node that redirects execution back to a previously executed node.

    This matches Flowise's Loop node behavior - when execution reaches this node,
    it "jumps" back to a specified target node that has already been executed,
    causing re-execution of that target and subsequent nodes.

    Tracks loop count to prevent infinite loops.

    Configuration Parameters:
        loop_back_to: The unique ID of a previously executed node to return to
        max_loop_count: Maximum number of times the loop can be performed (default: 5)

    Example:
        >>> loop = LoopNode(
        ...     id="retry_loop",
        ...     loop_back_to="process_node",
        ...     max_loop_count=5,
        ... )
    """

    def __init__(
        self,
        id: str,
        loop_back_to: str,
        max_loop_count: int = 5,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize loop node.

        Args:
            id: Node identifier
            loop_back_to: ID of the node to loop back to
            max_loop_count: Maximum loop iterations (default: 5)
            event_mode: Event emission mode (default: "full")
                - "full": All events - streams to chat
                - "status_only": Only progress indicators
                - "transient_events": All events prefixed with data-loop-node-*
                - "silent": No events
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.loop_back_to = loop_back_to
        self.max_loop_count = max_loop_count
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

        # Transient events - transform all events with data-loop-node-* prefix
        if self.event_mode == "transient_events":
            transformed_event = transform_event_for_transient_mode(event, "loop")
            await context.emit_event(transformed_event)
        else:
            # Full mode - emit events normally
            await context.emit_event(event)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Signal executor to jump back to target node.

        The executor handles the actual loop logic via NodeResult.loop_to_node
        and tracks loop count per node to enforce max_loop_count.

        Args:
            input: Input data (passed through to target node)
            context: Execution context

        Returns:
            NodeResult with loop_to_node set to trigger backward jump
        """
        # Emit loop event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "loop_back_to": self.loop_back_to,
                    "max_loop_count": self.max_loop_count,
                    "node_type": "loop",
                },
            )
        )

        return NodeResult(
            output=input,  # Pass input through to target node
            loop_to_node=self.loop_back_to,
            max_loops=self.max_loop_count,
            metadata={
                "loop_back_to": self.loop_back_to,
                "max_loop_count": self.max_loop_count,
                "node_type": "loop",
            },
        )

    def __repr__(self) -> str:
        return f"LoopNode(id='{self.id}', loop_back_to='{self.loop_back_to}', max={self.max_loop_count})"


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
