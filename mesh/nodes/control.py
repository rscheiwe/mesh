"""Control flow nodes for graph execution."""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from mesh.core.node import NodeConfig
from mesh.nodes.base import BaseNode
from mesh.state.state import GraphState


class ConditionalNode(BaseNode):
    """Node that makes decisions based on conditions."""

    def __init__(
        self,
        condition: Callable[[Dict[str, Any], Optional[GraphState]], bool],
        true_output: Any = True,
        false_output: Any = False,
        config: Optional[NodeConfig] = None,
    ):
        super().__init__(config)
        self.condition = condition
        self.true_output = true_output
        self.false_output = false_output

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute conditional logic.

        Args:
            input_data: Input data for evaluation
            state: Optional shared state

        Returns:
            Dict with 'result' and 'branch' keys
        """
        try:
            # Evaluate condition
            if asyncio.iscoroutinefunction(self.condition):
                result = await self.condition(input_data, state)
            else:
                result = self.condition(input_data, state)

            return {
                "result": self.true_output if result else self.false_output,
                "branch": "true" if result else "false",
                "condition_met": result,
                "input": input_data,
            }

        except Exception as e:
            return {
                "result": self.false_output,
                "branch": "error",
                "condition_met": False,
                "error": str(e),
                "input": input_data,
            }


class LoopNode(BaseNode):
    """Node that implements loop functionality."""

    def __init__(
        self,
        loop_condition: Callable[[Dict[str, Any], Optional[GraphState]], bool],
        max_iterations: int = 100,
        config: Optional[NodeConfig] = None,
    ):
        super().__init__(config)
        self.loop_condition = loop_condition
        self.max_iterations = max_iterations
        self.current_iteration = 0

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: Optional[GraphState] = None
    ) -> Dict[str, Any]:
        """Execute loop logic.

        Args:
            input_data: Input data for the loop
            state: Optional shared state

        Returns:
            Dict with loop status and data
        """
        # Check if we should continue looping
        should_continue = False

        try:
            if asyncio.iscoroutinefunction(self.loop_condition):
                should_continue = await self.loop_condition(input_data, state)
            else:
                should_continue = self.loop_condition(input_data, state)
        except Exception as e:
            should_continue = False
            error = str(e)
        else:
            error = None

        # Increment iteration counter
        self.current_iteration += 1

        # Check max iterations
        if self.current_iteration >= self.max_iterations:
            should_continue = False

        # Store iteration data in state if available
        if state:
            await state.set(f"{self.id}_iteration", self.current_iteration)

        return {
            "continue": should_continue,
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "max_reached": self.current_iteration >= self.max_iterations,
            "data": input_data,
            "error": error,
        }

    def reset(self) -> None:
        """Reset the node status and iteration counter."""
        super().reset()
        self.current_iteration = 0
