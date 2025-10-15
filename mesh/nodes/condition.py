"""Condition node for conditional branching.

This node evaluates conditions and determines which branch(es) to execute.
It's used for routing workflow based on data or state.
"""

from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext


@dataclass
class Condition:
    """Single branch condition.

    Attributes:
        name: Condition name/identifier
        predicate: Function that evaluates to True/False
        target_node: Node ID to route to if condition is True
    """

    name: str
    predicate: Callable[[Any], bool]
    target_node: str


class ConditionNode(BaseNode):
    """Conditional branching with multiple output paths.

    This node evaluates a list of conditions and marks which branches
    are fulfilled. The executor uses this information to skip unfulfilled
    branches when queuing downstream nodes.

    Similar to Flowise's conditionAgentflow node.

    Example:
        >>> def check_success(input):
        ...     return "success" in str(input).lower()
        >>>
        >>> condition = ConditionNode(
        ...     id="check_result",
        ...     conditions=[
        ...         Condition("success", check_success, "success_handler"),
        ...         Condition("failure", lambda x: not check_success(x), "error_handler"),
        ...     ],
        ... )
    """

    def __init__(
        self,
        id: str,
        conditions: List[Condition],
        default_target: Optional[str] = None,
        config: Dict[str, Any] = None,
    ):
        """Initialize condition node.

        Args:
            id: Node identifier
            conditions: List of Condition objects to evaluate
            default_target: Default node to route to if no conditions match
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.conditions = conditions
        self.default_target = default_target

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Evaluate conditions and return routing metadata.

        Args:
            input: Input data to evaluate
            context: Execution context

        Returns:
            NodeResult with routing information in metadata
        """
        fulfilled_conditions = []
        unfulfilled_conditions = []

        # Evaluate each condition
        for condition in self.conditions:
            try:
                # Call predicate with input
                is_fulfilled = condition.predicate(input)

                if is_fulfilled:
                    fulfilled_conditions.append(condition)
                else:
                    unfulfilled_conditions.append(condition)

            except Exception as e:
                # Treat evaluation errors as unfulfilled
                unfulfilled_conditions.append(condition)
                # Log the error
                print(f"Condition '{condition.name}' evaluation failed: {e}")

        # If no conditions fulfilled and default exists, use it
        if not fulfilled_conditions and self.default_target:
            fulfilled_conditions.append(
                Condition(
                    name="default",
                    predicate=lambda x: True,
                    target_node=self.default_target,
                )
            )

        # Build metadata for executor
        condition_metadata = []
        for condition in self.conditions:
            condition_metadata.append(
                {
                    "name": condition.name,
                    "target": condition.target_node,
                    "fulfilled": condition in fulfilled_conditions,
                }
            )

        # Add default if used
        if not any(c.name == "default" for c in self.conditions) and self.default_target:
            condition_metadata.append(
                {
                    "name": "default",
                    "target": self.default_target,
                    "fulfilled": not any(c in fulfilled_conditions for c in self.conditions),
                }
            )

        return NodeResult(
            output={
                "input": input,  # Pass through input
                "fulfilled": [c.name for c in fulfilled_conditions],
                "unfulfilled": [c.name for c in unfulfilled_conditions],
            },
            metadata={
                "conditions": condition_metadata,
                "node_type": "condition",
            },
        )

    def __repr__(self) -> str:
        return f"ConditionNode(id='{self.id}', conditions={len(self.conditions)})"


class SimpleCondition:
    """Helper class for creating conditions with simple comparison logic."""

    @staticmethod
    def equals(field: str, value: Any, target_node: str) -> Condition:
        """Create condition that checks if field equals value."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                return input.get(field) == value
            return False

        return Condition(
            name=f"{field}_equals_{value}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def contains(field: str, substring: str, target_node: str) -> Condition:
        """Create condition that checks if field contains substring."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                value = str(input.get(field, ""))
                return substring in value
            return False

        return Condition(
            name=f"{field}_contains_{substring}",
            predicate=predicate,
            target_node=target_node,
        )

    @staticmethod
    def greater_than(field: str, threshold: float, target_node: str) -> Condition:
        """Create condition that checks if field is greater than threshold."""

        def predicate(input: Any) -> bool:
            if isinstance(input, dict):
                try:
                    value = float(input.get(field, 0))
                    return value > threshold
                except (ValueError, TypeError):
                    return False
            return False

        return Condition(
            name=f"{field}_gt_{threshold}",
            predicate=predicate,
            target_node=target_node,
        )
