"""Smart edges that can handle complex routing logic."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .edge import Edge, EdgeType


class SmartEdgeType(str, Enum):
    """Extended edge types with built-in logic."""

    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    FEEDBACK_LOOP = "feedback_loop"  # Special type for feedback patterns
    MULTI_CONDITIONAL = "multi_conditional"  # Multiple conditions -> multiple targets


@dataclass
class RouterFunction:
    """Encapsulates routing logic that can return target node IDs."""

    func: Callable[[Any, Optional[Any]], Optional[str]]
    description: str = ""

    def __call__(self, data: Any, state: Optional[Any] = None) -> Optional[str]:
        """Execute the routing function."""
        return self.func(data, state)


@dataclass
class SmartEdge(Edge):
    """Edge that can make intelligent routing decisions.

    This edge type can handle complex routing patterns without
    requiring intermediate decision nodes.
    """

    # For FEEDBACK_LOOP type
    approval_check: Optional[Callable[[Any], bool]] = None
    max_iterations: int = 10
    iteration_key: str = "iterations"

    # For MULTI_CONDITIONAL type
    router: Optional[RouterFunction] = None
    targets: Dict[str, str] = field(default_factory=dict)  # condition_result -> node_id

    def should_continue_loop(
        self, data: Any, state: Optional[Any] = None, current_iterations: int = 0
    ) -> bool:
        """Determine if a feedback loop should continue."""

        if (
            self.edge_type != EdgeType.LOOP
            and self.edge_type != SmartEdgeType.FEEDBACK_LOOP
        ):
            return False

        # Check iteration count
        if current_iterations >= self.max_iterations:
            return False

        # Use approval check if provided
        if self.approval_check:
            return not self.approval_check(data)

        # Fall back to standard condition
        if self.condition:
            return self.condition(data)

        return False

    def get_next_target(self, data: Any, state: Optional[Any] = None) -> Optional[str]:
        """Get the next target node based on routing logic."""

        if self.edge_type == SmartEdgeType.MULTI_CONDITIONAL and self.router:
            # Use router function to determine target
            result = self.router(data, state)
            return self.targets.get(result) if result else None

        elif self.edge_type == SmartEdgeType.FEEDBACK_LOOP:
            # For feedback loops, return target if should continue
            if self.should_continue_loop(data, state):
                # Increment iteration counter
                if state:
                    import asyncio

                    current = asyncio.run(state.get(self.iteration_key, 0))
                    asyncio.run(state.set(self.iteration_key, current + 1))
                return self.target_node_id
            return None

        else:
            # Standard edge behavior
            if self.evaluate_condition(data):
                return self.target_node_id
            return None


@dataclass
class FeedbackLoopEdge(SmartEdge):
    """Specialized edge for feedback loop patterns.

    This edge automatically handles the approval check and iteration
    counting that would normally require a separate decision node.
    """

    def __init__(
        self,
        source_node_id: str,
        target_node_id: str,
        approval_keywords: List[str] = None,
        approval_field: str = "response",
        max_iterations: int = 5,
        **kwargs,
    ):
        if approval_keywords is None:
            approval_keywords = ["APPROVED", "LOOKS GOOD", "PERFECT", "DONE"]

        def check_approval(data):
            # Extract the field to check
            text = data
            if isinstance(data, dict):
                text = data.get(approval_field, "")

            # Check for approval keywords
            text_upper = str(text).upper()
            return any(keyword in text_upper for keyword in approval_keywords)

        super().__init__(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=EdgeType.LOOP,  # Use standard LOOP type so topological sort ignores it
            approval_check=check_approval,
            max_iterations=max_iterations,
            **kwargs,
        )


def create_feedback_loop(
    feedback_node_id: str,
    write_node_id: str,
    approval_keywords: Optional[List[str]] = None,
    max_iterations: int = 5,
) -> FeedbackLoopEdge:
    """Helper to create a feedback loop edge.

    Args:
        feedback_node_id: The node providing feedback
        write_node_id: The node to loop back to for improvements
        approval_keywords: Keywords that indicate approval
        max_iterations: Maximum number of iterations

    Returns:
        A configured FeedbackLoopEdge
    """
    return FeedbackLoopEdge(
        source_node_id=feedback_node_id,
        target_node_id=write_node_id,
        approval_keywords=approval_keywords,
        max_iterations=max_iterations,
    )
