"""Edge implementation for connecting nodes in the graph."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional
from uuid import uuid4


class EdgeType(str, Enum):
    """Types of edges in the graph."""

    SEQUENTIAL = "sequential"  # Always follows
    CONDITIONAL = "conditional"  # Follows based on condition
    PARALLEL = "parallel"  # Can execute in parallel
    LOOP = "loop"  # Creates a cycle


@dataclass
class Edge:
    """Represents a connection between two nodes in the graph."""

    source_node_id: str
    target_node_id: str
    edge_type: EdgeType = EdgeType.SEQUENTIAL
    condition: Optional[Callable[[Any], bool]] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate_condition(self, data: Any) -> bool:
        """Evaluate the edge condition.

        Args:
            data: Data to evaluate the condition against

        Returns:
            True if the edge should be followed, False otherwise
        """
        if self.edge_type != EdgeType.CONDITIONAL or self.condition is None:
            return True

        try:
            return self.condition(data)
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"Edge(id={self.id}, "
            f"{self.source_node_id} -> {self.target_node_id}, "
            f"type={self.edge_type.value})"
        )
