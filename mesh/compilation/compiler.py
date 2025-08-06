"""Base compiler interface and compilation strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from mesh.core.graph import Graph
from mesh.core.node import Node
from mesh.state.state import GraphState


class CompilationStrategy(str, Enum):
    """Available compilation strategies."""

    STATIC = "static"  # Pre-compile entire graph
    DYNAMIC = "dynamic"  # Compile at runtime based on execution path
    HYBRID = "hybrid"  # Mix of static and dynamic


@dataclass
class CompiledGraph:
    """Represents a compiled graph ready for execution."""

    graph: Graph
    execution_plan: List[List[str]]  # List of parallel execution groups
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[str]:
        """Get flat execution order.

        Returns:
            List of node IDs in execution order
        """
        order = []
        for group in self.execution_plan:
            order.extend(group)
        return order


class GraphCompiler(ABC):
    """Abstract base class for graph compilers."""

    @abstractmethod
    async def compile(
        self, graph: Graph, entry_points: Optional[List[str]] = None
    ) -> CompiledGraph:
        """Compile a graph for execution.

        Args:
            graph: Graph to compile
            entry_points: Optional specific entry points

        Returns:
            CompiledGraph ready for execution
        """
        pass

    def validate_graph(self, graph: Graph, strict: bool = True) -> List[str]:
        """Validate graph before compilation.

        Args:
            graph: Graph to validate
            strict: If True, enforce strict terminal node restrictions

        Returns:
            List of validation errors
        """
        return graph.validate(strict_terminal_check=strict)

    def identify_parallel_groups(
        self, graph: Graph, topological_order: List[str]
    ) -> List[List[str]]:
        """Identify nodes that can be executed in parallel.

        Args:
            graph: The graph
            topological_order: Nodes in topological order

        Returns:
            List of parallel execution groups
        """
        groups = []
        processed = set()

        for node_id in topological_order:
            if node_id in processed:
                continue

            # Start a new group with this node
            group = [node_id]
            processed.add(node_id)

            # Find other nodes that can execute in parallel with this group
            # A node can be added to the current group if ALL its predecessors
            # have been processed in PREVIOUS groups (not the current group)
            nodes_in_previous_groups = processed.copy()
            nodes_in_previous_groups.difference_update(group)

            for other_id in topological_order:
                if other_id in processed:
                    continue

                # Check if ALL predecessors are in previous groups
                predecessors = list(graph.get_predecessors(other_id))
                can_add_to_group = True

                for edge, _ in predecessors:
                    if edge.source_node_id not in nodes_in_previous_groups:
                        # Predecessor is either unprocessed or in current group
                        can_add_to_group = False
                        break

                if can_add_to_group:
                    group.append(other_id)
                    processed.add(other_id)

            groups.append(group)

        return groups
