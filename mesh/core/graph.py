"""Core graph data structures for Mesh.

This module implements the directed acyclic graph (DAG) structures that represent
agent workflows. Based on Flowise's constructGraphs() pattern.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, TYPE_CHECKING
from pydantic import BaseModel

from mesh.utils.errors import GraphValidationError, CycleDetectedError

if TYPE_CHECKING:
    from mesh.nodes.base import Node


class Edge(BaseModel):
    """Represents a directed connection between two nodes in the graph."""

    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None

    def __hash__(self):
        return hash((self.source, self.target, self.source_handle, self.target_handle))


class NodeConfig(BaseModel):
    """Configuration data for a node from React Flow JSON."""

    id: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Dict[str, float]] = None
    parent_node: Optional[str] = None  # For iteration sub-graphs

    class Config:
        arbitrary_types_allowed = True


@dataclass
class ExecutionGraph:
    """Compiled graph ready for execution.

    This is the core data structure that represents a validated, executable workflow.
    It contains all nodes, edges, and dependency information needed for execution.

    Attributes:
        nodes: Mapping of node IDs to Node instances
        edges: List of edges connecting nodes
        dependencies: Mapping of node IDs to their parent node IDs
        children: Mapping of node IDs to their child node IDs
        starting_nodes: List of node IDs with no dependencies (entry points)
        ending_nodes: List of node IDs with no children (exit points)
    """

    nodes: Dict[str, "Node"]
    edges: List[Edge]
    dependencies: Dict[str, Set[str]]
    children: Dict[str, List[str]]
    starting_nodes: List[str]
    ending_nodes: List[str]

    @classmethod
    def from_nodes_and_edges(
        cls, nodes: Dict[str, "Node"], edges: List[Edge]
    ) -> "ExecutionGraph":
        """Factory method to construct ExecutionGraph from nodes and edges.

        Args:
            nodes: Dictionary mapping node IDs to Node instances
            edges: List of Edge objects connecting nodes

        Returns:
            ExecutionGraph: Compiled graph with computed dependencies

        Raises:
            GraphValidationError: If graph structure is invalid
        """
        # Build dependency and children mappings
        dependencies: Dict[str, Set[str]] = {node_id: set() for node_id in nodes.keys()}
        children: Dict[str, List[str]] = {node_id: [] for node_id in nodes.keys()}

        for edge in edges:
            if edge.source not in nodes:
                raise GraphValidationError(
                    f"Edge references non-existent source node: {edge.source}"
                )
            if edge.target not in nodes:
                raise GraphValidationError(
                    f"Edge references non-existent target node: {edge.target}"
                )

            dependencies[edge.target].add(edge.source)
            if edge.target not in children[edge.source]:
                children[edge.source].append(edge.target)

        # Find starting and ending nodes
        starting_nodes = [node_id for node_id, deps in dependencies.items() if not deps]
        ending_nodes = [node_id for node_id, childs in children.items() if not childs]

        return cls(
            nodes=nodes,
            edges=edges,
            dependencies=dependencies,
            children=children,
            starting_nodes=starting_nodes,
            ending_nodes=ending_nodes,
        )

    def validate(self) -> None:
        """Validate graph topology.

        Checks for:
        - Cycles (DAG requirement)
        - At least one starting node
        - All nodes are reachable from starting nodes

        Raises:
            GraphValidationError: If validation fails
        """
        # Check for at least one starting node
        if not self.starting_nodes:
            raise GraphValidationError("Graph has no starting nodes (all nodes have dependencies)")

        # Detect cycles using DFS
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node_id: str) -> bool:
            """DFS helper to detect cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)

            # Visit all children
            for child_id in self.children.get(node_id, []):
                if child_id not in visited:
                    if has_cycle(child_id):
                        return True
                elif child_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        # Check for cycles starting from each starting node
        for start_node in self.starting_nodes:
            if start_node not in visited:
                if has_cycle(start_node):
                    raise CycleDetectedError("Graph contains a cycle")

        # Check that all nodes are reachable from starting nodes
        reachable = self._get_reachable_nodes()
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            raise GraphValidationError(
                f"Graph contains unreachable nodes: {', '.join(unreachable)}"
            )

    def _get_reachable_nodes(self) -> Set[str]:
        """Get all nodes reachable from starting nodes."""
        reachable: Set[str] = set()
        queue = list(self.starting_nodes)

        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue

            reachable.add(node_id)
            queue.extend(self.children.get(node_id, []))

        return reachable

    def get_execution_order(self) -> List[str]:
        """Compute topological sort for execution planning.

        Returns:
            List of node IDs in topologically sorted order

        Note:
            This is used for planning and visualization. The actual executor
            uses dynamic queue-based execution to handle conditional branching.
        """
        # Kahn's algorithm for topological sort
        in_degree = {node_id: len(deps) for node_id, deps in self.dependencies.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for child_id in self.children.get(node_id, []):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        return result

    def get_node(self, node_id: str) -> "Node":
        """Get a node by ID.

        Args:
            node_id: The node identifier

        Returns:
            Node instance

        Raises:
            KeyError: If node does not exist
        """
        return self.nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes

    def get_parents(self, node_id: str) -> Set[str]:
        """Get parent node IDs for a given node."""
        return self.dependencies.get(node_id, set())

    def get_children(self, node_id: str) -> List[str]:
        """Get child node IDs for a given node."""
        return self.children.get(node_id, [])
