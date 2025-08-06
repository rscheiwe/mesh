"""Graph implementation for orchestrating nodes."""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from uuid import uuid4

from mesh.core.edge import Edge, EdgeType
from mesh.core.node import Node, NodeOutput, NodeStatus


@dataclass
class GraphConfig:
    """Configuration for a graph."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    max_parallel_nodes: int = 10
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """Represents a directed graph of nodes."""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._adjacency: Dict[str, List[Edge]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[Edge]] = defaultdict(list)
        self._start_nodes: Set[str] = set()
        self._terminal_nodes: Set[str] = set()

    @property
    def id(self) -> str:
        """Get the graph ID."""
        return self.config.id

    @property
    def name(self) -> str:
        """Get the graph name."""
        return self.config.name or self.__class__.__name__

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node to add
        """
        if node.id in self._nodes:
            raise ValueError(f"Node with id {node.id} already exists")

        self._nodes[node.id] = node
        self._start_nodes.add(node.id)
        self._terminal_nodes.add(node.id)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.

        Args:
            edge: Edge to add
        """
        if edge.source_node_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_node_id} not found")

        if edge.target_node_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_node_id} not found")

        self._edges.append(edge)
        self._adjacency[edge.source_node_id].append(edge)
        self._reverse_adjacency[edge.target_node_id].append(edge)

        # Update start and terminal nodes
        self._terminal_nodes.discard(edge.source_node_id)
        self._start_nodes.discard(edge.target_node_id)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID.

        Args:
            node_id: ID of the node

        Returns:
            Node if found, None otherwise
        """
        return self._nodes.get(node_id)

    def get_successors(self, node_id: str) -> List[Tuple[Edge, Node]]:
        """Get all successor nodes of a given node.

        Args:
            node_id: ID of the node

        Returns:
            List of (edge, node) tuples
        """
        successors = []
        for edge in self._adjacency.get(node_id, []):
            node = self._nodes.get(edge.target_node_id)
            if node:
                successors.append((edge, node))
        return successors

    def get_predecessors(self, node_id: str) -> List[Tuple[Edge, Node]]:
        """Get all predecessor nodes of a given node.

        Args:
            node_id: ID of the node

        Returns:
            List of (edge, node) tuples
        """
        predecessors = []
        for edge in self._reverse_adjacency.get(node_id, []):
            node = self._nodes.get(edge.source_node_id)
            if node:
                predecessors.append((edge, node))
        return predecessors

    def get_terminal_nodes(self) -> Set[str]:
        """Get all terminal nodes (nodes with no outgoing edges).

        Returns:
            Set of terminal node IDs
        """
        return self._terminal_nodes.copy()

    def get_start_nodes(self) -> Set[str]:
        """Get all start nodes (nodes with no incoming edges).

        Returns:
            Set of start node IDs
        """
        return self._start_nodes.copy()

    def topological_sort(self) -> List[str]:
        """Perform topological sort on the graph.

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If the graph contains cycles
        """
        in_degree = dict.fromkeys(self._nodes, 0)

        for edges in self._adjacency.values():
            for edge in edges:
                if edge.edge_type != EdgeType.LOOP:
                    in_degree[edge.target_node_id] += 1

        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            for edge in self._adjacency.get(node_id, []):
                if edge.edge_type != EdgeType.LOOP:
                    in_degree[edge.target_node_id] -= 1
                    if in_degree[edge.target_node_id] == 0:
                        queue.append(edge.target_node_id)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains cycles")

        return result

    def validate(self, strict_terminal_check: bool = True) -> List[str]:
        """Validate the graph structure.

        Args:
            strict_terminal_check: If True, enforce terminal node restrictions

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for disconnected nodes
        for node_id in self._nodes:
            if (
                node_id not in self._start_nodes
                and node_id not in self._reverse_adjacency
            ):
                errors.append(f"Node {node_id} is disconnected (no incoming edges)")

        # Check that terminal nodes are allowed to be terminal (if strict mode)
        if strict_terminal_check:
            terminal_nodes = self.get_terminal_nodes()
            for node_id in terminal_nodes:
                node = self._nodes[node_id]
                if not node.can_be_terminal:
                    errors.append(
                        f"Node {node.name} (id={node_id}) cannot be a terminal node. "
                        f"Only LLMNode, AgentNode, ToolNode, and MultiToolNode can be terminal nodes."
                    )

        # Check for cycles (except LOOP edges)
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(str(e))

        return errors

    def reset(self) -> None:
        """Reset all nodes in the graph."""
        for node in self._nodes.values():
            node.reset()

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.id}, name={self.name}, "
            f"nodes={len(self._nodes)}, edges={len(self._edges)})"
        )
