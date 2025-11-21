"""Core graph data structures for Mesh.

This module implements the directed graph structures with controlled cycles that
represent agent workflows. Based on Flowise's constructGraphs() pattern.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, TYPE_CHECKING
from pydantic import BaseModel

from mesh.utils.errors import GraphValidationError, CycleDetectedError

if TYPE_CHECKING:
    from mesh.nodes.base import Node


class Edge(BaseModel):
    """Represents a directed connection between two nodes in the graph.

    Supports controlled cycles via loop conditions and max iterations.
    """

    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None

    # Loop control for cycles
    is_loop_edge: bool = False  # Marks this edge as part of a cycle
    loop_condition: Optional[Any] = None  # Callable: (state, output) -> bool to continue
    max_iterations: Optional[int] = None  # Max times this edge can be followed

    class Config:
        arbitrary_types_allowed = True  # Allow callables in loop_condition

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
class GraphMetadata:
    """Metadata about graph structure for FE rendering hints.

    Provides information about node types and execution flow to help
    frontend decide how to render different parts of the graph.

    Attributes:
        node_count: Total number of nodes in graph
        agent_nodes: List of AgentNode IDs
        llm_nodes: List of LLMNode IDs
        tool_nodes: List of ToolNode IDs
        condition_nodes: List of ConditionNode IDs
        loop_nodes: List of LoopNode IDs
        final_nodes: Node IDs that directly precede END node
        intermediate_nodes: Agent/LLM nodes that are not final
        graph_type: Classification (single_agent, multi_agent, tool_chain, hybrid)
        has_cycles: Whether graph contains loop edges
    """
    node_count: int
    agent_nodes: List[str]
    llm_nodes: List[str]
    tool_nodes: List[str]
    condition_nodes: List[str]
    loop_nodes: List[str]
    final_nodes: List[str]
    intermediate_nodes: List[str]
    graph_type: str  # single_agent, multi_agent, tool_chain, hybrid
    has_cycles: bool


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
        metadata: Computed metadata about graph structure for FE hints
    """

    nodes: Dict[str, "Node"]
    edges: List[Edge]
    dependencies: Dict[str, Set[str]]
    children: Dict[str, List[str]]
    starting_nodes: List[str]
    ending_nodes: List[str]
    metadata: Optional[GraphMetadata] = None

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

            # Loop edges are not dependencies - they're optional back-edges
            # Only add non-loop edges to dependencies to avoid deadlock
            if not edge.is_loop_edge:
                dependencies[edge.target].add(edge.source)

            # Add to children for execution routing (including loop edges)
            if edge.target not in children[edge.source]:
                children[edge.source].append(edge.target)

        # Find starting and ending nodes
        starting_nodes = [node_id for node_id, deps in dependencies.items() if not deps]
        ending_nodes = [node_id for node_id, childs in children.items() if not childs]

        # Compute metadata
        metadata = cls._compute_metadata(nodes, edges, children, ending_nodes)

        return cls(
            nodes=nodes,
            edges=edges,
            dependencies=dependencies,
            children=children,
            starting_nodes=starting_nodes,
            ending_nodes=ending_nodes,
            metadata=metadata,
        )

    @staticmethod
    def _compute_metadata(
        nodes: Dict[str, "Node"],
        edges: List[Edge],
        children: Dict[str, List[str]],
        ending_nodes: List[str]
    ) -> GraphMetadata:
        """Compute graph metadata for FE rendering hints.

        Args:
            nodes: Dictionary of nodes
            edges: List of edges
            children: Parent-child relationships
            ending_nodes: Nodes with no outgoing edges

        Returns:
            GraphMetadata with computed fields
        """
        # Categorize nodes by type
        agent_nodes = []
        llm_nodes = []
        tool_nodes = []
        condition_nodes = []
        loop_nodes = []

        for node_id, node in nodes.items():
            node_type = node.__class__.__name__
            if node_type == "AgentNode":
                agent_nodes.append(node_id)
            elif node_type == "LLMNode":
                llm_nodes.append(node_id)
            elif node_type == "ToolNode":
                tool_nodes.append(node_id)
            elif node_type == "ConditionNode":
                condition_nodes.append(node_id)
            elif node_type == "LoopNode":
                loop_nodes.append(node_id)

        # Find final nodes (nodes that directly feed into END or ending nodes)
        final_nodes = []
        for node_id in nodes.keys():
            # Check if this node's children include END or any ending node
            node_children = children.get(node_id, [])
            if "END" in node_children or any(child in ending_nodes for child in node_children):
                # Only include agent/llm nodes as "final"
                if node_id in agent_nodes or node_id in llm_nodes:
                    final_nodes.append(node_id)

        # Intermediate nodes = agent/llm nodes that are not final
        intermediate_nodes = [
            node_id for node_id in (agent_nodes + llm_nodes)
            if node_id not in final_nodes
        ]

        # Determine graph type
        total_agents_llms = len(agent_nodes) + len(llm_nodes)
        if total_agents_llms == 0:
            graph_type = "tool_chain"
        elif total_agents_llms == 1:
            graph_type = "single_agent"
        elif len(agent_nodes) >= 2 or len(llm_nodes) >= 2 or (len(agent_nodes) >= 1 and len(llm_nodes) >= 1):
            graph_type = "multi_agent"
        else:
            graph_type = "hybrid"

        # Check for cycles
        has_cycles = any(edge.is_loop_edge for edge in edges)

        return GraphMetadata(
            node_count=len(nodes),
            agent_nodes=agent_nodes,
            llm_nodes=llm_nodes,
            tool_nodes=tool_nodes,
            condition_nodes=condition_nodes,
            loop_nodes=loop_nodes,
            final_nodes=final_nodes,
            intermediate_nodes=intermediate_nodes,
            graph_type=graph_type,
            has_cycles=has_cycles,
        )

    def validate(self) -> None:
        """Validate graph topology.

        Checks for:
        - Controlled cycles (uncontrolled cycles are errors)
        - At least one starting node
        - All nodes are reachable from starting nodes
        - Loop edges have proper controls (condition or max_iterations)

        Raises:
            GraphValidationError: If validation fails
        """
        # Check for at least one starting node
        if not self.starting_nodes:
            raise GraphValidationError("Graph has no starting nodes (all nodes have dependencies)")

        # Validate loop edges have controls
        for edge in self.edges:
            if edge.is_loop_edge:
                if edge.loop_condition is None and edge.max_iterations is None:
                    raise GraphValidationError(
                        f"Loop edge {edge.source} -> {edge.target} must have "
                        f"either loop_condition or max_iterations specified"
                    )

        # Detect uncontrolled cycles using DFS
        # Controlled cycles (marked with is_loop_edge) are allowed
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_uncontrolled_cycle(node_id: str, path: List[str]) -> bool:
            """DFS helper to detect uncontrolled cycles."""
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            # Visit all children
            for child_id in self.children.get(node_id, []):
                # Check if this edge is a controlled loop edge
                edge_is_controlled = any(
                    e.source == node_id and e.target == child_id and e.is_loop_edge
                    for e in self.edges
                )

                if child_id not in visited:
                    if has_uncontrolled_cycle(child_id, path):
                        return True
                elif child_id in rec_stack and not edge_is_controlled:
                    # Found uncontrolled cycle
                    cycle_start = path.index(child_id)
                    cycle = " -> ".join(path[cycle_start:] + [child_id])
                    raise CycleDetectedError(
                        f"Graph contains an uncontrolled cycle: {cycle}. "
                        f"Mark cycle edges with is_loop_edge=True and add loop controls."
                    )

            path.pop()
            rec_stack.remove(node_id)
            return False

        # Check for uncontrolled cycles starting from each starting node
        for start_node in self.starting_nodes:
            if start_node not in visited:
                if has_uncontrolled_cycle(start_node, []):
                    raise CycleDetectedError("Graph contains an uncontrolled cycle")

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
