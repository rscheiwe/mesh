"""Dynamic compilation strategy implementation."""

from typing import Any, Dict, List, Optional, Set

from mesh.compilation.compiler import CompiledGraph, GraphCompiler
from mesh.core.edge import EdgeType
from mesh.core.graph import Graph
from mesh.core.node import NodeOutput


class DynamicCompiler(GraphCompiler):
    """Compiler that builds execution plan dynamically at runtime."""

    async def compile(
        self, graph: Graph, entry_points: Optional[List[str]] = None
    ) -> CompiledGraph:
        """Create initial compilation for dynamic execution.

        Args:
            graph: Graph to compile
            entry_points: Optional specific entry points

        Returns:
            CompiledGraph with minimal initial plan
        """
        # Validate graph
        errors = self.validate_graph(graph)
        if errors:
            raise ValueError(f"Graph validation failed: {errors}")

        # Determine entry points
        if entry_points:
            start_nodes = entry_points
        else:
            start_nodes = list(graph._start_nodes)

        if not start_nodes:
            raise ValueError("No entry points found in graph")

        # For dynamic compilation, only compile the initial nodes
        execution_plan = [start_nodes]

        return CompiledGraph(
            graph=graph,
            execution_plan=execution_plan,
            metadata={
                "strategy": "dynamic",
                "entry_points": start_nodes,
                "compiled_nodes": len(start_nodes),
            },
        )

    async def get_next_nodes(
        self,
        graph: Graph,
        completed_nodes: Dict[str, NodeOutput],
        current_state: Optional[Any] = None,
    ) -> List[List[str]]:
        """Dynamically determine next nodes to execute.

        Args:
            graph: The graph
            completed_nodes: Map of completed node IDs to their outputs
            current_state: Current execution state

        Returns:
            List of parallel groups to execute next
        """
        next_nodes = set()

        # Check successors of completed nodes
        for node_id, output in completed_nodes.items():
            for edge, successor in graph.get_successors(node_id):
                # Check if all prerequisites are met
                if self._can_execute_node(
                    graph, successor.id, completed_nodes, edge, output
                ):
                    next_nodes.add(successor.id)

        if not next_nodes:
            return []

        # Group nodes that can run in parallel
        return self._group_parallel_nodes(graph, list(next_nodes), completed_nodes)

    def _can_execute_node(
        self,
        graph: Graph,
        node_id: str,
        completed_nodes: Dict[str, NodeOutput],
        triggering_edge: Any,
        triggering_output: NodeOutput,
    ) -> bool:
        """Check if a node can be executed.

        Args:
            graph: The graph
            node_id: Node to check
            completed_nodes: Completed nodes
            triggering_edge: Edge that triggered this check
            triggering_output: Output from triggering node

        Returns:
            True if node can be executed
        """
        # Check edge condition
        if triggering_edge.edge_type == EdgeType.CONDITIONAL:
            if not triggering_edge.evaluate_condition(triggering_output.data):
                return False

        # Check all non-loop predecessors are completed
        for edge, pred in graph.get_predecessors(node_id):
            if edge.edge_type != EdgeType.LOOP and pred.id not in completed_nodes:
                return False

        return True

    def _group_parallel_nodes(
        self, graph: Graph, nodes: List[str], completed_nodes: Dict[str, NodeOutput]
    ) -> List[List[str]]:
        """Group nodes that can execute in parallel.

        Args:
            graph: The graph
            nodes: Nodes to group
            completed_nodes: Already completed nodes

        Returns:
            List of parallel groups
        """
        # Simple implementation: all independent nodes in one group
        # More sophisticated implementations could consider resource constraints
        return [nodes] if nodes else []
