"""Static compilation strategy implementation."""

from typing import List, Optional

from mesh.compilation.compiler import CompiledGraph, GraphCompiler
from mesh.core.graph import Graph


class StaticCompiler(GraphCompiler):
    """Compiler that pre-compiles the entire graph statically."""

    async def compile(
        self, graph: Graph, entry_points: Optional[List[str]] = None
    ) -> CompiledGraph:
        """Compile graph statically.

        Args:
            graph: Graph to compile
            entry_points: Optional specific entry points

        Returns:
            CompiledGraph with static execution plan
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

        # Get topological order
        topo_order = graph.topological_sort()

        # Filter to only include reachable nodes
        reachable = self._find_reachable_nodes(graph, start_nodes)
        filtered_order = [n for n in topo_order if n in reachable]

        # Identify parallel execution groups
        execution_plan = self.identify_parallel_groups(graph, filtered_order)

        return CompiledGraph(
            graph=graph,
            execution_plan=execution_plan,
            metadata={
                "strategy": "static",
                "entry_points": start_nodes,
                "total_nodes": len(filtered_order),
                "parallel_groups": len(execution_plan),
            },
        )

    def _find_reachable_nodes(self, graph: Graph, start_nodes: List[str]) -> set:
        """Find all nodes reachable from start nodes.

        Args:
            graph: The graph
            start_nodes: Starting node IDs

        Returns:
            Set of reachable node IDs
        """
        reachable = set(start_nodes)
        to_visit = list(start_nodes)

        while to_visit:
            current = to_visit.pop()

            for edge, node in graph.get_successors(current):
                if node.id not in reachable:
                    reachable.add(node.id)
                    to_visit.append(node.id)

        return reachable
