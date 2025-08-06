"""Tests for graph functionality."""

import pytest

from mesh import Edge, Graph
from mesh.core.edge import EdgeType
from mesh.nodes import CustomFunctionNode


class TestGraph:
    """Test graph operations."""

    def test_add_nodes(self):
        """Test adding nodes to graph."""
        graph = Graph()
        node = CustomFunctionNode(lambda x, s: x)

        graph.add_node(node)

        assert node.id in graph._nodes
        assert graph.get_node(node.id) == node

    def test_add_duplicate_node(self):
        """Test that adding duplicate node raises error."""
        graph = Graph()
        node = CustomFunctionNode(lambda x, s: x)

        graph.add_node(node)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node)

    def test_add_edge(self):
        """Test adding edges between nodes."""
        graph = Graph()
        start = CustomFunctionNode(lambda x, s: x)
        end = CustomFunctionNode(lambda x, s: x)

        graph.add_node(start)
        graph.add_node(end)
        graph.add_edge(Edge(start.id, end.id))

        successors = graph.get_successors(start.id)
        assert len(successors) == 1
        assert successors[0][1].id == end.id

    def test_terminal_nodes(self):
        """Test identification of terminal nodes."""
        graph = Graph()
        start = CustomFunctionNode(lambda x, s: x)
        middle = CustomFunctionNode(lambda x, s: x)
        end1 = CustomFunctionNode(lambda x, s: {"branch": 1})
        end2 = CustomFunctionNode(lambda x, s: {"branch": 2})

        # Add all nodes
        for node in [start, middle, end1, end2]:
            graph.add_node(node)

        # Create branching structure
        graph.add_edge(Edge(start.id, middle.id))
        graph.add_edge(Edge(middle.id, end1.id))
        graph.add_edge(Edge(middle.id, end2.id))

        # end1 and end2 should be terminal nodes
        terminal_nodes = graph.get_terminal_nodes()
        assert len(terminal_nodes) == 2
        assert end1.id in terminal_nodes
        assert end2.id in terminal_nodes
        assert start.id not in terminal_nodes
        assert middle.id not in terminal_nodes

    def test_topological_sort(self):
        """Test topological sorting."""
        graph = Graph()
        nodes = [CustomFunctionNode(lambda x, s: x) for _ in range(4)]

        for node in nodes:
            graph.add_node(node)

        # Create chain: 0 -> 1 -> 2 -> 3
        for i in range(3):
            graph.add_edge(Edge(nodes[i].id, nodes[i + 1].id))

        topo_order = graph.topological_sort()

        # Verify order
        node_positions = {node_id: i for i, node_id in enumerate(topo_order)}
        for i in range(3):
            assert node_positions[nodes[i].id] < node_positions[nodes[i + 1].id]

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        graph = Graph()
        nodes = [CustomFunctionNode(lambda x, s: x) for _ in range(3)]

        for node in nodes:
            graph.add_node(node)

        # Create cycle: 0 -> 1 -> 2 -> 0
        graph.add_edge(Edge(nodes[0].id, nodes[1].id))
        graph.add_edge(Edge(nodes[1].id, nodes[2].id))
        graph.add_edge(Edge(nodes[2].id, nodes[0].id))

        with pytest.raises(ValueError, match="contains cycles"):
            graph.topological_sort()

    def test_loop_edge_ignored_in_topo_sort(self):
        """Test that LOOP edges don't affect topological sort."""
        graph = Graph()
        nodes = [CustomFunctionNode(lambda x, s: x) for _ in range(3)]

        for node in nodes:
            graph.add_node(node)

        # Create chain with loop: 0 -> 1 -> 2, with 2 -> 1 loop
        graph.add_edge(Edge(nodes[0].id, nodes[1].id))
        graph.add_edge(Edge(nodes[1].id, nodes[2].id))
        graph.add_edge(Edge(nodes[2].id, nodes[1].id, edge_type=EdgeType.LOOP))

        # Should not raise despite the loop
        topo_order = graph.topological_sort()
        assert len(topo_order) == 3
