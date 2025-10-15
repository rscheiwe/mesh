"""Tests for graph data structures."""

import pytest
from mesh.core.graph import ExecutionGraph, Edge
from mesh.nodes import StartNode, EndNode
from mesh.utils.errors import GraphValidationError, CycleDetectedError


def test_create_simple_graph():
    """Test creating a simple graph."""
    nodes = {
        "start": StartNode("start"),
        "end": EndNode("end"),
    }
    edges = [Edge(source="start", target="end")]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.starting_nodes == ["start"]
    assert graph.ending_nodes == ["end"]


def test_graph_validation_passes():
    """Test that valid graph passes validation."""
    nodes = {
        "start": StartNode("start"),
        "end": EndNode("end"),
    }
    edges = [Edge(source="start", target="end")]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)
    graph.validate()  # Should not raise


def test_graph_detects_cycle():
    """Test that cycles are detected."""
    nodes = {
        "a": StartNode("a"),
        "b": EndNode("b"),
    }
    edges = [
        Edge(source="a", target="b"),
        Edge(source="b", target="a"),  # Creates cycle
    ]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)

    with pytest.raises(CycleDetectedError):
        graph.validate()


def test_graph_execution_order():
    """Test topological sort for execution order."""
    nodes = {
        "start": StartNode("start"),
        "middle": EndNode("middle"),
        "end": EndNode("end"),
    }
    edges = [
        Edge(source="start", target="middle"),
        Edge(source="middle", target="end"),
    ]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)
    order = graph.get_execution_order()

    assert order == ["start", "middle", "end"]


def test_get_parents_and_children():
    """Test getting node relationships."""
    nodes = {
        "start": StartNode("start"),
        "middle": EndNode("middle"),
        "end": EndNode("end"),
    }
    edges = [
        Edge(source="start", target="middle"),
        Edge(source="middle", target="end"),
    ]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)

    assert graph.get_parents("middle") == {"start"}
    assert graph.get_children("start") == ["middle"]
    assert graph.get_parents("start") == set()
    assert graph.get_children("end") == []
