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
    """Test that uncontrolled cycles are detected."""
    nodes = {
        "start": StartNode("start"),
        "a": EndNode("a"),
        "b": EndNode("b"),
    }
    edges = [
        Edge(source="start", target="a"),
        Edge(source="a", target="b"),
        Edge(source="b", target="a"),  # Creates uncontrolled cycle a->b->a
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


def test_controlled_loop_with_max_iterations():
    """Test that controlled loops with max_iterations are allowed."""
    nodes = {
        "start": StartNode("start"),
        "loop": EndNode("loop"),
    }
    edges = [
        Edge(source="start", target="loop"),
        Edge(source="loop", target="loop", is_loop_edge=True, max_iterations=10),
    ]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)
    graph.validate()  # Should not raise

    # Loop edge should not create a dependency
    assert graph.get_parents("loop") == {"start"}
    assert "loop" not in graph.get_parents("loop")


def test_controlled_loop_with_condition():
    """Test that controlled loops with conditions are allowed."""
    nodes = {
        "start": StartNode("start"),
        "a": EndNode("a"),
        "b": EndNode("b"),
    }
    edges = [
        Edge(source="start", target="a"),
        Edge(source="a", target="b"),
        Edge(
            source="b",
            target="a",
            is_loop_edge=True,
            loop_condition=lambda state, output: state.get("count", 0) < 5,
        ),
    ]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)
    graph.validate()  # Should not raise


def test_loop_edge_without_controls_fails():
    """Test that loop edges without controls fail validation."""
    nodes = {
        "start": StartNode("start"),
        "a": EndNode("a"),
    }
    edges = [
        Edge(source="start", target="a"),
        Edge(source="a", target="a", is_loop_edge=True),  # No controls!
    ]

    graph = ExecutionGraph.from_nodes_and_edges(nodes, edges)

    with pytest.raises(GraphValidationError, match="loop_condition or max_iterations"):
        graph.validate()
