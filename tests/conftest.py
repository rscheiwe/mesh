"""Pytest configuration and fixtures for Mesh tests."""

import pytest
import asyncio
from typing import AsyncGenerator

from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    MemoryBackend,
    NodeRegistry,
)
from mesh.nodes import StartNode, EndNode, LLMNode


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def registry():
    """Create a fresh node registry."""
    return NodeRegistry()


@pytest.fixture
def backend():
    """Create an in-memory backend for testing."""
    return MemoryBackend()


@pytest.fixture
def context():
    """Create a basic execution context."""
    return ExecutionContext(
        graph_id="test-graph",
        session_id="test-session",
        chat_history=[],
        variables={},
        state={},
    )


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    graph = StateGraph()
    graph.add_node("start", StartNode("start"))
    graph.add_node("end", EndNode("end"))
    graph.add_edge("start", "end")
    graph.set_entry_point("start")
    return graph.compile()


@pytest.fixture
async def executor(simple_graph, backend):
    """Create an executor with simple graph."""
    return Executor(simple_graph, backend)
