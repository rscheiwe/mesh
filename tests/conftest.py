"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import mesh
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    from mesh import Edge, Graph
    from mesh.nodes import CustomFunctionNode

    graph = Graph()
    process = CustomFunctionNode(
        lambda data, state: {"result": data.get("value", 0) * 2}
    )

    graph.add_node(process)

    return graph


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"
