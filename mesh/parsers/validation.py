"""Validation utilities for graph construction."""

from mesh.core.graph import ExecutionGraph
from mesh.utils.errors import GraphValidationError


def validate_graph(graph: ExecutionGraph) -> None:
    """Validate graph structure and topology.

    This is a convenience function that wraps graph.validate()
    and can be extended with additional validation logic.

    Args:
        graph: ExecutionGraph to validate

    Raises:
        GraphValidationError: If validation fails
    """
    graph.validate()


def validate_node_config(node_type: str, config: dict) -> None:
    """Validate node configuration.

    Args:
        node_type: Type of node
        config: Configuration dictionary

    Raises:
        GraphValidationError: If config is invalid
    """
    # Basic validation - can be extended
    if not isinstance(config, dict):
        raise GraphValidationError(
            f"Node config must be a dictionary, got {type(config)}"
        )

    # Type-specific validation
    if node_type == "agent" and "agent" not in config:
        raise GraphValidationError(
            "Agent node requires 'agent' in config"
        )

    if node_type == "tool" and "tool" not in config:
        raise GraphValidationError(
            "Tool node requires 'tool' in config"
        )

    if node_type == "condition" and "conditions" not in config:
        raise GraphValidationError(
            "Condition node requires 'conditions' in config"
        )
