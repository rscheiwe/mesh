"""Custom error classes for Mesh."""


class MeshError(Exception):
    """Base exception for all Mesh errors."""

    pass


class GraphValidationError(MeshError):
    """Raised when graph validation fails."""

    pass


class NodeExecutionError(MeshError):
    """Raised when node execution fails."""

    def __init__(self, node_id: str, message: str, original_error: Exception = None):
        self.node_id = node_id
        self.original_error = original_error
        super().__init__(f"Node '{node_id}' execution failed: {message}")


class VariableResolutionError(MeshError):
    """Raised when variable resolution fails."""

    def __init__(self, variable: str, message: str):
        self.variable = variable
        super().__init__(f"Variable '{variable}' resolution failed: {message}")


class CycleDetectedError(GraphValidationError):
    """Raised when a cycle is detected in the graph."""

    pass


class InvalidNodeTypeError(MeshError):
    """Raised when an invalid node type is encountered."""

    pass
