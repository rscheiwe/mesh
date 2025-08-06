"""Base node implementation for the mesh framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Status of a node during execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeOutput(BaseModel):
    """Base class for node outputs."""

    node_id: str
    status: NodeStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class NodeConfig:
    """Configuration for a node."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    retry_count: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    can_be_terminal: bool = False  # Whether this node can be a terminal node in the graph


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")
TState = TypeVar("TState")


class Node(ABC, Generic[TInput, TOutput, TState]):
    """Abstract base class for all nodes in the graph."""

    def __init__(self, config: Optional[NodeConfig] = None):
        self.config = config or NodeConfig()
        self._status = NodeStatus.PENDING

    @property
    def id(self) -> str:
        """Get the node ID."""
        return self.config.id

    @property
    def name(self) -> str:
        """Get the node name."""
        return self.config.name or self.__class__.__name__

    @property
    def status(self) -> NodeStatus:
        """Get the current status of the node."""
        return self._status

    @property
    def can_be_terminal(self) -> bool:
        """Check if this node can be a terminal node in the graph."""
        return self.config.can_be_terminal

    @abstractmethod
    async def execute(
        self, input_data: TInput, state: Optional[TState] = None
    ) -> TOutput:
        """Execute the node logic.

        Args:
            input_data: Input data for the node
            state: Optional shared state

        Returns:
            Output from the node execution
        """
        pass

    async def run(
        self, input_data: TInput, state: Optional[TState] = None
    ) -> NodeOutput:
        """Run the node with error handling and status management.

        Args:
            input_data: Input data for the node
            state: Optional shared state

        Returns:
            NodeOutput containing the result or error
        """
        self._status = NodeStatus.RUNNING

        try:
            result = await self.execute(input_data, state)
            self._status = NodeStatus.COMPLETED

            return NodeOutput(
                node_id=self.id,
                status=self._status,
                data=result,
                metadata={"node_name": self.name},
            )

        except Exception as e:
            self._status = NodeStatus.FAILED

            return NodeOutput(
                node_id=self.id,
                status=self._status,
                error=str(e),
                metadata={"node_name": self.name},
            )

    def reset(self) -> None:
        """Reset the node status."""
        self._status = NodeStatus.PENDING

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name})"
