"""Base node protocol and implementation.

This module defines the Node protocol that all nodes must implement,
along with a BaseNode class that provides common functionality like
retry logic and error handling.

Reference: Flowise's node execution pattern from buildAgentflow.ts:795-1228
"""

from typing import Protocol, Any, Dict, Optional, List, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio

from mesh.core.state import ExecutionContext


@dataclass
class NodeResult:
    """Result returned by node execution.

    Attributes:
        output: The primary output of the node
        state: State updates to persist
        chat_history: Chat messages to add to history
        loop_to_node: Node ID to loop back to (for LoopNode)
        max_loops: Maximum number of loop iterations
        metadata: Additional metadata about execution
    """

    output: Any
    state: Dict[str, Any] = field(default_factory=dict)
    chat_history: Optional[List[Dict]] = None
    loop_to_node: Optional[str] = None
    max_loops: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Node(Protocol):
    """Protocol that all nodes must implement.

    This defines the interface for executable nodes in the graph.
    Any class implementing this protocol can be used as a node.
    """

    id: str
    type: str
    config: Dict[str, Any]

    async def execute(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute node logic and return result.

        Args:
            input: Input data from parent nodes
            context: Execution context with state and variables

        Returns:
            NodeResult with output and metadata

        Raises:
            Exception: If execution fails
        """
        ...


class BaseNode(ABC):
    """Base implementation with common functionality.

    This provides:
    - Automatic retry logic with exponential backoff
    - Error handling and wrapping
    - Config management
    - Type detection

    Subclasses must implement _execute_impl() with their core logic.
    """

    def __init__(self, id: str, config: Dict[str, Any] = None):
        """Initialize base node.

        Args:
            id: Unique identifier for this node
            config: Configuration dictionary
        """
        self.id = id
        self.type = self.__class__.__name__
        self.config = config or {}
        self.retry_config = self.config.get("retry", {})

    @abstractmethod
    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Subclasses implement core logic here.

        Args:
            input: Input data from parent nodes
            context: Execution context

        Returns:
            NodeResult with output and metadata
        """
        pass

    async def execute(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute node with retry logic.

        This wraps the subclass implementation with retry logic
        based on the retry configuration.

        Args:
            input: Input data from parent nodes
            context: Execution context

        Returns:
            NodeResult from _execute_impl

        Raises:
            Exception: If all retry attempts fail
        """
        max_retries = self.retry_config.get("max_retries", 0)
        retry_delay = self.retry_config.get("delay", 1.0)
        retry_backoff = self.retry_config.get("backoff", 2.0)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._execute_impl(input, context)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = retry_delay * (retry_backoff ** attempt)
                    await asyncio.sleep(delay)
                    continue
                # All retries exhausted
                raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise RuntimeError("Execution failed without error")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.id}')"


class PassThroughNode(BaseNode):
    """Simple node that passes input through as output.

    Useful for testing and as a base for simple transformations.
    """

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Pass input through as output."""
        return NodeResult(output=input)
