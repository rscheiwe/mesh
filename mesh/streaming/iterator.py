"""AsyncIterator streaming interface for execution events."""

from typing import AsyncIterator, Any
from mesh.core.executor import Executor
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent


class StreamIterator:
    """AsyncIterator wrapper for execution events.

    This provides a convenient interface for streaming execution events
    as an async iterator.

    Example:
        >>> stream = StreamIterator(executor, input_data, context)
        >>> async for event in stream:
        ...     print(f"Event: {event.type}")
        ...     if event.type == "token":
        ...         print(event.content, end="", flush=True)
    """

    def __init__(
        self,
        executor: Executor,
        input_data: Any,
        context: ExecutionContext,
    ):
        """Initialize stream iterator.

        Args:
            executor: Executor instance
            input_data: Input to the graph
            context: Execution context
        """
        self.executor = executor
        self.input_data = input_data
        self.context = context

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        """Return async iterator from executor."""
        return self.executor.execute(self.input_data, self.context)

    async def collect(self) -> list[ExecutionEvent]:
        """Collect all events into a list.

        Returns:
            List of all execution events
        """
        events = []
        async for event in self:
            events.append(event)
        return events

    async def get_final_output(self) -> Any:
        """Stream all events and return final output.

        Returns:
            Final output from execution
        """
        final_output = None
        async for event in self:
            if event.type == "execution_complete":
                final_output = event.output
        return final_output
