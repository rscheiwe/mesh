"""Server-Sent Events (SSE) adapter for web streaming.

This module provides adapters to convert Mesh execution events
into SSE format for web applications.
"""

from typing import AsyncIterator, Optional, Any
import json

from mesh.core.events import ExecutionEvent


class SSEAdapter:
    """Adapt Mesh execution events to Server-Sent Events format.

    This adapter converts Mesh ExecutionEvent objects into SSE format
    for streaming to web clients.

    Example with FastAPI:
        >>> from fastapi.responses import StreamingResponse
        >>>
        >>> @app.post("/execute/stream")
        >>> async def execute_stream(request: ExecuteRequest):
        ...     executor = Executor(graph, backend)
        ...     context = ExecutionContext(...)
        ...
        ...     adapter = SSEAdapter()
        ...     return adapter.to_streaming_response(
        ...         executor.execute(request.input, context)
        ...     )
    """

    def __init__(self, event_name: Optional[str] = None):
        """Initialize SSE adapter.

        Args:
            event_name: Optional default event name for SSE events
        """
        self.event_name = event_name

    async def event_generator(
        self, event_stream: AsyncIterator[ExecutionEvent]
    ) -> AsyncIterator[str]:
        """Generate SSE-formatted events from execution stream.

        Args:
            event_stream: AsyncIterator of ExecutionEvent objects

        Yields:
            SSE-formatted event strings
        """
        async for event in event_stream:
            # Convert event to SSE format
            sse_event = self.format_event(event)
            yield sse_event

    def format_event(self, event: ExecutionEvent) -> str:
        """Format a single event as SSE.

        Args:
            event: ExecutionEvent to format

        Returns:
            SSE-formatted string
        """
        # Build SSE event
        lines = []

        # Event type
        event_name = self.event_name or event.type
        lines.append(f"event: {event_name}")

        # Event data
        data = event.to_dict()
        data_json = json.dumps(data)
        lines.append(f"data: {data_json}")

        # Add blank line to complete event
        lines.append("")

        return "\n".join(lines) + "\n"

    def to_sse_response(
        self, event_stream: AsyncIterator[ExecutionEvent]
    ) -> "EventSourceResponse":
        """Convert event stream to SSE response (sse-starlette).

        Args:
            event_stream: AsyncIterator of ExecutionEvent objects

        Returns:
            EventSourceResponse for FastAPI/Starlette

        Raises:
            ImportError: If sse-starlette is not installed
        """
        try:
            from sse_starlette.sse import EventSourceResponse
        except ImportError:
            raise ImportError(
                "sse-starlette not installed. "
                "Install with: pip install sse-starlette"
            )

        async def event_generator():
            async for event in event_stream:
                yield {
                    "event": event.type,
                    "data": json.dumps(event.to_dict()),
                }

        return EventSourceResponse(event_generator())

    def to_streaming_response(
        self, event_stream: AsyncIterator[ExecutionEvent]
    ) -> "StreamingResponse":
        """Convert event stream to generic streaming response.

        This is compatible with FastAPI's StreamingResponse for SSE.

        Args:
            event_stream: AsyncIterator of ExecutionEvent objects

        Returns:
            StreamingResponse for FastAPI

        Raises:
            ImportError: If FastAPI is not installed
        """
        try:
            from fastapi.responses import StreamingResponse
        except ImportError:
            raise ImportError(
                "FastAPI not installed. Install with: pip install fastapi"
            )

        return StreamingResponse(
            self.event_generator(event_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )


class WebSocketAdapter:
    """Adapter for WebSocket streaming (future implementation).

    This will provide WebSocket support as an alternative to SSE.
    """

    def __init__(self):
        """Initialize WebSocket adapter."""
        pass

    async def stream_to_websocket(
        self, event_stream: AsyncIterator[ExecutionEvent], websocket: Any
    ):
        """Stream events to WebSocket connection.

        Args:
            event_stream: AsyncIterator of ExecutionEvent objects
            websocket: WebSocket connection

        Note:
            This is a placeholder for future WebSocket support.
        """
        async for event in event_stream:
            await websocket.send_json(event.to_dict())
