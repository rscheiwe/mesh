"""SSEAdapter formatting."""

from mesh.core.events import EventType, ExecutionEvent
from mesh.streaming.sse import SSEAdapter


def test_sse_event_line_uses_the_wire_type():
    event = ExecutionEvent(type=EventType.TEXT_DELTA, raw_event={"type": "text-delta", "id": "t", "delta": "hi"})

    lines = SSEAdapter().format_event(event).splitlines()

    assert lines[0] == "event: text-delta"
    assert lines[1] == 'data: {"type": "text-delta", "id": "t", "delta": "hi"}'
