"""SSEAdapter formatting."""

from mesh.core.events import EventType, ExecutionEvent
from mesh.streaming.sse import SSEAdapter


def test_sse_event_line_uses_the_wire_type():
    raw = {"type": "text-delta", "id": "t", "delta": "hi"}
    event = ExecutionEvent(type=EventType.TEXT_DELTA, raw_event=raw)

    lines = SSEAdapter().format_event(event).splitlines()

    assert lines[0] == "event: text-delta"
    assert lines[1] == 'data: {"type": "text-delta", "id": "t", "delta": "hi"}'
