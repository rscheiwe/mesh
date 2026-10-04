"""UIMessageStreamAdapter: interrupts, disconnects, sources and unknown chunks."""

import asyncio

from mesh import ExecutionContext, Executor, MemoryBackend, StateGraph, UIMessageStreamAdapter
from mesh.core.events import EventType, ExecutionEvent
from tests.dags.conftest import assert_valid_ui_stream, ui_chunks


async def _events(*events):
    for event in events:
        yield event


def _ctx(**state):
    return ExecutionContext(graph_id="g", session_id="s", state=dict(state))


async def test_interrupt_is_reported_without_state():
    graph = StateGraph()
    graph.add_node(
        "draft", lambda state: state.update(secret="s3cr3t") or {"ok": 1}, node_type="tool"
    )
    graph.add_node("publish", lambda: {"published": True}, node_type="tool")
    graph.add_edge("draft", "publish")
    graph.set_entry_point("draft")
    graph.set_interrupt_before("publish")
    executor = Executor(graph.compile(), MemoryBackend())

    chunks = await ui_chunks(UIMessageStreamAdapter(), executor.execute("go", _ctx()))

    assert_valid_ui_stream(chunks, forbidden=("s3cr3t", "_interrupt_state"))
    pauses = [c for c in chunks if c["type"] == "data-mesh-interrupt"]
    assert len(pauses) == 1
    assert pauses[0]["data"]["node_id"] == "publish"
    assert pauses[0]["data"]["position"] == "before"


async def test_closing_the_stream_cancels_the_running_node():
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def slow_model_call():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"never": True}

    graph = StateGraph()
    graph.add_node("slow", slow_model_call, node_type="tool")
    graph.set_entry_point("slow")
    executor = Executor(graph.compile(), MemoryBackend())
    stream = UIMessageStreamAdapter().chunks(executor.execute("go", _ctx()))

    async for chunk in stream:
        if started.is_set():
            break
    await stream.aclose()  # client disconnected

    await asyncio.wait_for(cancelled.wait(), timeout=1)


async def test_vel_sources_become_source_url_chunks():
    source = ExecutionEvent(
        type=EventType.SOURCE,
        raw_event={
            "type": "source",
            "sources": [{"url": "https://a.example", "title": "A"}, {"title": "no url"}],
        },
    )

    chunks = await ui_chunks(UIMessageStreamAdapter(), _events(source))

    assert chunks[1:-1] == [
        {"type": "source-url", "sourceId": "source-0", "url": "https://a.example", "title": "A"}
    ]


async def test_non_protocol_keys_and_chunks_are_dropped():
    events = _events(
        ExecutionEvent(type=EventType.START_STEP, raw_event={"type": "start-step"}),
        ExecutionEvent(
            type=EventType.TEXT_START,
            node_id="a",
            raw_event={"type": "text-start", "id": "t", "metadata": {"node_id": "a"}, "_cursor": 3},
        ),
        ExecutionEvent(
            type=EventType.TEXT_DELTA,
            node_id="a",
            raw_event={"type": "text-delta", "id": "t", "delta": "x"},
        ),
        ExecutionEvent(
            type=EventType.RESPONSE_METADATA, raw_event={"type": "response-metadata", "usage": {}}
        ),
        ExecutionEvent(type=EventType.FINISH, raw_event={"type": "finish"}),
    )

    chunks = await ui_chunks(UIMessageStreamAdapter(), events)

    assert_valid_ui_stream(chunks)  # also closes the dangling text block and step
    assert chunks[2] == {"type": "text-start", "id": "a:t"}
    assert [c["type"] for c in chunks].count("finish") == 1
