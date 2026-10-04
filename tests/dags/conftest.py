"""Shared helpers for the example-DAG tests."""

import json
from collections.abc import AsyncIterator
from typing import Any

from mesh.core.events import EventType, ExecutionEvent

# Node types that are wiring rather than pipeline steps.
_WIRING_NODE_TYPES = {"start", "condition"}


async def collect(stream: AsyncIterator[ExecutionEvent]) -> list[ExecutionEvent]:
    """Drain an executor stream into a list."""
    return [event async for event in stream]


def completed_nodes(events: list[ExecutionEvent]) -> list[str]:
    """Pipeline nodes in completion order (start and condition wiring excluded)."""
    wiring = {
        e.node_id
        for e in events
        if (e.metadata or {}).get("node_type") in _WIRING_NODE_TYPES
    }
    order: list[str] = []
    for event in events:
        if event.type != EventType.NODE_COMPLETE or not event.node_id:
            continue
        if event.node_id in wiring:
            continue
        if event.node_id not in order:
            order.append(event.node_id)
    return order


def wire_types(events: list[ExecutionEvent]) -> list[str]:
    """Serialized chunk types, in order."""
    return [event.to_dict()["type"] for event in events]


def tool_parts(events: list[ExecutionEvent], tool_name: str) -> list[dict[str, Any]]:
    """Serialized tool-* chunks belonging to calls of ``tool_name``."""
    chunks = [event.to_dict() for event in events]
    call_ids = {
        c["toolCallId"]
        for c in chunks
        if c["type"].startswith("tool-input") and c.get("toolName") == tool_name
    }
    return [c for c in chunks if c.get("toolCallId") in call_ids]


def assert_valid_ui_stream(chunks: list[dict[str, Any]], forbidden: tuple[str, ...] = ()) -> None:
    """Invariants a useChat client relies on (structure, not exact counts)."""
    from mesh.streaming.ui_message_stream import CHUNK_KEYS, DATA_KEYS

    types = [c["type"] for c in chunks]
    assert types[0] == "start" and types.count("start") == 1, types
    assert types[-1] == "finish" and types.count("finish") == 1, types
    assert types.count("error") <= 1, types

    open_blocks: set[str] = set()
    step_open = False
    tool_calls: dict[str, str] = {}
    for chunk in chunks[1:-1]:
        kind = chunk["type"]
        if kind.startswith("data-"):
            assert set(chunk) <= {"type"} | DATA_KEYS, chunk
            continue
        if kind == "error":
            assert set(chunk) == {"type", "errorText"}, chunk
            continue
        assert kind in CHUNK_KEYS, f"non-protocol chunk {kind}"
        assert set(chunk) <= {"type"} | CHUNK_KEYS[kind], chunk
        if kind == "start-step":
            assert not step_open, "nested start-step"
            step_open = True
        elif kind == "finish-step":
            assert step_open, "finish-step without start-step"
            step_open = False
        elif kind.endswith("-start") and kind.split("-")[0] in ("text", "reasoning"):
            assert chunk["id"] not in open_blocks, f"block {chunk['id']} reopened"
            open_blocks.add(chunk["id"])
        elif kind.endswith("-delta") and kind.split("-")[0] in ("text", "reasoning"):
            assert chunk["id"] in open_blocks, f"delta outside block {chunk['id']}"
        elif kind.endswith("-end") and kind.split("-")[0] in ("text", "reasoning"):
            assert chunk["id"] in open_blocks, f"end of unopened block {chunk['id']}"
            open_blocks.discard(chunk["id"])
        elif kind in ("tool-input-start", "tool-input-available"):
            tool_calls.setdefault(chunk["toolCallId"], "open")
        elif kind in ("tool-output-available", "tool-output-error"):
            assert chunk["toolCallId"] in tool_calls, f"output for unknown call {chunk}"
            tool_calls[chunk["toolCallId"]] = "done"
    assert not open_blocks, f"unclosed blocks {open_blocks}"
    assert not step_open, "unclosed step"
    assert all(state == "done" for state in tool_calls.values()), tool_calls

    serialized = json.dumps(chunks, default=str)
    for needle in forbidden:
        assert needle not in serialized, f"{needle!r} leaked into the stream"


async def ui_chunks(adapter, stream) -> list[dict[str, Any]]:
    return [chunk async for chunk in adapter.chunks(stream)]
