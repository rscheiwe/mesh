"""Shared helpers for the example-DAG tests."""

from collections.abc import AsyncIterator
from typing import Any

from mesh.core.events import EventType, ExecutionEvent

# Synthetic nodes that are wiring rather than pipeline steps.
_WIRING_SUFFIXES = ("_condition",)


async def collect(stream: AsyncIterator[ExecutionEvent]) -> list[ExecutionEvent]:
    """Drain an executor stream into a list."""
    return [event async for event in stream]


def completed_nodes(events: list[ExecutionEvent]) -> list[str]:
    """Pipeline nodes in completion order (START and condition wiring excluded)."""
    order: list[str] = []
    for event in events:
        if event.type != EventType.NODE_COMPLETE or not event.node_id:
            continue
        if event.node_id == "START" or event.node_id.endswith(_WIRING_SUFFIXES):
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
