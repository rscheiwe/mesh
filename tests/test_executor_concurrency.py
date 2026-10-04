"""Concurrent runs on one Executor must not see each other's events."""

import asyncio

from mesh import ExecutionContext, Executor, MemoryBackend, StateGraph
from mesh.core.events import EventEmitter, EventType, ExecutionEvent


async def _tagging_tool(context, state):
    for step in range(3):
        await context.emit_event(
            ExecutionEvent(
                type=EventType.CUSTOM_DATA,
                raw_event={"type": "data-run", "data": {"run": state["run"], "step": step}},
            )
        )
        await asyncio.sleep(0.01)
    return {"done": state["run"]}


def _executor(emitter=None):
    graph = StateGraph()
    graph.add_node("work", _tagging_tool, node_type="tool")
    graph.set_entry_point("work")
    return Executor(graph.compile(), MemoryBackend(), event_emitter=emitter)


async def _run_tags(executor, run):
    context = ExecutionContext(graph_id="g", session_id=f"s-{run}", state={"run": run})
    tags = []
    async for event in executor.execute("go", context):
        chunk = event.to_dict()
        if chunk["type"] == "data-run":
            tags.append(chunk["data"]["run"])
    return tags


async def test_concurrent_runs_on_one_executor_do_not_cross_talk():
    executor = _executor()

    run_a, run_b = await asyncio.gather(_run_tags(executor, "a"), _run_tags(executor, "b"))

    assert run_a == ["a", "a", "a"]
    assert run_b == ["b", "b", "b"]


async def test_executor_level_listeners_still_see_every_run():
    emitter = EventEmitter()
    seen = []

    async def listener(event):
        if event.to_dict()["type"] == "data-run":
            seen.append(event.to_dict()["data"]["run"])

    emitter.on(listener)
    executor = _executor(emitter)

    await asyncio.gather(_run_tags(executor, "a"), _run_tags(executor, "b"))

    assert sorted(seen) == ["a", "a", "a", "b", "b", "b"]
