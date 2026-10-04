"""DAG B: fan-out/fan-in (explain | build_card) -> respond."""

import asyncio
import time

import pytest

from examples.dags import brief_to_recommendation as dag
from examples.dags import parallel_card
from examples.dags.scripted import scripted_agent, text_turn
from mesh import Executor, MemoryBackend, UIMessageStreamAdapter
from tests.dags.conftest import assert_valid_ui_stream, collect, completed_nodes, ui_chunks

SLOTS = {"vertical": "Retail", "kpi": "ctr", "geo": "US", "budget": 30000}


def _context():
    return dag.new_context({"slots": dict(SLOTS)})


async def test_fan_in_waits_for_both_branches():
    explain = scripted_agent("explain", [text_turn("Commerce Connect leads on CTR.")])
    executor = Executor(parallel_card.build_graph(explain), MemoryBackend())
    context = _context()

    events = await collect(executor.execute({}, context))

    order = completed_nodes(events)
    assert order[-1] == "respond"
    assert {"explain", "build_card"} <= set(order[:-1])
    assert order.count("respond") == 1
    assert context.state["card"]["lines"][0]["product_id"] == "P003"


async def test_fan_in_aggregator_shapes_the_join_input():
    seen = {}

    def aggregate(results):
        seen.update(results)
        return {"content": results["explain"]["content"], "branches": sorted(results)}

    explain = scripted_agent("explain", [text_turn("Rationale.")])
    graph = parallel_card.build_graph(explain, aggregator=aggregate)
    executor = Executor(graph, MemoryBackend())
    context = _context()

    events = await collect(executor.execute({}, context))

    assert set(seen) == {"explain", "build_card"}
    respond = [e for e in events if e.node_id == "respond" and e.output]
    assert respond[-1].output["rationale"] == "Rationale."


@pytest.mark.xfail(
    strict=True,
    reason="Executor runs add_parallel_edges branches sequentially (known gap, raised to owner)",
)
async def test_parallel_branches_run_concurrently(monkeypatch):
    async def slow_card(state):
        await asyncio.sleep(0.3)
        return parallel_card.build_card(state)

    async def slow_explain(state):
        await asyncio.sleep(0.3)
        return {"content": "Rationale."}

    monkeypatch.setattr(parallel_card, "build_card", slow_card)
    graph = parallel_card.build_graph(scripted_agent("explain", []))
    graph.nodes["explain"] = _tool("explain", slow_explain)
    executor = Executor(graph, MemoryBackend())

    started = time.perf_counter()
    await collect(executor.execute({}, _context()))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5, f"branches ran sequentially ({elapsed:.2f}s)"


async def test_parallel_stream_is_protocol_clean():
    explain = scripted_agent("explain", [text_turn("Commerce Connect leads on CTR.")])
    graph = parallel_card.build_graph(explain)
    executor = Executor(graph, MemoryBackend())

    chunks = await ui_chunks(UIMessageStreamAdapter(graph=graph), executor.execute({}, _context()))

    assert_valid_ui_stream(chunks)


def _tool(node_id, fn):
    from mesh.nodes.tool import ToolNode

    return ToolNode(id=node_id, tool_fn=fn)
