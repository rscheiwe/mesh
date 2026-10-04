"""DAG B: fan-out / fan-in after the recommendation is computed.

    allocate -> ( explain (vel agent) | build_card (tool) ) -> respond

``explain`` writes the rationale while ``build_card`` assembles the
recommendation card; ``respond`` waits for both. Reuses DAG A's tools.

Run offline with scripted models:
    uv run python -m examples.dags.parallel_card
"""

import asyncio
from typing import Any

from examples.dags import brief_to_recommendation as dag
from mesh import Executor, MemoryBackend, StateGraph
from mesh.core.graph import ExecutionGraph


def build_card(state: dict[str, Any]) -> dict[str, Any]:
    """Structured card for the UI: requirements, allocation, benchmarks."""
    allocation = state["allocation"]
    card = {
        "requirements": state["slots"],
        "lines": allocation["lines"],
        "benchmarks": {
            line["product_id"]: state["benchmarks"][line["product_id"]]
            for line in allocation["lines"]
        },
        "unspent": allocation["unspent"],
    }
    state["card"] = card
    return {"card_ready": True}


def respond(input: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Join: both the rationale and the card must exist."""
    return {
        "rationale": input.get("content") if isinstance(input, dict) else None,
        "card_ready": bool(state.get("card")),
    }


def build_graph(explain_agent: Any, aggregator: Any = None) -> ExecutionGraph:
    """Slots are pre-filled; the graph runs from benchmarks to respond."""
    graph = StateGraph()
    for name in ("benchmarks", "scale", "viability", "allocate"):
        graph.add_node(name, getattr(dag, name), node_type="tool")
    graph.add_node("explain", explain_agent, node_type="agent")
    graph.add_node("build_card", build_card, node_type="tool")
    graph.add_node("respond", respond, node_type="tool")
    graph.set_entry_point("benchmarks")
    graph.add_sequence(["benchmarks", "scale", "viability", "allocate"])
    graph.add_parallel_edges("allocate", ["explain", "build_card"])
    graph.add_fan_in_edge(["explain", "build_card"], "respond", aggregator=aggregator)
    return graph.compile()


async def _demo() -> None:
    from examples.dags.scripted import scripted_agent, text_turn

    explain = scripted_agent("explain", [text_turn("Commerce Connect leads on CTR.")])
    executor = Executor(build_graph(explain), MemoryBackend())
    slots = {"vertical": "Retail", "kpi": "ctr", "geo": "US", "budget": 30000}
    context = dag.new_context({"slots": slots})
    async for event in executor.execute({}, context):
        print(event.to_dict()["type"], event.node_id or "")
    print(context.state["card"]["lines"])


if __name__ == "__main__":
    asyncio.run(_demo())
