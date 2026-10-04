"""DAG C: follow-up questions after a recommendation, answered by a vel agent with tools.

    START -> followup (vel agent; tools: what_if, explain_product)

Here the model chooses the tools. Each tool re-runs DAG A's deterministic
steps on a copy of the conversation state, so every number still comes from
code. Tools read state through vel's ``tool_context``, which only raw
``ToolSpec(handler=(input, ctx))`` handlers receive (``from_function`` tools
get no ctx).

Run offline with scripted models:
    uv run python -m examples.dags.followup_tools
"""

import asyncio
import copy
from typing import Any

from vel import ToolSpec

from examples.dags import brief_to_recommendation as dag
from mesh import Executor, MemoryBackend, StateGraph
from mesh.core.graph import ExecutionGraph

FOLLOWUP_PROMPT = (
    "Answer the strategist's follow-up about the current recommendation. Use the "
    "tools for any numbers; never compute them yourself."
)


def what_if(input: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Re-run the recommendation with changed budget or impression goal."""
    scenario = copy.deepcopy(ctx["state"])
    for key in ("budget", "impression_goal"):
        if input.get(key) is not None:
            scenario["slots"][key] = input[key]
    for step in (dag.benchmarks, dag.scale, dag.viability, dag.allocate):
        step(scenario)
    return {"slots": scenario["slots"], "allocation": scenario["allocation"]}


def explain_product(input: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Benchmarks, scale and status of one product for the current brief."""
    state = ctx["state"]
    product_id = input["product_id"]
    if product_id not in state.get("candidates", {}):
        return {"error": f"Unknown product {product_id}"}
    in_plan = any(line["product_id"] == product_id for line in state["allocation"]["lines"])
    return {
        "product_id": product_id,
        "benchmarks": state["benchmarks"][product_id],
        "scale": state["candidates"][product_id],
        "rejected_reason": state["rejected"].get(product_id),
        "in_plan": in_plan,
    }


WHAT_IF = ToolSpec(
    name="what_if",
    description="Re-run the recommendation with a different budget or impression goal.",
    input_schema={
        "type": "object",
        "properties": {
            "budget": {"type": "number"},
            "impression_goal": {"type": "integer"},
        },
    },
    output_schema={},
    handler=what_if,
)
EXPLAIN_PRODUCT = ToolSpec(
    name="explain_product",
    description="Explain one product's benchmarks, scale and status for the current brief.",
    input_schema={
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
    output_schema={},
    handler=explain_product,
)
FOLLOWUP_TOOLS = [WHAT_IF, EXPLAIN_PRODUCT]


def build_graph(followup_agent: Any) -> ExecutionGraph:
    """The agent must be built with ``tools=FOLLOWUP_TOOLS`` and
    ``tool_context={"state": <conversation state>}``."""
    graph = StateGraph()
    graph.add_node("followup", followup_agent, node_type="agent")
    graph.set_entry_point("followup")
    return graph.compile()


def recommended_state() -> dict[str, Any]:
    """Conversation state after DAG A recommended for the Acme Shoes brief."""
    state: dict[str, Any] = {
        "slots": {"vertical": "Retail", "kpi": "ctr", "geo": "US", "budget": 30000}
    }
    for step in (dag.benchmarks, dag.scale, dag.viability, dag.allocate):
        step(state)
    return state


async def _demo() -> None:
    from examples.dags.scripted import scripted_agent, text_turn, tool_turn

    state = recommended_state()
    agent = scripted_agent(
        "followup",
        [tool_turn("c1", "what_if", {"budget": 40000}), text_turn("At $40k the bundle grows.")],
        tools=FOLLOWUP_TOOLS,
        tool_context={"state": state},
    )
    executor = Executor(build_graph(agent), MemoryBackend())
    async for event in executor.execute("What if the budget were $40k?", dag.new_context(state)):
        print(event.to_dict()["type"], event.node_id or "")


if __name__ == "__main__":
    asyncio.run(_demo())
