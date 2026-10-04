"""DAG A (brief -> recommendation) on the raw executor, with real vel agents.

Covers routing on conditional edges, a join after exclusive branches,
structured output from an agent node, tool-chain state, and a tool node that
emits a FORM_REQUIRED tool part.
"""

import pytest

from examples.dags import brief_to_recommendation as dag
from examples.dags.scripted import json_turn, scripted_agent, text_turn
from mesh import Executor, MemoryBackend
from tests.dags.conftest import collect, completed_nodes, tool_parts

ACME_BRIEF = (
    "Acme Shoes is a Retail advertiser running in the US with a $30,000 budget. "
    "They care most about click-through rate."
)
ACME_SLOTS = {
    "client_name": "Acme Shoes",
    "vertical": "Retail",
    "kpi": "ctr",
    "geo": "US",
    "budget": 30000,
}
RECOMMEND_PATH = ["benchmarks", "scale", "viability", "allocate", "explain", "guard"]


def _agents(extract_slots=None, explain_text="Commerce Connect leads on CTR."):
    extract = scripted_agent(
        "extract",
        [json_turn(extract_slots)] if extract_slots is not None else [],
        output_type=dag.Slots,
    )
    explain = scripted_agent("explain", [text_turn(explain_text)])
    return extract, explain


async def _run(extract, explain, payload, state=None):
    context = dag.new_context(state)
    executor = Executor(dag.build_graph(extract, explain), MemoryBackend())
    events = await collect(executor.execute(payload, context))
    return events, context


async def test_free_text_brief_recommends_spillover_bundle():
    extract, explain = _agents(ACME_SLOTS)

    events, context = await _run(extract, explain, {"text": ACME_BRIEF})

    assert completed_nodes(events) == ["route_input", "extract", "validate", *RECOMMEND_PATH]
    lines = context.state["allocation"]["lines"]
    assert [line["product_id"] for line in lines] == ["P003", "P006"]
    # P003 fills its risk-adjusted capacity (1,720,000 x 0.83 at $15 CPM), P006 takes the rest.
    assert lines[0]["spend"] == pytest.approx(21_414.0)
    assert lines[1]["spend"] == pytest.approx(8_586.0)
    assert context.state["allocation"]["unspent"] == 0
    assert context.state["rationale"]["guard_passed"] is True


async def test_extract_agent_receives_the_brief_text():
    extract, explain = _agents(ACME_SLOTS)

    await _run(extract, explain, {"text": ACME_BRIEF})

    user_messages = [
        m for m in extract._custom_provider.calls[0] if m.get("role") == "user"
    ]
    assert user_messages[-1]["content"] == ACME_BRIEF


async def test_form_submission_merges_without_calling_the_llm():
    extract, explain = _agents()
    stored = {"slots": {"kpi": "ctr", "geo": "US", "budget": 30000}}
    payload = {
        "text": "Vertical: Retail",
        "metadata": {
            "type": "tool_form_submission",
            "target_tool": "run_recommendation",
            "parameters": {"vertical": "Retail"},
        },
    }

    events, context = await _run(extract, explain, payload, state=stored)

    assert completed_nodes(events) == ["route_input", "merge_form", "validate", *RECOMMEND_PATH]
    assert extract._custom_provider.calls == []
    assert context.state["slots"]["vertical"] == "Retail"


async def test_missing_vertical_emits_form_and_stops():
    extract, explain = _agents({**ACME_SLOTS, "vertical": None})

    events, context = await _run(extract, explain, {"text": "A US app, $25,000, in-view."})

    assert completed_nodes(events) == ["route_input", "extract", "validate", "brief_form"]
    parts = tool_parts(events, "brief_details_form")
    assert [p["type"] for p in parts] == [
        "tool-input-start",
        "tool-input-available",
        "tool-output-available",
    ]
    form = parts[-1]["output"]
    assert form["status"] == "FORM_REQUIRED"
    assert [f["name"] for f in form["form_config"]["fields"]] == ["vertical"]
    assert "allocation" not in context.state
    assert explain._custom_provider.calls == []


async def test_invalid_form_answer_asks_again():
    extract, explain = _agents()
    stored = {"slots": {"vertical": "Retail", "kpi": "ctr", "budget": 30000}}
    payload = {
        "text": "Geo: LATAM",
        "metadata": {"type": "tool_form_submission", "parameters": {"geo": "LATAM"}},
    }

    events, context = await _run(extract, explain, payload, state=stored)

    assert completed_nodes(events)[-1] == "brief_form"
    fields = context.state["pending_form"]["form_config"]["fields"]
    assert [f["name"] for f in fields] == ["geo"]
