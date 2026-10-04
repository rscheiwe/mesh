"""DAG E: DAG A from Flowise-style node/edge JSON must behave like the builder version."""

import pytest

from examples.dags import brief_to_recommendation as dag
from mesh import Executor, MemoryBackend
from tests.dags.conftest import collect, completed_nodes
from tests.dags.test_dag_a import ACME_BRIEF, ACME_SLOTS, _agents

SCENARIOS = {
    "free_text": ({"text": ACME_BRIEF}, None, ACME_SLOTS),
    "form_submission": (
        {
            "text": "Vertical: Retail",
            "metadata": {"type": "tool_form_submission", "parameters": {"vertical": "Retail"}},
        },
        {"slots": {"kpi": "ctr", "geo": "US", "budget": 30000}},
        None,
    ),
    "missing_slot": (
        {"text": "A US app, $25,000, in-view."},
        None,
        {**ACME_SLOTS, "vertical": None},
    ),
}


async def _run(builder, payload, state, extract_slots):
    extract, explain = _agents(extract_slots)
    executor = Executor(builder(extract, explain), MemoryBackend())
    context = dag.new_context(None if state is None else {k: dict(v) for k, v in state.items()})
    events = await collect(executor.execute(payload, context))
    return completed_nodes(events), context.state


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
async def test_flow_json_matches_builder_graph(scenario):
    payload, state, extract_slots = SCENARIOS[scenario]

    builder_path, builder_state = await _run(dag.build_graph, payload, state, extract_slots)
    flow_path, flow_state = await _run(dag.build_graph_from_flow, payload, state, extract_slots)

    assert flow_path == builder_path
    assert flow_state.get("allocation") == builder_state.get("allocation")
    assert flow_state["slots"] == builder_state["slots"]
