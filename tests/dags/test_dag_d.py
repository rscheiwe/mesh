"""DAG D: failure paths through DAG A on the raw executor.

A failing tool, a failing routing function and a failing model call must stop
the run at that node, report which node failed, and run nothing downstream.
"""

import pytest

from examples.dags import brief_to_recommendation as dag
from examples.dags.scripted import json_turn, scripted_agent, text_turn
from mesh import Executor, MemoryBackend
from mesh.core.events import EventType
from mesh.utils.errors import NodeExecutionError
from tests.dags.conftest import completed_nodes
from tests.dags.test_dag_a import ACME_BRIEF, ACME_SLOTS


async def _run_until_error(extract, explain):
    executor = Executor(dag.build_graph(extract, explain), MemoryBackend())
    events = []
    with pytest.raises(NodeExecutionError) as error:
        async for event in executor.execute({"text": ACME_BRIEF}, dag.new_context()):
            events.append(event)
    return events, error.value


def _agents(extract_script=None, explain_script=None):
    extract = scripted_agent(
        "extract", extract_script or [json_turn(ACME_SLOTS)], output_type=dag.Slots
    )
    explain = scripted_agent("explain", explain_script or [text_turn("ok")])
    return extract, explain


async def test_failing_tool_stops_the_run_at_that_node(monkeypatch):
    def broken_scale(state):
        raise ValueError("inventory feed unavailable")

    monkeypatch.setattr(dag, "scale", broken_scale)

    events, error = await _run_until_error(*_agents())

    assert error.node_id == "scale"
    assert "inventory feed unavailable" in str(error)
    assert completed_nodes(events)[-1] == "benchmarks"
    node_errors = [e for e in events if e.type == EventType.NODE_ERROR]
    assert {e.node_id for e in node_errors} == {"scale"}


async def test_failing_routing_function_raises_instead_of_misrouting(monkeypatch):
    monkeypatch.setattr(dag, "route_input", lambda input, state: {"no_kind": True})

    events, error = await _run_until_error(*_agents())

    assert error.node_id == "route_input_condition"
    assert "kind" in str(error)
    assert completed_nodes(events) == ["route_input"]


async def test_failing_model_call_stops_the_run_at_the_agent_node():
    extract, explain = _agents(
        explain_script=[[RuntimeError("model provider returned 503")]]
    )

    events, error = await _run_until_error(extract, explain)

    assert error.node_id == "explain"
    assert completed_nodes(events)[-1] == "allocate"
    assert "guard" not in completed_nodes(events)
