"""Conditional routing: StateGraph.add_conditional_edges, ConditionNode, joins."""

import logging

import pytest

from mesh import ExecutionContext, Executor, MemoryBackend, StateGraph
from mesh.core.events import EventType
from mesh.nodes.condition import Condition, ConditionNode
from mesh.utils.errors import NodeExecutionError


def _ctx():
    return ExecutionContext(graph_id="g", session_id="s")


async def _completed(graph, payload="go"):
    executor = Executor(graph.compile(), MemoryBackend())
    events = [e async for e in executor.execute(payload, _ctx())]
    return [
        e.node_id
        for e in events
        if e.type == EventType.NODE_COMPLETE
        and e.node_id not in (None, "START")
        and not e.node_id.endswith("_condition")
    ]


def _branching_graph(route, **kwargs):
    graph = StateGraph()
    graph.add_node("decide", lambda input: {"route": route}, node_type="tool")
    graph.add_node("left", lambda: {"side": "left"}, node_type="tool")
    graph.add_node("right", lambda: {"side": "right"}, node_type="tool")
    graph.add_conditional_edges(
        "decide", lambda out: out["route"], {"l": "left", "r": "right"}, **kwargs
    )
    graph.set_entry_point("decide")
    return graph


@pytest.mark.parametrize("route, taken", [("l", "left"), ("r", "right")])
async def test_conditional_edges_take_the_selected_branch(route, taken):
    assert await _completed(_branching_graph(route)) == ["decide", taken]


async def test_conditional_edges_fall_back_to_default():
    graph = _branching_graph("nowhere", default="right")

    assert await _completed(graph) == ["decide", "right"]


async def test_conditional_edges_raise_when_condition_fn_fails():
    graph = StateGraph()
    graph.add_node("decide", lambda input: {}, node_type="tool")
    graph.add_node("left", lambda: {}, node_type="tool")
    graph.add_conditional_edges("decide", lambda out: out["missing_key"], {"l": "left"})
    graph.set_entry_point("decide")

    with pytest.raises(NodeExecutionError, match="missing_key"):
        await _completed(graph)


async def test_condition_node_counts_only_required_params():
    seen = []

    def one_required(value, key="unused"):
        seen.append(key)
        return True

    node = ConditionNode(
        id="c",
        conditions=[Condition(name="a", predicate=one_required, target_node="x")],
    )
    result = await node.execute(input={"v": 1}, context=_ctx())

    assert seen == ["unused"]
    assert result.output["fulfilled"] == ["a"]


async def test_condition_node_passes_context_to_two_arg_predicates():
    node = ConditionNode(
        id="c",
        conditions=[
            Condition(
                name="a",
                predicate=lambda value, context: context.state["flag"],
                target_node="x",
            )
        ],
    )
    context = _ctx()
    context.state["flag"] = True

    result = await node.execute(input={}, context=context)

    assert result.output["fulfilled"] == ["a"]


async def test_condition_node_unfulfilled_mode_logs_instead_of_printing(caplog):
    def boom(value):
        raise KeyError("oops")

    node = ConditionNode(
        id="c", conditions=[Condition(name="a", predicate=boom, target_node="x")]
    )
    with caplog.at_level(logging.WARNING, logger="mesh.nodes.condition"):
        result = await node.execute(input={}, context=_ctx())

    assert result.output["unfulfilled"] == ["a"]
    assert "Condition 'a' on node 'c' failed" in caplog.text


def test_condition_node_rejects_unknown_on_error_mode():
    with pytest.raises(ValueError, match="on_error"):
        ConditionNode(id="c", conditions=[], on_error="ignore")


async def test_join_after_exclusive_branches_runs_once_with_live_input():
    graph = _branching_graph("l")
    graph.add_node("join", lambda input: {"joined": input["side"]}, node_type="tool")
    graph.add_edge("left", "join")
    graph.add_edge("right", "join")

    assert await _completed(graph) == ["decide", "left", "join"]


async def test_skipped_branch_descendants_do_not_run():
    graph = _branching_graph("l")
    graph.add_node("after_right", lambda: {}, node_type="tool")
    graph.add_edge("right", "after_right")

    assert await _completed(graph) == ["decide", "left"]


async def test_default_target_shared_with_a_mapped_branch_still_runs():
    graph = StateGraph()
    graph.add_node("decide", lambda input: {"route": "other"}, node_type="tool")
    graph.add_node("handler", lambda: {}, node_type="tool")
    graph.add_conditional_edges(
        "decide", lambda out: out["route"], {"known": "handler"}, default="handler"
    )
    graph.set_entry_point("decide")

    assert await _completed(graph) == ["decide", "handler"]
