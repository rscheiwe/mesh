"""ToolNode: argument injection, output wrapping, sync/async, errors."""

import pytest

from mesh import ExecutionContext
from mesh.nodes.tool import ToolNode


def _ctx():
    return ExecutionContext(
        graph_id="g",
        session_id="s",
        chat_history=[{"role": "user", "content": "hi"}],
        variables={"region": "US"},
        state={"count": 1},
    )


async def test_special_parameters_are_injected():
    seen = {}

    def tool(input, context, state, variables, chat_history):
        seen.update(
            input=input, context=context, state=state, variables=variables, history=chat_history
        )
        return {"ok": True}

    context = _ctx()
    await ToolNode(id="t", tool_fn=tool).execute(input={"a": 1}, context=context)

    assert seen["input"] == {"a": 1}
    assert seen["context"] is context
    assert seen["state"] is context.state
    assert seen["variables"] == {"region": "US"}
    assert seen["history"] == [{"role": "user", "content": "hi"}]


async def test_named_parameters_come_from_input_then_bindings_then_defaults():
    def tool(budget, geo, kpi="ctr"):
        return {"budget": budget, "geo": geo, "kpi": kpi}

    node = ToolNode(id="t", tool_fn=tool, config={"bindings": {"geo": "EMEA", "budget": 1}})
    result = await node.execute(input={"budget": 30000}, context=_ctx())

    assert result.output == {"budget": 30000, "geo": "EMEA", "kpi": "ctr"}


async def test_state_mutations_persist_on_the_context():
    def tool(state):
        state["count"] += 1
        return {}

    context = _ctx()
    await ToolNode(id="t", tool_fn=tool).execute(input={}, context=context)

    assert context.state["count"] == 2


@pytest.mark.parametrize(
    "value, expected", [({"k": 1}, {"k": 1}), (42, {"output": 42}), (None, {"output": None})]
)
async def test_non_dict_results_are_wrapped(value, expected):
    result = await ToolNode(id="t", tool_fn=lambda: value).execute(input={}, context=_ctx())

    assert result.output == expected


async def test_async_tools_are_awaited():
    async def tool(input):
        return {"echo": input["x"]}

    result = await ToolNode(id="t", tool_fn=tool).execute(input={"x": 5}, context=_ctx())

    assert result.output == {"echo": 5}


async def test_tool_errors_name_the_function():
    def broken_lookup():
        raise KeyError("P999")

    with pytest.raises(RuntimeError, match="broken_lookup"):
        await ToolNode(id="t", tool_fn=broken_lookup).execute(input={}, context=_ctx())
