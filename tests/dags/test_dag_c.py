"""DAG C: follow-up Q&A where the model picks tools inside a mesh agent node."""

from vel import ToolSpec

from examples.dags import brief_to_recommendation as dag
from examples.dags import followup_tools as dag_c
from examples.dags.scripted import scripted_agent, text_turn, tool_turn
from mesh import Executor, MemoryBackend, UIMessageStreamAdapter
from tests.dags.conftest import assert_valid_ui_stream, ui_chunks


def _agent(script, state, tools=None):
    return scripted_agent(
        "followup",
        script,
        tools=tools or dag_c.FOLLOWUP_TOOLS,
        tool_context={"state": state},
    )


async def _stream(agent, question, state):
    graph = dag_c.build_graph(agent)
    executor = Executor(graph, MemoryBackend())
    adapter = UIMessageStreamAdapter(graph=graph)
    return await ui_chunks(adapter, executor.execute(question, dag.new_context(state)))


async def test_what_if_tool_call_streams_as_a_tool_part():
    state = dag_c.recommended_state()
    agent = _agent(
        [tool_turn("c1", "what_if", {"budget": 40000}), text_turn("At $40k the bundle grows.")],
        state,
    )

    chunks = await _stream(agent, "What if the budget were $40k?", state)

    assert_valid_ui_stream(chunks)
    tool_input = next(c for c in chunks if c["type"] == "tool-input-available")
    assert tool_input == {
        "type": "tool-input-available",
        "toolCallId": "c1",
        "toolName": "what_if",
        "input": {"budget": 40000},
    }
    output = next(c for c in chunks if c["type"] == "tool-output-available")["output"]
    assert output["slots"]["budget"] == 40000
    assert sum(line["spend"] for line in output["allocation"]["lines"]) > 30000
    assert state["slots"]["budget"] == 30000  # the scenario ran on a copy


async def test_tool_results_reach_the_model_on_the_next_call():
    state = dag_c.recommended_state()
    agent = _agent(
        [tool_turn("c1", "explain_product", {"product_id": "P001"}), text_turn("P001 trails.")],
        state,
    )

    await _stream(agent, "Why not Display Plus?", state)

    second_call = agent._custom_provider.calls[1]
    tool_messages = [m for m in second_call if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "P001" in str(tool_messages[0]["content"])


async def test_failing_tool_is_reported_and_the_model_continues():
    def broken(input, ctx):
        raise RuntimeError("inventory service down")

    state = dag_c.recommended_state()
    broken_tool = ToolSpec(
        name="what_if",
        description="broken",
        input_schema={"type": "object", "properties": {}},
        output_schema={},
        handler=broken,
    )
    agent = _agent(
        [tool_turn("c1", "what_if", {}), text_turn("I could not re-run that scenario.")],
        state,
        tools=[broken_tool],
    )

    chunks = await _stream(agent, "What if?", state)

    assert_valid_ui_stream(chunks)
    assert any(c["type"] == "tool-output-error" for c in chunks)
    assert [c["delta"] for c in chunks if c["type"] == "text-delta"] == [
        "I could not re-run that scenario."
    ]
