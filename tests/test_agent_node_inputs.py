"""AgentNode input options: vel sessions, caller-owned history, input parsing."""

import pytest

import mesh.nodes.agent as agent_module
from examples.dags.scripted import scripted_agent, text_turn
from mesh import ExecutionContext
from mesh.nodes.agent import AgentNode


def _ctx(history=None):
    return ExecutionContext(graph_id="g", session_id="same-session", chat_history=history or [])


def _roles(call):
    return [m["role"] for m in call]


async def test_stable_session_sends_one_instruction_per_call():
    """vel >= 0.5 keeps run-scoped system messages out of session history."""
    agent = scripted_agent("a", [text_turn("1"), text_turn("2")])
    node = AgentNode(id="n", agent=agent, system_prompt="Be brief.")
    context = _ctx()

    for turn in ("hi", "again"):
        await node.execute(input=turn, context=context)

    assert _roles(agent._custom_provider.calls[1]).count("system") == 1


async def test_use_session_false_sends_one_instruction_per_call():
    agent = scripted_agent("a", [text_turn("1"), text_turn("2"), text_turn("3")])
    node = AgentNode(id="n", agent=agent, system_prompt="Be brief.", use_session=False)
    context = _ctx()

    for turn in ("hi", "again", "third"):
        await node.execute(input=turn, context=context)

    for call in agent._custom_provider.calls:
        assert _roles(call) == ["system", "user"]


async def test_messages_mode_sends_the_callers_history():
    history = [
        {"role": "user", "content": "Acme Shoes, Retail, US, $30k, CTR."},
        {"role": "assistant", "content": "Commerce Connect plus Attention Builder."},
    ]
    agent = scripted_agent("a", [text_turn("Because of capacity.")])
    node = AgentNode(id="n", agent=agent, input_mode="messages")

    await node.execute(input="Why a bundle?", context=_ctx(history))

    sent = [
        (m["role"], m["content"]) for m in agent._custom_provider.calls[0] if m["role"] != "system"
    ]
    assert sent == [
        ("user", "Acme Shoes, Retail, US, $30k, CTR."),
        ("assistant", "Commerce Connect plus Attention Builder."),
        ("user", "Why a bundle?"),
    ]


async def test_auto_parse_input_false_skips_the_hidden_parse_call(monkeypatch):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("input parser was called")

    monkeypatch.setattr(agent_module, "parse_natural_language_input", must_not_run)
    agent = scripted_agent("a", [text_turn("ok")])
    node = AgentNode(
        id="n",
        agent=agent,
        system_prompt="Brand {{$input.brand}} in {{$input.geo}}.",
        auto_parse_input=False,
    )

    result = await node.execute(input="Acme in the US", context=_ctx())

    assert result.output == {"content": "ok"}


def test_unknown_input_mode_is_rejected():
    with pytest.raises(ValueError, match="input_mode"):
        AgentNode(id="n", agent=scripted_agent("a", []), input_mode="history")
