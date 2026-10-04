"""Scripted vel provider for running real vel agents offline.

Follows vel's own test pattern (vel/tests/test_harness/test_suspend_resume_m3.py):
a ``BaseProvider`` subclass replays canned provider events, and the agent is
pointed at it through ``agent._custom_provider``. The agent loop, structured
output and event emission are real vel code; only the model is scripted.
"""

import json
from typing import Any

from vel import Agent, ToolSpec
from vel.events import (
    FinishMessageEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
)
from vel.providers import BaseProvider


class ScriptedProvider(BaseProvider):
    """Replays one batch of provider events per model call.

    Records the messages of every call so tests can assert what the model saw.
    """

    name = "scripted"

    def __init__(self, script: list[list[Any]]):
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    async def stream(self, messages, model, tools, generation_config=None):
        # Snapshot: vel keeps mutating the list it passed in after the call
        self.calls.append([dict(m) for m in messages])
        if not self._script:
            raise RuntimeError("ScriptedProvider: script exhausted")
        for event in self._script.pop(0):
            if isinstance(event, Exception):
                raise event
            yield event

    async def generate(self, messages, model, tools, generation_config=None):
        return {"done": True}


def text_turn(text: str, block_id: str = "b") -> list[Any]:
    """One model call that answers with plain text."""
    return [
        TextStartEvent(block_id=block_id),
        TextDeltaEvent(block_id=block_id, delta=text),
        TextEndEvent(block_id=block_id),
        FinishMessageEvent(finish_reason="stop"),
    ]


def json_turn(payload: dict[str, Any]) -> list[Any]:
    """One model call that answers with a JSON object (for output_type agents)."""
    return text_turn(json.dumps(payload))


def tool_turn(call_id: str, tool_name: str, args: dict[str, Any]) -> list[Any]:
    """One model call that requests a single tool call."""
    return [
        ToolInputAvailableEvent(tool_call_id=call_id, tool_name=tool_name, input=args),
        FinishMessageEvent(finish_reason="tool_calls"),
    ]


def scripted_agent(
    agent_id: str,
    script: list[list[Any]],
    output_type: type | None = None,
    tools: list[ToolSpec] | None = None,
) -> Agent:
    """Build a real vel Agent whose model calls are served by ``script``."""
    agent = Agent(
        id=agent_id,
        model={"provider": "scripted", "model": "scripted"},
        output_type=output_type,
        tools=tools or [],
    )
    agent._custom_provider = ScriptedProvider(script)
    return agent
