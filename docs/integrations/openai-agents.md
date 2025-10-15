---
layout: default
title: OpenAI Agents SDK
parent: Integrations
nav_order: 2
---

# OpenAI Agents SDK Integration

Integrate Mesh with the OpenAI Agents SDK.

## Installation

```bash
pip install "mesh[agents]"
```

## Basic Usage

```python
from agents import Agent
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

# Create OpenAI agent
openai_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

# Add to graph (Vel translation by default)
graph = StateGraph()
graph.add_node("agent", openai_agent, node_type="agent")
graph.add_edge("START", "agent")
graph.set_entry_point("agent")

# Execute
compiled = graph.compile()
executor = Executor(compiled, MemoryBackend())
context = ExecutionContext(
    graph_id="openai-example",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("Hi there!", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

## Native Events

Use OpenAI's native event format:

```python
graph.add_node("agent", openai_agent, node_type="agent",
               use_native_events=True)
```

## See Also

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [Event Translation Guide](event-translation)
