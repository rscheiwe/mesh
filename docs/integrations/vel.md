---
layout: default
title: Vel SDK
parent: Integrations
nav_order: 1
---

# Vel SDK Integration

Integrate Mesh with the Vel agent runtime.

## Installation

```bash
pip install "mesh[vel]"
```

## Basic Usage

```python
from vel import Agent as VelAgent
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

# Create Vel agent
vel_agent = VelAgent(
    id="assistant",
    model={
        "provider": "openai",
        "name": "gpt-4",
        "temperature": 0.7,
    },
)

# Add to graph
graph = StateGraph()
graph.add_node("agent", vel_agent, node_type="agent")
graph.add_edge("START", "agent")
graph.set_entry_point("agent")

# Execute with streaming
compiled = graph.compile()
executor = Executor(compiled, MemoryBackend())
context = ExecutionContext(
    graph_id="vel-example",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("Hello!", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

## See Also

- [Vel Documentation](https://rscheiwe.github.io/vel)
- [Vel Integration Guide](https://github.com/rscheiwe/mesh/blob/main/examples/VEL_INTEGRATION.md)
