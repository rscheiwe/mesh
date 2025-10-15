---
layout: default
title: Home
nav_order: 1
description: "Mesh is a Python agent graph orchestration engine for building multi-agent workflows as executable graphs."
permalink: /
---

# Mesh Agent Graph Orchestration
{: .fs-9 }

Build multi-agent workflows as executable graphs with token-by-token streaming, state management, and seamless integration with Vel and OpenAI Agents SDK.
{: .fs-6 .fw-300 }

[Get started now](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/rscheiwe/mesh){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Features

**Graph-Based Workflows**
: Build agent workflows as directed graphs with controlled cycles, using programmatic or declarative interfaces

**Dual API**
: LangGraph-style programmatic builder or React Flow JSON (Flowise-compatible)

**Token-by-Token Streaming**
: Real-time streaming with provider-agnostic events via AsyncIterator

**Event Translation**
: Use Vel's standardized events or provider-native events with opt-in flag

**Multi-Provider Support**
: OpenAI, Anthropic, Google via Vel, plus native OpenAI Agents SDK integration

**7 Core Node Types**
: Start, End, Agent, LLM, Tool, Condition, Loop

**State Persistence**
: Pluggable backends (SQLite, in-memory, custom)

**Variable Resolution**
: Template variables for dynamic workflows (`{{$question}}`, `{{node_id}}`, etc.)

**Production Ready**
: Error handling, retries, structured logging, and SSE streaming

---

## Quick Start

### Installation

```bash
# Basic installation
pip install mesh

# With Vel SDK support
pip install mesh[vel]

# With OpenAI Agents SDK
pip install mesh[agents]

# All features
pip install mesh[all]
```

### Basic Usage

```python
import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

async def main():
    # Build graph
    graph = StateGraph()
    graph.add_node("llm", None, node_type="llm", model="gpt-4")
    graph.add_edge("START", "llm")
    graph.set_entry_point("llm")

    # Compile and execute
    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="my-graph",
        session_id="session-1",
        chat_history=[],
        variables={},
        state={}
    )

    # Stream results
    async for event in executor.execute("What is 2+2?", context):
        if event.type == "token":
            print(event.content, end="", flush=True)

asyncio.run(main())
```

---

## Documentation

<div class="grid">
  <div class="grid-item">
    <h3><a href="getting-started">Getting Started</a></h3>
    <p>Installation and configuration guide</p>
  </div>

  <div class="grid-item">
    <h3><a href="quick-start">Quick Start</a></h3>
    <p>Build your first agent graph in minutes</p>
  </div>

  <div class="grid-item">
    <h3><a href="concepts/graphs">Graphs</a></h3>
    <p>Understanding graph structure and execution</p>
  </div>

  <div class="grid-item">
    <h3><a href="concepts/nodes">Nodes</a></h3>
    <p>7 core node types and how to use them</p>
  </div>

  <div class="grid-item">
    <h3><a href="concepts/events">Events</a></h3>
    <p>Provider-agnostic event streaming</p>
  </div>

  <div class="grid-item">
    <h3><a href="guides/streaming">Streaming</a></h3>
    <p>Token-by-token streaming patterns</p>
  </div>

  <div class="grid-item">
    <h3><a href="guides/variables">Variables</a></h3>
    <p>Template variable resolution</p>
  </div>

  <div class="grid-item">
    <h3><a href="integrations/vel">Vel Integration</a></h3>
    <p>Using Vel agents with Mesh</p>
  </div>

  <div class="grid-item">
    <h3><a href="integrations/openai-agents">OpenAI Agents SDK</a></h3>
    <p>OpenAI Agents SDK integration</p>
  </div>

  <div class="grid-item">
    <h3><a href="integrations/event-translation">Event Translation</a></h3>
    <p>Vel vs native event handling</p>
  </div>

  <div class="grid-item">
    <h3><a href="examples">Examples</a></h3>
    <p>Complete working examples</p>
  </div>

  <div class="grid-item">
    <h3><a href="api-reference">API Reference</a></h3>
    <p>Complete API documentation</p>
  </div>
</div>

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           User Application                  │
│       (FastAPI/Flask/Django)                │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│            MESH LIBRARY                     │
├─────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐      │
│  │  Parsers     │    │  Builders    │      │
│  │  - ReactFlow │    │  - StateGraph│      │
│  └──────┬───────┘    └──────┬───────┘      │
│         └──────────┬─────────┘              │
│                    ▼                         │
│         ┌──────────────────┐                │
│         │  Graph Compiler  │                │
│         └────────┬─────────┘                │
│                  ▼                           │
│         ┌──────────────────┐                │
│         │ Execution Engine │                │
│         │  - Queue-based   │                │
│         │  - Streaming     │                │
│         └────────┬─────────┘                │
│                  ▼                           │
│      ┌───────────────────────┐              │
│      │   Node Implementations│              │
│      │  Agent│LLM│Tool│...   │              │
│      └───────────────────────┘              │
└─────────────────────────────────────────────┘
```

---

## Key Concepts

### StateGraph Builder

LangGraph-style programmatic API for building graphs:

```python
from mesh import StateGraph

graph = StateGraph()
graph.add_node("agent", my_agent, node_type="agent")
graph.add_node("tool", my_function, node_type="tool")
graph.add_edge("START", "agent")
graph.add_edge("agent", "tool")
graph.set_entry_point("agent")

compiled = graph.compile()
```

### Event Streaming

Provider-agnostic events via Vel translation by default:

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "message_complete":
        print("\n[Done]")
```

Or use native provider events:

```python
graph.add_node("agent", agent, node_type="agent", use_native_events=True)
```

### State Management

Persistent state across executions:

```python
from mesh.backends import SQLiteBackend

backend = SQLiteBackend("mesh_state.db")
executor = Executor(compiled, backend)

# State automatically persists
```

---

## Why Mesh?

- **Zero Framework Lock-in**: No LangChain dependency, use any agent SDK
- **Flowise Compatible**: Parse and execute React Flow JSON workflows
- **Production Ready**: State persistence, error handling, retries
- **Streaming First**: Token-by-token streaming with AsyncIterator
- **Provider Agnostic**: Consistent events across OpenAI, Anthropic, Google
- **Flexible**: Programmatic or declarative, Vel or native events

---

## Examples

See the [examples directory](examples) for:

- Simple LLM streaming
- Vel agent integration
- OpenAI Agents SDK integration
- Event translation comparison
- React Flow JSON parsing
- FastAPI server with SSE streaming

---

## License

Mesh is distributed under the [MIT License](https://github.com/rscheiwe/mesh/blob/main/LICENSE).

---

## Credits

Inspired by:
- [Flowise](https://github.com/FlowiseAI/Flowise) - React Flow execution patterns
- [LangGraph](https://github.com/langchain-ai/langgraph) - StateGraph API design
- [Vel](https://github.com/rscheiwe/vel) - Event translation layer
