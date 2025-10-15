---
layout: default
title: Streaming
parent: Guides
nav_order: 1
---

# Streaming Guide

Token-by-token streaming with Mesh.

## Overview

Mesh provides real-time token-by-token streaming for all agent and LLM nodes via AsyncIterator.

## Basic Streaming

```python
async for event in executor.execute("Tell me a story", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

## Complete Example

```python
import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

async def main():
    graph = StateGraph()
    graph.add_node("llm", None, node_type="llm", model="gpt-4",
                   system_prompt="Tell a short story")
    graph.add_edge("START", "llm")
    graph.set_entry_point("llm")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="streaming",
        session_id="session-1",
        chat_history=[],
        variables={},
        state={}
    )

    print("Story: ", end="")
    async for event in executor.execute("Tell me a story", context):
        if event.type == "token":
            print(event.content, end="", flush=True)
    print()  # New line

asyncio.run(main())
```

## Event Handling Patterns

### Pattern 1: Display Only

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Pattern 2: Collect and Display

```python
full_response = ""
async for event in executor.execute(input, context):
    if event.type == "token":
        full_response += event.content
        print(event.content, end="", flush=True)

# Use full_response later
save_to_database(full_response)
```

### Pattern 3: UI Updates

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        await websocket.send(event.content)
    elif event.type == "execution_complete":
        await websocket.send({"status": "complete"})
```

## SSE Streaming (FastAPI)

```python
from fastapi import FastAPI
from mesh.streaming import SSEAdapter

app = FastAPI()

@app.post("/execute/stream")
async def stream_execute(request: dict):
    adapter = SSEAdapter()
    return adapter.to_streaming_response(
        executor.execute(request["input"], context)
    )
```

See [Quick Start](../quick-start#fastapi-integration) for complete FastAPI example.
