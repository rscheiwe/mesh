---
layout: default
title: Quick Start
nav_order: 3
---

# Quick Start Guide

Get up and running with Mesh in under 5 minutes.

## Installation

```bash
pip install "mesh[all]"
```

## Your First Graph

Create a file `simple_llm.py`:

```python
import asyncio
from dotenv import load_dotenv
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

load_dotenv()

async def main():
    # 1. Build graph
    graph = StateGraph()
    graph.add_node("llm", None, node_type="llm", model="gpt-4")
    graph.add_edge("START", "llm")
    graph.set_entry_point("llm")

    # 2. Compile
    compiled = graph.compile()

    # 3. Create executor
    executor = Executor(compiled, MemoryBackend())

    # 4. Create context
    context = ExecutionContext(
        graph_id="simple",
        session_id="session-1",
        chat_history=[],
        variables={},
        state={}
    )

    # 5. Execute with streaming
    async for event in executor.execute("Tell me a joke", context):
        if event.type == "token":
            print(event.content, end="", flush=True)

    print()  # New line

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python simple_llm.py
```

### Visualize Your Graph

Before running, visualize the graph structure:

```python
# Generate Mermaid diagram
graph.save_visualization(title="my_first_graph")
# Saves to: mesh/visualizations/my_first_graph_{timestamp}.png
```

This creates a diagram showing your graph's structure with color-coded nodes. See the [Visualization Guide](guides/visualization) for more details.

## Common Patterns

### Pattern 1: Simple LLM Chain

```python
graph = StateGraph()

# Add two LLM nodes in sequence
graph.add_node("step1", None, node_type="llm",
               model="gpt-4",
               system_prompt="Analyze this: {{$question}}")
graph.add_node("step2", None, node_type="llm",
               model="gpt-4",
               system_prompt="Summarize: {{step1}}")

# Connect them
graph.add_edge("START", "step1")
graph.add_edge("step1", "step2")

graph.set_entry_point("step1")
```

### Pattern 2: Tool Execution

```python
def search_database(input: dict) -> dict:
    query = input.get("query", "")
    # Your search logic here
    return {"results": ["result1", "result2"]}

graph = StateGraph()
graph.add_node("llm", None, node_type="llm", model="gpt-4")
graph.add_node("search", search_database, node_type="tool")
graph.add_edge("START", "llm")
graph.add_edge("llm", "search")
graph.set_entry_point("llm")
```

### Pattern 3: Conditional Branching

```python
from mesh.nodes import Condition

def is_positive(output: dict) -> bool:
    content = output.get("content", "")
    return "positive" in content.lower()

def is_negative(output: dict) -> bool:
    content = output.get("content", "")
    return "negative" in content.lower()

graph = StateGraph()
graph.add_node("analyzer", None, node_type="llm", model="gpt-4",
               system_prompt="Analyze sentiment: {{$question}}")

# Add conditional branching
graph.add_node("condition", [
    Condition("positive", is_positive, "positive_handler"),
    Condition("negative", is_negative, "negative_handler"),
], node_type="condition")

graph.add_node("positive_handler", None, node_type="llm", model="gpt-4")
graph.add_node("negative_handler", None, node_type="llm", model="gpt-4")

graph.add_edge("START", "analyzer")
graph.add_edge("analyzer", "condition")
graph.add_edge("condition", "positive_handler")
graph.add_edge("condition", "negative_handler")

graph.set_entry_point("analyzer")
```

### Pattern 4: Loop Processing

```python
graph = StateGraph()

# Loop over array items
graph.add_node("loop", None, node_type="loop",
               array_path="$.items",
               max_iterations=100)

# Process each item
graph.add_node("processor", None, node_type="llm",
               model="gpt-4",
               system_prompt="Process: {{$iteration}}")

graph.add_edge("START", "loop")
graph.add_edge("loop", "processor")
graph.set_entry_point("loop")
```

### Pattern 5: Controlled Cycles (Loop Until Done)

```python
def increment_value(input: dict) -> dict:
    value = input.get("value", 0)
    return {"value": value + 1, "done": value + 1 >= 10}

graph = StateGraph()
graph.add_node("increment", increment_value, node_type="tool")

# Create controlled cycle
graph.add_edge("START", "increment")
graph.add_edge(
    "increment",
    "increment",  # Loop back to itself
    is_loop_edge=True,
    loop_condition=lambda state, output: not output.get("done", False),
    max_iterations=50  # Safety limit
)

graph.set_entry_point("increment")
```

**Key points:**
- Mark cycle edges with `is_loop_edge=True`
- Provide `loop_condition` (returns True to continue) and/or `max_iterations`
- Loop condition receives `(state: Dict, output: Dict) -> bool`

### Pattern 6: Multi-Agent Workflow

```python
from vel import Agent as VelAgent

# Create agents
analyzer = VelAgent(
    id="analyzer",
    model={"provider": "openai", "name": "gpt-4"},
)

writer = VelAgent(
    id="writer",
    model={"provider": "openai", "name": "gpt-4"},
)

# Build workflow
graph = StateGraph()
graph.add_node("analyze", analyzer, node_type="agent")
graph.add_node("write", writer, node_type="agent")
graph.add_edge("START", "analyze")
graph.add_edge("analyze", "write")
graph.set_entry_point("analyze")
```

## Using Variables

Variables allow dynamic content in system prompts and configs:

```python
graph = StateGraph()

# Reference user input
graph.add_node("step1", None, node_type="llm",
               system_prompt="User asked: {{$question}}")

# Reference previous node output
graph.add_node("step2", None, node_type="llm",
               system_prompt="Previous result: {{step1}}")

# Reference nested fields
graph.add_node("step3", None, node_type="llm",
               system_prompt="The content was: {{step1.content}}")

# Use global variables
context = ExecutionContext(
    graph_id="test",
    session_id="session-1",
    chat_history=[],
    variables={"user_name": "Alice"},
    state={}
)

graph.add_node("step4", None, node_type="llm",
               system_prompt="Hello {{$vars.user_name}}!")
```

## State Management

### In-Memory (Development)

```python
from mesh.backends import MemoryBackend

backend = MemoryBackend()
executor = Executor(compiled, backend)
```

### SQLite (Production)

```python
from mesh.backends import SQLiteBackend

backend = SQLiteBackend("mesh_state.db")
executor = Executor(compiled, backend)

# State persists across restarts
```

## Event Handling

### Basic Token Streaming

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Handle All Event Types

```python
async for event in executor.execute(input, context):
    if event.type == "execution_start":
        print("[Starting execution]")

    elif event.type == "node_start":
        print(f"\n[Node {event.node_id} starting]")

    elif event.type == "token":
        print(event.content, end="", flush=True)

    elif event.type == "node_complete":
        print(f"\n[Node {event.node_id} completed]")

    elif event.type == "execution_complete":
        print("\n[Execution complete]")
        print(f"Final output: {event.output}")

    elif event.type == "node_error":
        print(f"\n[Error in {event.node_id}]: {event.error}")
```

### Collect Full Response

```python
full_response = ""
async for event in executor.execute(input, context):
    if event.type == "token":
        full_response += event.content
        print(event.content, end="", flush=True)

print(f"\n\nFull response: {full_response}")
```

## FastAPI Integration

Create a streaming API server:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from mesh.streaming import SSEAdapter

app = FastAPI()

class ExecuteRequest(BaseModel):
    input: str
    session_id: str | None = None

@app.post("/execute/stream")
async def execute_stream(request: ExecuteRequest):
    # Create context
    context = ExecutionContext(
        graph_id="api-graph",
        session_id=request.session_id or "default",
        chat_history=[],
        variables={},
        state=await backend.load(request.session_id) or {}
    )

    # Stream via SSE
    adapter = SSEAdapter()
    return adapter.to_streaming_response(
        executor.execute(request.input, context)
    )

# Run with: uvicorn main:app --reload
```

## Next Steps

- [Core Concepts: Graphs](concepts/graphs) - Deep dive into graph structure
- [Core Concepts: Nodes](concepts/nodes) - Learn all 7 node types
- [Guides: Streaming](guides/streaming) - Advanced streaming patterns
- [Guides: Variables](guides/variables) - Variable resolution in depth
- [Integration: Vel](integrations/vel) - Vel agent integration
- [Integration: OpenAI Agents](integrations/openai-agents) - OpenAI SDK integration
- [Examples](examples) - Complete working examples

## Tips & Tricks

### Tip 1: Always Set Entry Point

```python
# ❌ Forgets entry point
graph.add_node("llm", None, node_type="llm")
compiled = graph.compile()  # Error!

# ✅ Sets entry point
graph.add_node("llm", None, node_type="llm")
graph.set_entry_point("llm")
compiled = graph.compile()
```

### Tip 2: Connect All Nodes

```python
# ❌ Orphaned node
graph.add_node("llm1", None, node_type="llm")
graph.add_node("llm2", None, node_type="llm")  # Not connected!
graph.add_edge("START", "llm1")

# ✅ All nodes connected
graph.add_node("llm1", None, node_type="llm")
graph.add_node("llm2", None, node_type="llm")
graph.add_edge("START", "llm1")
graph.add_edge("llm1", "llm2")  # Connected
```

### Tip 3: Use Descriptive Node IDs

```python
# ❌ Generic IDs
graph.add_node("node1", None, node_type="llm")
graph.add_node("node2", None, node_type="llm")

# ✅ Descriptive IDs
graph.add_node("analyzer", None, node_type="llm")
graph.add_node("summarizer", None, node_type="llm")
```

### Tip 4: Load Environment Variables

```python
# ✅ Always load .env
from dotenv import load_dotenv
load_dotenv()

# Then use Mesh
from mesh import StateGraph
```

## Common Errors

### "Graph validation failed: No entry point"

**Solution:** Call `graph.set_entry_point("node_id")` before compiling.

### "Node not found: START"

**Solution:** Don't create a node called "START". It's a reserved keyword that represents the graph entry.

### "Cycle detected in graph"

**Solution:** Mark cycle edges as controlled loops with `is_loop_edge=True` and add controls:

```python
# Mark as loop edge with max_iterations
graph.add_edge("node_a", "node_b", is_loop_edge=True, max_iterations=10)

# Or use a condition
graph.add_edge("node_a", "node_b", is_loop_edge=True,
               loop_condition=lambda state, output: output.get("count", 0) < 5)
```

### "No module named 'openai'"

**Solution:** Install dependencies: `pip install "mesh[agents]"` or `pip install "mesh[all]"`
