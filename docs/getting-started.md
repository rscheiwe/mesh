---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started with Mesh

Complete guide to installing and using Mesh for the first time.

## Installation

### Prerequisites

- Python 3.11 or higher
- pip

### Install from PyPI

```bash
# Basic installation
pip install mesh

# With Vel SDK support
pip install "mesh[vel]"

# With OpenAI Agents SDK
pip install "mesh[agents]"

# With FastAPI server support
pip install "mesh[server]"

# Development installation
pip install "mesh[dev]"

# All features
pip install "mesh[all]"
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/rscheiwe/mesh
cd mesh

# Install in development mode
pip install -e .

# Optional: Install with all dependencies
pip install -e ".[all]"
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Copy the example (if available)
cp .env.example .env

# Or create manually
touch .env
```

#### Required Variables

```bash
# For OpenAI (LLM nodes and OpenAI agents)
OPENAI_API_KEY=sk-...

# OR for Google Gemini (via Vel)
GOOGLE_API_KEY=...

# OR for Anthropic Claude (via Vel)
ANTHROPIC_API_KEY=sk-ant-...
```

**Note:** You only need API keys for the providers you plan to use.

### Loading Environment Variables

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file
```

Or use Mesh's built-in utility:

```python
from mesh.utils import load_env
load_env()
```

## Quick Start

### 1. Simple LLM Node

```python
import asyncio
from dotenv import load_dotenv
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

load_dotenv()

async def main():
    # Build graph
    graph = StateGraph()
    graph.add_node("llm", None, node_type="llm", model="gpt-4")
    graph.add_edge("START", "llm")
    graph.set_entry_point("llm")

    # Compile
    compiled = graph.compile()

    # Execute
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="test",
        session_id="session-1",
        chat_history=[],
        variables={},
        state={}
    )

    # Stream tokens
    async for event in executor.execute("What is 2+2?", context):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "execution_complete":
            print("\n[Done]")

asyncio.run(main())
```

### 2. Using Vel Agents

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
    graph_id="vel-test",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("Hello!", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### 3. Using OpenAI Agents SDK

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
    graph_id="openai-test",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("Hi there!", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### 4. Multi-Node Workflow

```python
# Define a custom tool
def multiply(input: dict) -> dict:
    a = input.get("a", 0)
    b = input.get("b", 0)
    return {"result": a * b}

# Build graph with multiple nodes
graph = StateGraph()
graph.add_node("llm", None, node_type="llm", model="gpt-4",
               system_prompt="Extract two numbers from: {{$question}}")
graph.add_node("tool", multiply, node_type="tool")
graph.add_edge("START", "llm")
graph.add_edge("llm", "tool")
graph.set_entry_point("llm")

compiled = graph.compile()
```

### 5. State Persistence

```python
from mesh.backends import SQLiteBackend

# Use SQLite for persistent state
backend = SQLiteBackend("my_app.db")
executor = Executor(compiled, backend)

# State automatically persists across sessions
context1 = ExecutionContext(
    graph_id="my-graph",
    session_id="user-123",
    chat_history=[],
    variables={},
    state={}
)

# First execution
async for event in executor.execute("Remember: my name is Alice", context1):
    pass

# Load state for next execution
context2 = ExecutionContext(
    graph_id="my-graph",
    session_id="user-123",  # Same session
    chat_history=context1.chat_history,
    variables={},
    state=await backend.load("user-123") or {}
)

# Second execution - remembers Alice
async for event in executor.execute("What's my name?", context2):
    if event.type == "token":
        print(event.content, end="")
```

## Project Structure

A typical Mesh project might look like:

```
my-mesh-project/
├── .env                    # Environment variables
├── requirements.txt        # Dependencies
├── main.py                 # Your application
├── graphs/
│   ├── simple_llm.py      # Graph definitions
│   └── multi_agent.py
├── tools/
│   └── custom_tools.py    # Custom tool functions
└── flows/                  # Optional: React Flow JSONs
    └── workflow.json
```

## Next Steps

- [Quick Start Guide](quick-start) - More examples and patterns
- [Core Concepts: Graphs](concepts/graphs) - Understanding graph structure
- [Core Concepts: Nodes](concepts/nodes) - Learn about node types
- [Guides: Streaming](guides/streaming) - Token-by-token streaming
- [Integration: Vel](integrations/vel) - Deep dive into Vel integration
- [Integration: OpenAI Agents SDK](integrations/openai-agents) - OpenAI integration details

## Troubleshooting

### "No module named 'mesh'"

Make sure you installed the package:

```bash
pip install mesh
# or
pip install -e .  # for development
```

### "Illegal header value" or API Key Errors

Your API key is not set. Check:

1. `.env` file exists and contains `OPENAI_API_KEY=sk-...`
2. You're loading the environment: `load_dotenv()` or `load_env()`
3. The key is valid (no extra spaces, quotes, etc.)

### Import Errors for Vel or OpenAI Agents SDK

Install the optional dependencies:

```bash
# For Vel
pip install "mesh[vel]"

# For OpenAI Agents SDK
pip install "mesh[agents]"
```

### No Streaming Output

Make sure you're checking for the correct event type:

```python
async for event in executor.execute(input, context):
    if event.type == "token":  # Check for "token" not "content"
        print(event.content, end="", flush=True)
```

### "Graph validation failed"

Common issues:
- No entry point set: call `graph.set_entry_point("node_id")`
- Disconnected nodes: all nodes must connect to START
- Uncontrolled cycles detected: mark loop edges with `is_loop_edge=True` and add `max_iterations` or `loop_condition`

## Getting Help

- [GitHub Issues](https://github.com/rscheiwe/mesh/issues)
- [Documentation](https://rscheiwe.github.io/mesh)
- [Examples](examples)
