---
layout: default
title: Examples
nav_order: 7
---

# Examples

Complete working examples demonstrating Mesh features.

## Available Examples

All examples are in the [examples/ directory](https://github.com/rscheiwe/mesh/tree/main/examples).

### Basic Examples

#### [simple_agent.py](https://github.com/rscheiwe/mesh/blob/main/examples/simple_agent.py)
Basic LLM usage with token-by-token streaming.

```bash
python examples/simple_agent.py
```

### Agent Integration

#### [vel_agent_streaming.py](https://github.com/rscheiwe/mesh/blob/main/examples/vel_agent_streaming.py)
Vel agent with streaming events.

```bash
pip install "mesh[vel]"
python examples/vel_agent_streaming.py
```

#### [openai_agent_streaming.py](https://github.com/rscheiwe/mesh/blob/main/examples/openai_agent_streaming.py)
OpenAI Agents SDK with Vel translation.

```bash
pip install "mesh[agents]"
python examples/openai_agent_streaming.py
```

### Event Translation

#### [event_translation_comparison.py](https://github.com/rscheiwe/mesh/blob/main/examples/event_translation_comparison.py)
Side-by-side comparison of Vel-translated vs native events.

```bash
python examples/event_translation_comparison.py
```

### React Flow

#### [react_flow_parse.py](https://github.com/rscheiwe/mesh/blob/main/examples/react_flow_parse.py)
Parse and execute Flowise-compatible React Flow JSON.

```bash
python examples/react_flow_parse.py
```

### Server

#### [fastapi_server.py](https://github.com/rscheiwe/mesh/blob/main/examples/fastapi_server.py)
Complete FastAPI server with SSE streaming.

```bash
pip install "mesh[server]"
python examples/fastapi_server.py
# Visit http://localhost:8000/docs
```

## Running Examples

### Setup

```bash
# 1. Install Mesh
pip install -e ".[all]"

# 2. Configure environment
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# 3. Run any example
python examples/simple_agent.py
```

## Example Patterns

### Pattern: Simple Streaming

```python
async for event in executor.execute("Hello", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

### Pattern: Multi-Node Graph

```python
graph.add_node("analyze", analyzer_agent, node_type="agent")
graph.add_node("process", processor_tool, node_type="tool")
graph.add_node("respond", responder_agent, node_type="agent")

graph.add_edge("START", "analyze")
graph.add_edge("analyze", "process")
graph.add_edge("process", "respond")
```

### Pattern: Conditional Flow

```python
from mesh.nodes import Condition

conditions = [
    Condition("positive", check_positive, "positive_handler"),
    Condition("negative", check_negative, "negative_handler"),
]

graph.add_node("router", conditions, node_type="condition")
```

## See Also

- [Quick Start Guide](quick-start) - Build your first graph
- [Guides](guides) - Detailed usage guides
- [API Reference](api-reference) - Complete API documentation
