# OpenAI Agents SDK Integration Guide

## Overview

Mesh integrates with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) to provide seamless agent execution with token-by-token streaming.

## Installation

```bash
# Install Mesh with OpenAI Agents SDK support
pip install "mesh[agents]"  # Use quotes in zsh
```

This installs Mesh with the OpenAI Agents SDK.

## Basic Usage

```python
from agents import Agent
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

# Create OpenAI agent
openai_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

# Add to graph
graph = StateGraph()
graph.add_node("agent", openai_agent, node_type="agent")
graph.add_edge("START", "agent")
graph.set_entry_point("agent")

# Execute with streaming
compiled = graph.compile()
executor = Executor(compiled, MemoryBackend())
context = ExecutionContext(
    graph_id="my-graph",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

# Stream token-by-token
async for event in executor.execute("Hello!", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

## How It Works

### Auto-Detection

Mesh's `AgentNode` automatically detects whether an agent is from OpenAI Agents SDK or Vel:

```python
# In mesh/nodes/agent.py
def _detect_agent_type(self, agent):
    # Check for OpenAI Agents SDK (from 'agents' package)
    if "agents" in agent_module.lower() and hasattr(agent, "name") and hasattr(agent, "instructions"):
        return "openai"
    # Check for Vel agent
    elif "vel" in agent_module.lower():
        return "vel"
```

### Event Translation

OpenAI Agents SDK provides streaming events that Mesh translates for consistency:

- `Runner.run_streamed()` → Mesh execution
- Raw response events → Token streaming
- Run item events → Progress updates
- Token streaming preserved
- Chat history maintained
- State persistence handled

### OpenAI Agents SDK API

```python
from agents import Agent, Runner

# Create agent
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

# Run with streaming
result = Runner.run_streamed(
    agent=agent,
    input="Hello!"
)

# Stream events
async for event in result.stream_events():
    if event.type == "raw_response_event":
        print(event.data.delta, end="", flush=True)
```

## Event Types

Mesh automatically translates OpenAI Agents SDK streaming events:

| OpenAI Event | Mesh Event | Description |
|--------------|------------|-------------|
| `raw_response_event` | `token` | Token-by-token LLM generation |
| `run_item_stream_event` (message) | `message_complete` | Message output complete |
| `run_item_stream_event` (tool, in_progress) | `tool_call_start` | Tool call begins |
| `run_item_stream_event` (tool, completed) | `tool_call_complete` | Tool execution complete |
| `agent_updated_stream_event` | (ignored) | Agent state change |

### Example Event Stream

```python
# OpenAI Agents SDK emits events like this:
raw_response_event(data={"delta": "Hello"})
raw_response_event(data={"delta": " world"})
run_item_stream_event(item={"type": "message_output_item", "status": "completed"})

# Mesh translates to:
ExecutionEvent(type=EventType.TOKEN, content='Hello', ...)
ExecutionEvent(type=EventType.TOKEN, content=' world', ...)
ExecutionEvent(type=EventType.MESSAGE_COMPLETE, ...)
```

## Event Handling

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        # Print text tokens as they stream
        print(event.content, end="", flush=True)

    elif event.type == "message_complete":
        # Message generation complete
        print("\n")

    elif event.type == "tool_call_start":
        # Tool execution starting
        tool_name = event.metadata.get('tool_name', 'unknown')
        print(f"\n[Calling: {tool_name}]", end="", flush=True)

    elif event.type == "tool_call_complete":
        # Tool execution complete with result
        result = event.output
        print(f" ✓ Result: {result}\n", end="", flush=True)
```

## Custom System Prompts

Override agent instructions per execution:

```python
graph.add_node(
    "agent",
    openai_agent,
    node_type="agent",
    system_prompt="Process this data: {{$question}}"
)
```

## Troubleshooting

### OpenAI Agents SDK Not Installed

```
ImportError: No module named 'agents'
```

**Solution:**
```bash
pip install mesh[agents]
```

### No Streaming Output

Ensure you're checking for the correct event type:

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        # Print tokens as they stream
        print(event.content, end="", flush=True)
```

### Agent Not Detected

If your OpenAI agent isn't being detected, verify:
1. It's imported from the `agents` package
2. It has both `name` and `instructions` attributes
3. The module name contains "agents"

## Examples

- **Basic Streaming:** `examples/openai_agent_streaming.py`

## API Compatibility

The integration uses the OpenAI Agents SDK from https://openai.github.io/openai-agents-python/

Key features supported:
- Token-by-token streaming via `Runner.run_streamed()`
- Tool calling with progress updates
- Built-in agent loop handling
- Conversation history management

## Next Steps

- Review OpenAI Agents SDK documentation
- Try the example: `python examples/openai_agent_streaming.py`
- Integrate with your own OpenAI agents
