# Vel Agent Integration Guide

## Overview

Mesh integrates with the [Vel SDK](https://github.com/rscheiwe/vel) to provide seamless agent execution with token-by-token streaming.

## Installation

```bash
pip install "mesh[vel]"  # Use quotes in zsh
```

This installs Mesh with Vel SDK support.

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
    prompt_env="prod",
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

Mesh's `AgentNode` automatically detects whether an agent is from Vel or OpenAI:

```python
# In mesh/nodes/agent.py
def _detect_agent_type(self, agent):
    if "vel" in agent.__class__.__module__.lower():
        return "vel"
    elif hasattr(agent, "id") and hasattr(agent, "instructions"):
        return "openai"
```

### Event Translation

Vel provides provider-agnostic events that Mesh translates for consistency:

- Vel events → Mesh ExecutionEvent
- Token streaming preserved
- Chat history maintained
- State persistence handled

### Actual Vel API

Vel Agent API:

```python
class Agent:
    def __init__(
        self,
        id: str,
        model: Dict[str, Any],  # {"provider": "openai", "name": "gpt-4", ...}
        prompt_env: str = "prod",
        tools: List[str] | None = None,
        session_persistence: "transient" | "persistent" | None = None,
        session_storage: "memory" | "database" | None = None,
        ...
    ):
        pass

    async def run_stream(
        self,
        input: Dict[str, Any],  # {"message": "Hello"}
        session_id: str | None = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield streaming events as dicts"""
        yield {"content": "token"}  # or {"delta": "...", "text": "..."}
```

## Event Types

Mesh automatically translates Vel's Stream Protocol events to a provider-agnostic format. Based on the [official Vel Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol.html):

| Vel Event | Mesh Event | Description |
|-----------|------------|-------------|
| `start` | `message_start` | Message generation begins |
| `text-start` | `message_start` | Text block starts |
| `text-delta` | `token` | Text chunk arrives |
| `text-end` | `message_complete` | Text block completes |
| `tool-input-start` | `tool_call_start` | Tool call begins |
| `tool-input-delta` | `token` | Tool argument chunk |
| `tool-input-available` | `tool_call_start` | Tool arguments ready |
| `tool-output-available` | `tool_call_complete` | Tool execution result |
| `finish-message` | `message_complete` | Message complete (with reason) |
| `error` | `node_error` | Error occurred |

### Example Vel Event Stream

```python
# Text-only response:
{'type': 'start', 'messageId': 'msg_abc123'}
{'type': 'text-start', 'id': 'f81a31b9-19f5-492b-9c36-efe5f9d42246'}
{'type': 'text-delta', 'id': 'f81a31b9-...', 'delta': 'Hello'}
{'type': 'text-delta', 'id': 'f81a31b9-...', 'delta': ' world'}
{'type': 'text-end', 'id': 'f81a31b9-19f5-492b-9c36-efe5f9d42246'}
{'type': 'finish-message', 'finishReason': 'stop'}

# With tool calls:
{'type': 'tool-input-start', 'toolCallId': 'call_123', 'toolName': 'get_weather'}
{'type': 'tool-input-delta', 'toolCallId': 'call_123', 'inputTextDelta': '{"city":'}
{'type': 'tool-input-delta', 'toolCallId': 'call_123', 'inputTextDelta': '"Paris"}'}
{'type': 'tool-input-available', 'toolCallId': 'call_123', 'toolName': 'get_weather', 'input': {'city': 'Paris'}}
{'type': 'tool-output-available', 'toolCallId': 'call_123', 'output': {'temp_f': 72, 'condition': 'sunny'}}
{'type': 'finish-message', 'finishReason': 'tool_calls'}

# Mesh translates to:
ExecutionEvent(type=EventType.MESSAGE_START, metadata={'message_id': 'msg_abc123'})
ExecutionEvent(type=EventType.MESSAGE_START, metadata={'text_block_id': 'f81a31b9-...'})
ExecutionEvent(type=EventType.TOKEN, content='Hello', ...)
ExecutionEvent(type=EventType.TOKEN, content=' world', ...)
ExecutionEvent(type=EventType.MESSAGE_COMPLETE, metadata={'text_block_id': 'f81a31b9-...'})
ExecutionEvent(type=EventType.MESSAGE_COMPLETE, metadata={'finish_reason': 'stop'})
```

### Finish Reasons

Vel provides the following finish reasons in `finish-message` events:
- `stop`: Natural completion
- `length`: Maximum tokens reached
- `tool_calls`: Completed with tool calls
- `content_filter`: Blocked by content filter
- `error`: Error occurred

## Multi-Turn Conversations

```python
# First message
context1 = ExecutionContext(
    graph_id="chat",
    session_id="user-123",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("Hi!", context1):
    # Process events...
    pass

# Second message - preserves context
context2 = ExecutionContext(
    graph_id="chat",
    session_id="user-123",  # Same session
    chat_history=context1.chat_history,  # Preserve history
    variables={},
    state=await backend.load("user-123") or {}
)

async for event in executor.execute("Remember me?", context2):
    # Agent has conversation context
    pass
```

## State Persistence

```python
from mesh.backends import SQLiteBackend

# Use persistent backend
backend = SQLiteBackend("vel_conversations.db")
executor = Executor(compiled, backend)

# State automatically persists across sessions
```

## Custom System Prompts

Override Vel agent instructions per execution:

```python
graph.add_node(
    "agent",
    vel_agent,
    node_type="agent",
    system_prompt="Process this data: {{$question}}"
)
```

## Troubleshooting

### Vel Not Installed

```
ImportError: No module named 'vel'
```

**Solution:**
```bash
pip install mesh[vel]
```

### Event Translation Issues

If Vel's event format differs from expectations, update the translator:

**File:** `mesh/core/events.py`

```python
class VelEventTranslator:
    def from_vel_event(self, vel_event):
        # Adjust based on actual Vel event structure
        pass
```

### No Streaming Output

Ensure you're checking for the correct event types based on the Vel Stream Protocol:

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        # Check if it's tool input streaming
        if event.metadata.get("event_subtype") == "tool_input":
            # Tool argument streaming - optionally display
            pass
        else:
            # Print text tokens as they stream
            print(event.content, end="", flush=True)

    elif event.type == "message_start":
        # Message generation starting (start or text-start)
        pass

    elif event.type == "message_complete":
        # Text block complete (text-end) or message finished (finish-message)
        if "finish_reason" in event.metadata:
            print(f"\n[Finished: {event.metadata['finish_reason']}]")

    elif event.type == "tool_call_start":
        # Tool call begins or arguments ready
        tool_name = event.metadata.get('tool_name', 'unknown')
        if event.metadata.get("input"):
            # tool-input-available: arguments are ready
            print(f"\n[Calling: {tool_name}]", end="", flush=True)
        else:
            # tool-input-start: just beginning
            pass

    elif event.type == "tool_call_complete":
        # tool-output-available: result ready
        result = event.output
        print(f" ✓ Result: {result}\n", end="", flush=True)
```

## Examples

- **Basic Streaming:** `examples/vel_agent_streaming.py`
- **Multi-Turn Chat:** Uncomment multi_turn_conversation() in example
- **With Tools:** See Vel SDK documentation for tool integration

## API Compatibility

If Vel's API differs from the implementation, please:

1. Check Vel's actual API: https://github.com/rscheiwe/vel
2. Update `mesh/nodes/agent.py` accordingly
3. Update `mesh/core/events.py` event translation
4. Submit an issue or PR

## Next Steps

- Review Vel SDK documentation
- Try the example: `python examples/vel_agent_streaming.py`
- Experiment with multi-turn conversations
- Integrate with your own Vel agents
