---
layout: default
title: Events
parent: Concepts
nav_order: 4
---

# Events

Understanding Mesh's event streaming system.

## Overview

Mesh emits **streaming events** during execution to provide real-time feedback. Events are provider-agnostic and consistent across all agent types.

## Event Types

### Execution Events

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `execution_start` | Graph execution begins | Initialize UI, logging |
| `execution_complete` | Graph execution finishes | Finalize, get results |

### Node Events

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `node_start` | Node begins execution | Track progress |
| `node_complete` | Node finishes execution | Capture output |
| `node_error` | Node encounters error | Handle failures |

### Streaming Events

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `token` | Token-by-token streaming | Real-time output |
| `message_start` | Message begins | Initialize message |
| `message_complete` | Message finishes | Finalize message |
| `tool_call_start` | Tool call begins | Track tool usage |
| `tool_call_complete` | Tool result available | Handle tool output |

## Event Structure

### ExecutionEvent

```python
@dataclass
class ExecutionEvent:
    type: EventType              # Event type enum
    node_id: str                 # Source node ID
    content: Optional[str]       # Text content (for tokens)
    output: Optional[Dict]       # Structured output
    metadata: Dict[str, Any]     # Extra information
    timestamp: float             # Event timestamp
    error: Optional[str]         # Error message (if any)
```

## Handling Events

### Basic Pattern

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "execution_complete":
        print(f"\nDone! Output: {event.output}")
```

### All Event Types

```python
async for event in executor.execute(input, context):
    # Execution lifecycle
    if event.type == "execution_start":
        print("[Starting execution]")

    elif event.type == "execution_complete":
        print(f"\n[Complete] Final output: {event.output}")

    # Node lifecycle
    elif event.type == "node_start":
        print(f"\n[{event.node_id}] Starting...")

    elif event.type == "node_complete":
        print(f"[{event.node_id}] ✓ Complete")

    elif event.type == "node_error":
        print(f"[{event.node_id}] ✗ Error: {event.error}")

    # Streaming
    elif event.type == "token":
        print(event.content, end="", flush=True)

    elif event.type == "message_start":
        # Message beginning
        pass

    elif event.type == "message_complete":
        # Message finished
        print()

    # Tool calls
    elif event.type == "tool_call_start":
        tool_name = event.metadata.get("tool_name")
        print(f"\n[Calling tool: {tool_name}]")

    elif event.type == "tool_call_complete":
        result = event.output
        print(f"[Tool result: {result}]")
```

### Collecting Response

```python
full_response = ""
metadata = {}

async for event in executor.execute(input, context):
    if event.type == "token":
        full_response += event.content

    elif event.type == "execution_complete":
        metadata = event.metadata

print(f"Response: {full_response}")
print(f"Metadata: {metadata}")
```

## Event Translation

### Vel-Translated Events (Default)

By default, Mesh uses Vel's event translators for consistent events:

```python
# OpenAI Agents SDK → Vel events → Mesh events
graph.add_node("agent", openai_agent, node_type="agent")

async for event in executor.execute(input, context):
    # Consistent event types regardless of provider
    if event.type == "token":
        print(event.content, end="")
```

### Native Provider Events (Opt-in)

Use native events from the provider:

```python
# OpenAI native events (no translation)
graph.add_node("agent", openai_agent, node_type="agent",
               use_native_events=True)

async for event in executor.execute(input, context):
    # Provider-specific event structure
    print(event)
```

See [Event Translation Guide](../integrations/event-translation) for details.

## Event Metadata

Events include metadata with extra context:

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        # Token-specific metadata
        block_id = event.metadata.get("text_block_id")
        event_subtype = event.metadata.get("event_subtype")

    elif event.type == "message_complete":
        # Message metadata
        finish_reason = event.metadata.get("finish_reason")
        message_id = event.metadata.get("message_id")

    elif event.type == "tool_call_start":
        # Tool metadata
        tool_name = event.metadata.get("tool_name")
        tool_call_id = event.metadata.get("tool_call_id")
```

## SSE Streaming (FastAPI)

Stream events via Server-Sent Events:

```python
from mesh.streaming import SSEAdapter

@app.post("/execute/stream")
async def stream_execute(request: dict):
    adapter = SSEAdapter()
    return adapter.to_streaming_response(
        executor.execute(request["input"], context)
    )
```

Client receives:

```
event: execution_start
data: {"type": "execution_start", "node_id": "START", ...}

event: token
data: {"type": "token", "content": "Hello", ...}

event: token
data: {"type": "token", "content": " world", ...}

event: execution_complete
data: {"type": "execution_complete", "output": {...}, ...}
```

## Best Practices

### 1. Always Handle Errors

```python
async for event in executor.execute(input, context):
    if event.type == "node_error":
        logger.error(f"Error in {event.node_id}: {event.error}")
        # Handle gracefully
```

### 2. Flush Token Output

```python
# ✅ Good: Immediate display
if event.type == "token":
    print(event.content, end="", flush=True)

# ❌ Bad: Buffered output
if event.type == "token":
    print(event.content, end="")  # No flush
```

### 3. Track Progress

```python
total_tokens = 0
async for event in executor.execute(input, context):
    if event.type == "token":
        total_tokens += 1
        print(f"\rTokens: {total_tokens}", end="")
```

### 4. Use Execution Complete

```python
async for event in executor.execute(input, context):
    if event.type == "execution_complete":
        # Final state is available
        final_output = event.output
        execution_time = event.metadata.get("duration_ms")
```

## Event Flow Example

```
execution_start (graph_id=my-graph)
  ↓
node_start (node_id=analyzer)
  ↓
message_start
  ↓
token ("The")
token (" analysis")
token (" shows")
  ↓
message_complete
  ↓
node_complete (node_id=analyzer)
  ↓
node_start (node_id=tool)
  ↓
tool_call_start (tool_name=search)
  ↓
tool_call_complete (output={...})
  ↓
node_complete (node_id=tool)
  ↓
execution_complete (output={...})
```

## See Also

- [Graphs](graphs) - Graph structure
- [Nodes](nodes) - Node types
- [Execution](execution) - Execution model
- [Streaming Guide](../guides/streaming) - Streaming patterns
- [Event Translation](../integrations/event-translation) - Vel vs native events
