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

### Reasoning Events (o1/o3/Claude Extended Thinking)

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `reasoning_start` | Reasoning block begins | Initialize thinking UI |
| `reasoning_token` | Reasoning tokens streaming | Display thinking process |
| `reasoning_end` | Reasoning block completes | Finalize thinking display |

### Metadata Events

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `response_metadata` | Response metadata available | Track usage, costs, timing |
| `source` | Citations/sources available | Display grounding references (Gemini) |
| `file` | File attachment available | Handle multi-modal content |

### Multi-Step Events

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `step_start` | Agent step begins | Track multi-step execution |
| `step_complete` | Agent step finishes | Capture step metadata |

### Custom Events

| Event Type | When Emitted | Purpose |
|------------|--------------|---------|
| `custom_data` | Custom data event | Extensibility (progress, RLM, etc.) |

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

    # Reasoning (o1/o3/Claude Extended Thinking)
    elif event.type == "reasoning_start":
        print("\n[Thinking...]")

    elif event.type == "reasoning_token":
        print(event.content, end="", flush=True)

    elif event.type == "reasoning_end":
        print("\n[Done thinking]")

    # Metadata
    elif event.type == "response_metadata":
        usage = event.metadata.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        print(f"\n[Usage: {tokens} tokens]")

    elif event.type == "source":
        sources = event.metadata.get("sources", [])
        print(f"\n[Sources: {len(sources)} citations]")

    elif event.type == "file":
        filename = event.metadata.get("name")
        print(f"\n[File: {filename}]")

    # Multi-step execution
    elif event.type == "step_start":
        step = event.metadata.get("step_index", 0)
        print(f"\n[Step {step + 1}]")

    elif event.type == "step_complete":
        usage = event.metadata.get("usage", {})
        print(f"[Step complete - {usage.get('total_tokens', 0)} tokens]")

    # Custom data events
    elif event.type == "custom_data":
        data_type = event.metadata.get("data_type")
        print(f"\n[Custom: {data_type}]")
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

## Advanced Event Handling

### Reasoning Events (o1/o3 Models)

OpenAI's o1/o3 models and Claude's Extended Thinking emit reasoning tokens before the final answer:

```python
reasoning_content = ""
response_content = ""

async for event in executor.execute(input, context):
    if event.type == "reasoning_start":
        print("🤔 Thinking...\n")

    elif event.type == "reasoning_token":
        # Display thinking process
        reasoning_content += event.content
        print(event.content, end="", flush=True)

    elif event.type == "reasoning_end":
        print("\n\n✅ Done thinking\n")

    elif event.type == "token":
        # Regular response tokens
        response_content += event.content
        print(event.content, end="", flush=True)

print(f"\nReasoning: {len(reasoning_content)} chars")
print(f"Response: {len(response_content)} chars")
```

### Usage Tracking

Track token usage and costs in real-time:

```python
total_tokens = 0
total_cost = 0.0

async for event in executor.execute(input, context):
    if event.type == "response_metadata":
        usage = event.metadata.get("usage", {})
        model = event.metadata.get("model_id", "")

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Calculate cost (example rates)
        if "gpt-4" in model:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
        elif "gpt-3.5" in model:
            cost = (prompt_tokens * 0.001 + completion_tokens * 0.002) / 1000

        total_cost += cost
        print(f"\n💰 Tokens: {total_tokens} | Cost: ${cost:.4f}")

    elif event.type == "step_complete":
        # Per-step usage (multi-step agents)
        step_usage = event.metadata.get("usage", {})
        step_tokens = step_usage.get("total_tokens", 0)
        print(f"Step {event.metadata.get('step_index') + 1}: {step_tokens} tokens")
```

### Source Citations (Gemini Grounding)

Handle citations and grounding sources:

```python
citations = []

async for event in executor.execute(input, context):
    if event.type == "source":
        sources = event.metadata.get("sources", [])
        for source in sources:
            url = source.get("url")
            title = source.get("title", "Unknown")
            citations.append({"url": url, "title": title})
            print(f"\n📚 Source: {title}")
            print(f"   URL: {url}")

    elif event.type == "token":
        print(event.content, end="", flush=True)

# Display all citations
if citations:
    print("\n\nReferences:")
    for i, cite in enumerate(citations, 1):
        print(f"{i}. {cite['title']} - {cite['url']}")
```

### Multi-Modal Files

Handle file attachments (images, PDFs, etc.):

```python
files_received = []

async for event in executor.execute(input, context):
    if event.type == "file":
        filename = event.metadata.get("name")
        mime_type = event.metadata.get("mime_type")
        content = event.metadata.get("content")  # Base64 encoded

        files_received.append({
            "name": filename,
            "type": mime_type,
            "data": content
        })

        print(f"\n📎 File: {filename} ({mime_type})")

        # Save or process file
        if mime_type.startswith("image/"):
            import base64
            with open(filename, "wb") as f:
                f.write(base64.b64decode(content))
            print(f"   Saved to {filename}")
```

### Custom Progress Events

Monitor custom progress indicators:

```python
async for event in executor.execute(input, context):
    if event.type == "custom_data":
        data_type = event.metadata.get("data_type")
        data = event.content

        # Handle different custom event types
        if data_type == "data-progress":
            percent = data.get("percent", 0)
            print(f"\rProgress: {percent}%", end="", flush=True)

        elif data_type.startswith("data-rlm-"):
            # RLM middleware events
            if data_type == "data-rlm-step-start":
                step = data.get("step")
                print(f"\n[RLM Step {step}]")

            elif data_type == "data-rlm-complete":
                total_steps = data.get("total_steps")
                print(f"\n[RLM Complete: {total_steps} steps]")

        # Transient events (don't persist)
        if event.metadata.get("transient"):
            # Don't save to history
            pass
```

### Multi-Step Agent Tracking

Track multi-step agent execution:

```python
steps = []
current_step = None

async for event in executor.execute(input, context):
    if event.type == "step_start":
        step_index = event.metadata.get("step_index", 0)
        current_step = {
            "index": step_index,
            "content": "",
            "usage": {}
        }
        print(f"\n{'='*50}")
        print(f"Step {step_index + 1}")
        print(f"{'='*50}")

    elif event.type == "token":
        if current_step:
            current_step["content"] += event.content
        print(event.content, end="", flush=True)

    elif event.type == "step_complete":
        if current_step:
            current_step["usage"] = event.metadata.get("usage", {})
            current_step["finish_reason"] = event.metadata.get("finish_reason")
            steps.append(current_step)

        usage = event.metadata.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        print(f"\n\n✓ Step complete ({tokens} tokens)")

    elif event.type == "execution_complete":
        print(f"\n\nTotal steps: {len(steps)}")
        total_tokens = sum(s["usage"].get("total_tokens", 0) for s in steps)
        print(f"Total tokens: {total_tokens}")
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
