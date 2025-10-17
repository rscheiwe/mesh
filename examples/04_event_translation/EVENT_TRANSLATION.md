# Event Translation in Mesh

## Overview

Mesh supports two modes of event handling for agent nodes:

1. **Vel-Translated Events (Default)** - Standardized stream protocol events
2. **Native Provider Events** - Provider-specific event structures

## Why Event Translation?

Different LLM providers (OpenAI, Anthropic, Google) have different streaming event formats. Vel's Translation API provides a unified event protocol across all providers, making your code portable and consistent.

## Default Behavior

By default, Mesh uses Vel's event translation for non-Vel agents:

```python
from agents import Agent
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

# Create OpenAI agent
agent = Agent(name="Assistant", instructions="You are helpful")

# Add to graph (default: use_native_events=False)
graph = StateGraph()
graph.add_node("agent", agent, node_type="agent", config={"model": "gpt-4"})
graph.add_edge("START", "agent")
graph.set_entry_point("agent")

# All events follow Vel stream protocol
compiled = graph.compile()
executor = Executor(compiled, MemoryBackend())
context = ExecutionContext(...)

async for event in executor.execute("Hello", context):
    if event.type == "token":
        # Consistent token streaming across all providers
        print(event.content, end="", flush=True)
    elif event.type == "message_complete":
        # Consistent completion events
        print(f"\nFinished: {event.metadata.get('finish_reason')}")
```

## Using Native Events

To use provider-specific events, set `use_native_events=True`:

```python
graph.add_node(
    "agent",
    agent,
    node_type="agent",
    use_native_events=True,  # Use OpenAI Agents SDK native events
    config={"model": "gpt-4"}
)

async for event in executor.execute("Hello", context):
    # Now you get OpenAI Agents SDK specific event types
    if event.type == "token":
        print(event.content, end="", flush=True)
```

## Event Protocol Reference

### Vel-Translated Events (Default)

When using Vel translation, all providers emit these standardized events:

| Event Type | Description | Fields |
|------------|-------------|--------|
| `start` | Message generation begins | `message_id` |
| `text-start` | Text block starts | `text_block_id` |
| `text-delta` | Text chunk arrives | `delta`, `text_block_id` |
| `text-end` | Text block completes | `text_block_id` |
| `tool-input-start` | Tool call begins | `tool_call_id`, `tool_name` |
| `tool-input-delta` | Tool argument chunk | `tool_call_id`, `input_delta` |
| `tool-input-available` | Tool arguments ready | `tool_call_id`, `tool_name`, `input` |
| `tool-output-available` | Tool execution result | `tool_call_id`, `output` |
| `finish-message` | Message complete | `finish_reason` |
| `error` | Error occurred | `error` |

See [Vel Stream Protocol](https://rscheiwe.github.io/vel/stream-protocol.html) for complete details.

### Mesh Event Mapping

Mesh translates Vel events to its internal event types:

| Vel Event | Mesh EventType | Usage |
|-----------|---------------|-------|
| `text-delta` | `TOKEN` | Token-by-token streaming |
| `text-start` | `MESSAGE_START` | Message begins |
| `text-end`, `finish-message` | `MESSAGE_COMPLETE` | Message ends |
| `tool-input-start` | `TOOL_CALL_START` | Tool begins |
| `tool-output-available` | `TOOL_CALL_COMPLETE` | Tool result |
| `error` | `NODE_ERROR` | Error handling |

## Comparison

### Vel-Translated (Default)

**Pros:**
- ✅ Consistent across all providers (OpenAI, Anthropic, Google)
- ✅ Write once, run anywhere
- ✅ Battle-tested in production
- ✅ Follows standardized stream protocol
- ✅ Automatic updates when Vel adds new providers

**Cons:**
- ⚠️ Requires Vel package
- ⚠️ Slight translation overhead

**Best for:**
- Multi-provider applications
- Consistent event handling
- Production applications requiring portability

### Native Events

**Pros:**
- ✅ Direct access to provider-specific features
- ✅ No translation overhead
- ✅ Works without Vel dependency
- ✅ Full control over provider API

**Cons:**
- ⚠️ Different event structure per provider
- ⚠️ Provider-specific code required
- ⚠️ Must update code when switching providers

**Best for:**
- Single-provider applications
- Performance-critical streaming
- Need for provider-specific features

## Examples

### Multi-Provider with Vel Translation

```python
from mesh import StateGraph

# Same code works with any provider
for provider in ["openai", "anthropic", "google"]:
    agent = create_agent(provider)  # Your agent creation logic

    graph = StateGraph()
    graph.add_node("agent", agent, node_type="agent")
    # ... configure graph

    # Same event handling for all providers
    async for event in executor.execute(input, context):
        if event.type == "token":
            print(event.content, end="")
```

### Tool Calling with Vel Translation

```python
graph.add_node(
    "agent",
    agent,
    node_type="agent",
    config={"model": "gpt-4"}
)

async for event in executor.execute("What's the weather?", context):
    if event.type == "token":
        # Text streaming
        print(event.content, end="", flush=True)

    elif event.type == "tool_call_start":
        # Tool execution starting
        tool_name = event.metadata.get('tool_name')
        print(f"\n[Calling: {tool_name}]", end="")

    elif event.type == "tool_call_complete":
        # Tool result available
        result = event.output
        print(f" ✓ Result: {result}")
```

### Performance-Critical with Native Events

```python
# For performance-critical applications
graph.add_node(
    "agent",
    agent,
    node_type="agent",
    use_native_events=True,  # Direct provider events
)

async for event in executor.execute(input, context):
    # Handle OpenAI Agents SDK specific events
    if event.type == "token":
        print(event.content, end="")
```

## Fallback Behavior

If Vel is not installed or the provider is not supported:

1. Mesh automatically falls back to native events
2. No error is raised
3. Metadata includes `event_translation: "native"` or `event_translation: "vel"`

```python
# Check which mode is being used
async for event in executor.execute(input, context):
    if event.type == "node_complete":
        translation_mode = event.metadata.get("event_translation")
        print(f"Using: {translation_mode} events")
```

## Integration with Vel Agents

**Important:** Vel agents always use Vel's native event streaming, regardless of the `use_native_events` flag:

```python
from vel import Agent as VelAgent

vel_agent = VelAgent(
    id="assistant",
    model={"provider": "openai", "model": "gpt-4"}
)

# Vel agents always emit Vel stream protocol events
graph.add_node("agent", vel_agent, node_type="agent")
# use_native_events flag has no effect on Vel agents
```

## Dependencies

**For Vel-Translated Events:**
```bash
pip install mesh[vel]
```

**For Native OpenAI Events:**
```bash
pip install mesh[agents]
```

**For Both:**
```bash
pip install mesh[all]
```

## See Also

- [Vel Stream Protocol Documentation](https://rscheiwe.github.io/vel/stream-protocol.html)
- [Vel Translation API](../vel/TRANSLATION_API.md)
- [Event Translation Comparison Example](event_translation_comparison.py)
- [Vel Integration Guide](VEL_INTEGRATION.md)
