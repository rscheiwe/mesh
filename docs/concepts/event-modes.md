# Event Modes

Event Modes provide fine-grained control over how nodes emit events during execution. This is critical for multi-agent workflows where you want different nodes to surface differently in your UI - some streaming to chat, others running silently, and others providing progress indicators.

## Overview

Every node in Mesh supports an `event_mode` parameter that controls event emission behavior. This gives you complete control over what surfaces in your application UI versus what happens silently in the background.

**Available Modes:**
- **`full`** (default): All events stream normally to chat
- **`status_only`**: Only progress indicators (node start/complete events)
- **`transient_events`**: ALL events emitted but prefixed with `data-{node_type}-node-*` for custom UI rendering
- **`silent`**: No events emitted at all

## Use Cases

### Multi-Agent Orchestration

In a multi-agent workflow like `START → AGENT_1 → AGENT_2 → END`, you typically want:
- **AGENT_1**: Silent or status-only execution (intermediate processing)
- **AGENT_2**: Full streaming (final output to user)

```python
from mesh import ExecutionGraph
from mesh.nodes import AgentNode, StartNode

# Intermediate agent - runs silently
agent_1 = AgentNode(
    id="researcher",
    agent=research_agent,
    event_mode="silent"  # No events
)

# Final agent - streams to chat
agent_2 = AgentNode(
    id="writer",
    agent=writer_agent,
    event_mode="full"  # Full streaming
)
```

### Progress Indicators

For long-running tool or LLM nodes, use `status_only` to show progress without cluttering the chat:

```python
tool_node = ToolNode(
    id="data_processor",
    tool_fn=process_large_dataset,
    event_mode="status_only"  # Only start/complete events
)
```

### Custom UI Rendering

Use `transient_events` mode when you want ALL events but need to render them differently (e.g., in a dedicated progress panel vs. chat):

```python
llm_node = LLMNode(
    id="analyzer",
    model="gpt-4",
    event_mode="transient_events"  # All events prefixed with data-llm-node-*
)
```

## Event Transformation

### Transient Events Mode

When `event_mode="transient_events"`, ALL events are transformed with a special prefix pattern:

**Pattern:** `data-{node_type}-node-{event_name}`

**Examples:**
- `text-delta` → `data-agent-node-text-delta`
- `tool-input-start` → `data-agent-node-tool-input-start`
- `tool-output-available` → `data-agent-node-tool-output-available`
- `text-delta` (from LLM) → `data-llm-node-text-delta`
- Node start → `data-tool-node-start`

These prefixed events follow the Vercel AI SDK / Vel pattern and signal to your frontend that these events should be rendered differently (e.g., in a progress sidebar instead of chat).

### Event Structure

Transient events are wrapped in a `CUSTOM_DATA` event with metadata:

```python
{
    "type": "CUSTOM_DATA",
    "node_id": "agent_1",
    "metadata": {
        "data_event_type": "data-agent-node-text-delta",
        "data": {
            "content": "Hello",
            "delta": "Hello",  # Compatibility field
            "node_id": "agent_1"
        },
        "transient": True,  # Not saved to history
        "original_event_type": "token"  # For debugging
    }
}
```

## Node-Specific Event Prefixes

Each node type uses its own prefix in transient mode:

| Node Type | Prefix | Examples |
|-----------|--------|----------|
| Agent | `data-agent-node-*` | `data-agent-node-text-delta`, `data-agent-node-tool-input-start` |
| LLM | `data-llm-node-*` | `data-llm-node-text-delta`, `data-llm-node-start` |
| Tool | `data-tool-node-*` | `data-tool-node-start`, `data-tool-node-complete` |
| Condition | `data-condition-node-*` | `data-condition-node-evaluating`, `data-condition-node-branch-selected` |
| ForEach | `data-foreach-node-*` | `data-foreach-node-iteration-start`, `data-foreach-node-item-complete` |
| Loop | `data-loop-node-*` | `data-loop-node-iteration-start`, `data-loop-node-next-loop` |
| Start | `data-start-node-*` | `data-start-node-begin` |

## Configuration

### Python API

```python
from mesh.nodes import AgentNode, LLMNode, ToolNode

# Silent execution
agent = AgentNode(
    id="processor",
    agent=my_agent,
    event_mode="silent"
)

# Status indicators only
llm = LLMNode(
    id="analyzer",
    model="gpt-4",
    event_mode="status_only"
)

# Transient events for custom UI
tool = ToolNode(
    id="scraper",
    tool_fn=scrape_data,
    event_mode="transient_events"
)

# Full streaming (default)
final_agent = AgentNode(
    id="final",
    agent=final_agent,
    event_mode="full"  # Can be omitted, this is default
)
```

### ReactFlow JSON (Flowise UI)

```json
{
  "nodes": [
    {
      "id": "agent_1",
      "type": "agentAgentflow",
      "data": {
        "name": "agentAgentflow",
        "inputs": {
          "agent": "research_agent",
          "eventMode": "silent"
        }
      }
    },
    {
      "id": "agent_2",
      "type": "agentAgentflow",
      "data": {
        "name": "agentAgentflow",
        "inputs": {
          "agent": "writer_agent",
          "eventMode": "full"
        }
      }
    }
  ],
  "edges": [
    {
      "source": "agent_1",
      "target": "agent_2"
    }
  ]
}
```

### mesh-app UI

In the mesh-app visual builder, every node has an "Event Mode" dropdown in its configuration panel:

1. Select a node
2. Open the configuration panel
3. Find "Event Mode" dropdown
4. Choose from:
   - **Full Streaming** - streams to chat
   - **Status Only** - progress indicators
   - **Transient Events** - prefixed events for custom rendering
   - **Silent** - no events

## Event Flow Examples

### Full Mode (Default)

```
Agent emits: {type: "text-delta", content: "Hello"}
Frontend receives: {type: "text-delta", content: "Hello"}
→ Rendered in chat
```

### Status Only Mode

```
Agent emits: {type: "text-delta", content: "Hello"}
Frontend receives: (nothing)

Agent emits: {type: "node_start", node_id: "agent_1"}
Frontend receives: {type: "node_start", node_id: "agent_1"}
→ Progress indicator shows "agent_1 started"
```

### Transient Events Mode

```
Agent emits: {type: "text-delta", content: "Hello"}
Frontend receives: {
  type: "CUSTOM_DATA",
  metadata: {
    data_event_type: "data-agent-node-text-delta",
    data: {content: "Hello", delta: "Hello"},
    transient: true
  }
}
→ Rendered in progress panel (not chat)
```

### Silent Mode

```
Agent emits: {type: "text-delta", content: "Hello"}
Frontend receives: (nothing)

Agent emits: {type: "node_complete", output: "Result"}
Frontend receives: (nothing)
```

## Frontend Integration

### Handling Transient Events

```typescript
// Example: useChat pattern with transient event handling
const { messages, append } = useChat({
  onEvent: (event) => {
    // Check if this is a transient event
    if (event.type === "CUSTOM_DATA" && event.metadata?.transient) {
      const dataEventType = event.metadata.data_event_type;

      // Route to progress panel instead of chat
      if (dataEventType.startsWith("data-agent-node-")) {
        updateProgressPanel("agent", event.metadata.data);
      } else if (dataEventType.startsWith("data-llm-node-")) {
        updateProgressPanel("llm", event.metadata.data);
      }
    } else {
      // Regular events go to chat
      appendToChat(event);
    }
  }
});
```

### React Component Example

```tsx
function MultiAgentChat() {
  const [progress, setProgress] = useState({});
  const { messages, append } = useChat({
    api: "/api/chat",
    onEvent: (event) => {
      if (event.metadata?.transient) {
        // Update progress panel
        setProgress(prev => ({
          ...prev,
          [event.node_id]: event.metadata.data
        }));
      }
    }
  });

  return (
    <div>
      {/* Chat messages (full mode events) */}
      <ChatPanel messages={messages} />

      {/* Progress indicators (transient events) */}
      <ProgressSidebar progress={progress} />
    </div>
  );
}
```

## Best Practices

### 1. Intermediate Nodes Should Be Silent

For multi-agent workflows, intermediate nodes should typically use `silent` or `status_only`:

```python
# ❌ Bad: Intermediate agents streaming to chat
agent_1 = AgentNode(id="research", agent=researcher, event_mode="full")
agent_2 = AgentNode(id="analyze", agent=analyzer, event_mode="full")
agent_3 = AgentNode(id="write", agent=writer, event_mode="full")  # Final

# ✅ Good: Only final agent streams
agent_1 = AgentNode(id="research", agent=researcher, event_mode="silent")
agent_2 = AgentNode(id="analyze", agent=analyzer, event_mode="silent")
agent_3 = AgentNode(id="write", agent=writer, event_mode="full")
```

### 2. Use Status Only for Long Operations

```python
# Tool that processes large dataset
processor = ToolNode(
    id="processor",
    tool_fn=process_dataset,
    event_mode="status_only"  # Show start/complete, skip details
)
```

### 3. Use Transient Events for Rich Progress UIs

```python
# Multi-step LLM analysis
analyzer = LLMNode(
    id="analyzer",
    model="gpt-4",
    event_mode="transient_events"  # All events available, custom rendering
)
```

### 4. Control Flow Nodes

Control flow nodes (Condition, Loop, ForEach) should typically use `status_only`:

```python
condition = ConditionNode(
    id="router",
    conditions=[...],
    event_mode="status_only"  # Show routing decision, skip evaluation details
)

loop = LoopNode(
    id="retry",
    loop_back_to="process",
    event_mode="status_only"  # Show loop iterations
)
```

## Architecture

### Event Emission Flow

```
Node executes
    ↓
Check event_mode
    ↓
├─ silent → Return (no emission)
├─ status_only → Filter (only node_start/complete)
├─ transient_events → Transform (add prefix)
└─ full → Emit normally
    ↓
EventEmitter.emit()
    ↓
Context listeners
    ↓
WebSocket/SSE to frontend
```

### Transform Function

The `transform_event_for_transient_mode()` function in `mesh/core/events.py` handles the transformation:

```python
def transform_event_for_transient_mode(
    event: ExecutionEvent,
    node_type: str  # "agent", "llm", "tool", etc.
) -> ExecutionEvent:
    """Transform event to data-{node_type}-node-* format."""
    # Maps internal types to Vercel AI SDK names
    event_type_map = {
        "token": "text-delta",
        "message_start": "text-start",
        "message_complete": "text-end",
        # ... etc
    }

    mapped_type = event_type_map.get(event_type_str, event_type_str)
    data_event_type = f"data-{node_type}-node-{mapped_type}"

    return ExecutionEvent(
        type=EventType.CUSTOM_DATA,
        metadata={
            "data_event_type": data_event_type,
            "data": {...},  # Original event data
            "transient": True
        }
    )
```

## Troubleshooting

### Events Not Appearing

**Problem:** Node is emitting events but frontend isn't receiving them.

**Solution:**
1. Check node's `event_mode` - if `silent`, no events will be emitted
2. Verify WebSocket/SSE connection is established
3. Check frontend event listener is registered

### Too Many Events

**Problem:** Chat is cluttered with intermediate agent output.

**Solution:**
```python
# Change intermediate nodes to silent or status_only
intermediate_agent.event_mode = "silent"
```

### Custom UI Not Updating

**Problem:** Using `transient_events` but progress panel not updating.

**Solution:**
1. Verify frontend is checking `event.metadata.transient === true`
2. Confirm `data_event_type` prefix matches your filter
3. Check that you're extracting data from `event.metadata.data`

### Status Only Showing Too Little

**Problem:** `status_only` mode not showing enough context.

**Solution:**
```python
# Switch to transient_events for more control
node.event_mode = "transient_events"

# Or switch to full for debugging
node.event_mode = "full"
```

## Reference

### Event Types Affected

All event types can be controlled by `event_mode`:

- `EXECUTION_START`, `EXECUTION_COMPLETE`, `EXECUTION_ERROR`
- `NODE_START`, `NODE_COMPLETE`, `NODE_ERROR`
- `TOKEN`, `MESSAGE_START`, `MESSAGE_COMPLETE`
- `TOOL_CALL_START`, `TOOL_CALL_COMPLETE`
- `STEP_START`, `STEP_COMPLETE`
- `STATE_UPDATE`
- `REASONING_START`, `REASONING_TOKEN`, `REASONING_END`
- `RESPONSE_METADATA`
- `SOURCE`, `FILE`
- `CUSTOM_DATA`

### Mode Comparison

| Mode | Events Emitted | UI Rendering | Use Case |
|------|----------------|--------------|----------|
| `full` | All events | Chat (normal) | Final output nodes |
| `status_only` | Start/complete only | Progress indicators | Long-running operations |
| `transient_events` | All events (prefixed) | Custom (e.g., sidebar) | Rich progress UIs |
| `silent` | No events | None | Intermediate processing |

### Supported Nodes

Event modes are supported on **all** node types:

- ✅ AgentNode
- ✅ LLMNode
- ✅ ToolNode
- ✅ ConditionNode
- ✅ LoopNode
- ✅ ForEachNode
- ✅ StartNode
- ❌ EndNode (deprecated)

## Related Documentation

- [Events System](./events.md) - Core event architecture
- [Multi-Agent Workflows](../guides/multi-agent-workflows.md) - Orchestration patterns
- [Variable Resolution](./variables.md) - Passing data between nodes
