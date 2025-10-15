---
layout: default
title: API Reference
nav_order: 8
---

# API Reference

Complete API documentation for Mesh.

## Core Classes

### StateGraph

Build graphs programmatically with LangGraph-style API.

```python
from mesh import StateGraph

graph = StateGraph()
graph.add_node(node_id, node_or_config, node_type, **kwargs)
graph.add_edge(source, target)
graph.set_entry_point(node_id)
compiled = graph.compile()
```

**Methods:**

- `add_node(id, node, node_type, **kwargs)` - Add node to graph
- `add_edge(source, target)` - Add edge between nodes
- `set_entry_point(node_id)` - Set starting node
- `compile() -> ExecutionGraph` - Validate and compile graph

### Executor

Execute compiled graphs with streaming.

```python
from mesh import Executor

executor = Executor(graph, backend)
async for event in executor.execute(input, context):
    # Handle events
```

**Methods:**

- `execute(input, context) -> AsyncIterator[ExecutionEvent]` - Execute graph with streaming

### ExecutionContext

Runtime context for graph execution.

```python
from mesh import ExecutionContext

context = ExecutionContext(
    graph_id="my-graph",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)
```

**Fields:**

- `graph_id: str` - Graph identifier
- `session_id: str` - Session identifier
- `chat_history: List[Dict]` - Conversation history
- `variables: Dict[str, Any]` - Global variables
- `state: Dict[str, Any]` - Persistent state

## Backends

### MemoryBackend

In-memory state storage (development).

```python
from mesh.backends import MemoryBackend

backend = MemoryBackend()
```

### SQLiteBackend

SQLite state persistence (production).

```python
from mesh.backends import SQLiteBackend

backend = SQLiteBackend("mesh_state.db")
```

## Node Types

### AgentNode

Wraps Vel or OpenAI agents.

```python
graph.add_node("agent", agent_instance, node_type="agent",
               system_prompt="...",
               use_native_events=False)
```

### LLMNode

Direct LLM calls.

```python
graph.add_node("llm", None, node_type="llm",
               model="gpt-4",
               system_prompt="...",
               temperature=0.7)
```

### ToolNode

Execute Python functions.

```python
graph.add_node("tool", function, node_type="tool",
               config={"bindings": {...}})
```

### ConditionNode

Conditional branching.

```python
from mesh.nodes import Condition

conditions = [
    Condition("name", predicate, "target_node")
]
graph.add_node("condition", conditions, node_type="condition",
               default_target="fallback")
```

### LoopNode

Array iteration.

```python
graph.add_node("loop", None, node_type="loop",
               array_path="$.items",
               max_iterations=100)
```

## Events

### ExecutionEvent

```python
@dataclass
class ExecutionEvent:
    type: EventType
    node_id: str
    content: Optional[str]
    output: Optional[Dict]
    metadata: Dict[str, Any]
    timestamp: float
    error: Optional[str]
```

### EventType

```python
class EventType(Enum):
    EXECUTION_START = "execution_start"
    EXECUTION_COMPLETE = "execution_complete"
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    NODE_ERROR = "node_error"
    TOKEN = "token"
    MESSAGE_START = "message_start"
    MESSAGE_COMPLETE = "message_complete"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
```

## Utilities

### load_env()

Load environment variables from .env file.

```python
from mesh.utils import load_env

load_env()
```

## Streaming

### SSEAdapter

Server-Sent Events adapter for FastAPI.

```python
from mesh.streaming import SSEAdapter

adapter = SSEAdapter()
return adapter.to_streaming_response(executor.execute(input, context))
```

## See Also

- [Getting Started](getting-started) - Installation guide
- [Quick Start](quick-start) - Usage examples
- [Concepts](concepts) - Core concepts
- [GitHub Repository](https://github.com/rscheiwe/mesh)
