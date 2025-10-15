---
layout: default
title: Execution
parent: Core Concepts
nav_order: 3
---

# Execution

Understanding how Mesh executes graphs.

## Execution Model

Mesh uses a **queue-based execution model** with dependency tracking:

```
Queue → Dequeue Node → Execute → Emit Events → Queue Children → Repeat
```

### Execution Flow

1. **Initialize** queue with entry point
2. **Dequeue** next node
3. **Execute** node logic
4. **Emit** streaming events
5. **Update** shared state
6. **Queue** child nodes (if dependencies met)
7. **Repeat** until queue empty

## Executor

The `Executor` class orchestrates graph execution:

```python
from mesh import Executor, ExecutionContext, MemoryBackend

executor = Executor(graph, backend)
```

**Parameters:**
- `graph`: Compiled `ExecutionGraph`
- `backend`: State persistence backend

### Executing a Graph

```python
context = ExecutionContext(
    graph_id="my-graph",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("user input", context):
    if event.type == "token":
        print(event.content, end="", flush=True)
```

## Execution Context

The `ExecutionContext` carries runtime information:

```python
@dataclass
class ExecutionContext:
    graph_id: str              # Graph identifier
    session_id: str            # Session identifier
    chat_history: List[Dict]   # Conversation history
    variables: Dict[str, Any]  # Global variables
    state: Dict[str, Any]      # Persistent state
    trace_id: str              # Execution trace ID
    iteration_context: Dict    # Loop iteration data
```

### Creating Context

```python
context = ExecutionContext(
    graph_id="workflow-1",
    session_id="user-123",
    chat_history=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"}
    ],
    variables={"user_name": "Alice"},
    state={"step": 1}
)
```

### Updating Context

Context is mutable during execution:

```python
# Nodes can update state
node_result = NodeResult(
    output={...},
    state={"new_key": "value"}  # Updates context.state
)

# Chat history grows
context.chat_history.append({"role": "assistant", "content": "..."})
```

## Dependency Resolution

When nodes have multiple parents, Mesh waits for all inputs:

```
    A
   / \
  /   \
 B     C
  \   /
   \ /
    D  (waits for both B and C)
```

### How It Works

1. Node D starts with empty input tracker
2. Node B completes → D receives input from B
3. Node C completes → D receives input from C
4. All dependencies met → D executes with combined inputs

### Input Combination

```python
# B output: {"from_b": "data"}
# C output: {"from_c": "data"}

# D receives:
{
    "B": {"from_b": "data"},
    "C": {"from_c": "data"}
}
```

## Execution Events

Mesh emits events during execution for streaming:

```python
async for event in executor.execute(input, context):
    if event.type == "execution_start":
        # Graph execution begins
        pass

    elif event.type == "node_start":
        # Node starting
        print(f"[{event.node_id} starting]")

    elif event.type == "token":
        # Streaming token
        print(event.content, end="", flush=True)

    elif event.type == "node_complete":
        # Node finished
        print(f"[{event.node_id} completed]")

    elif event.type == "execution_complete":
        # Graph finished
        print(f"Final: {event.output}")

    elif event.type == "node_error":
        # Error occurred
        print(f"Error in {event.node_id}: {event.error}")
```

See [Events](events) for complete event reference.

## State Management

State persists across nodes and executions:

### In-Memory State

```python
from mesh.backends import MemoryBackend

backend = MemoryBackend()
executor = Executor(graph, backend)

# State lives in memory only
```

### Persistent State (SQLite)

```python
from mesh.backends import SQLiteBackend

backend = SQLiteBackend("mesh_state.db")
executor = Executor(graph, backend)

# State persists to database
```

### Accessing State

```python
# In node execution
class MyNode(BaseNode):
    async def _execute_impl(self, input, context):
        # Read state
        count = context.state.get("count", 0)
        
        # Update state
        return NodeResult(
            output={...},
            state={"count": count + 1}
        )
```

## Error Handling

### Node-Level Retries

```python
graph.add_node("flaky", my_function, node_type="tool",
               config={
                   "retry": {
                       "max_retries": 3,
                       "delay": 1.0
                   }
               })
```

### Execution-Level Errors

```python
try:
    async for event in executor.execute(input, context):
        if event.type == "node_error":
            print(f"Error: {event.error}")
            # Handle error
except Exception as e:
    print(f"Execution failed: {e}")
```

### Fail-Fast vs Continue

By default, Mesh fails fast on errors. To continue:

```python
# Coming in future version
executor = Executor(graph, backend, fail_fast=False)
```

## Loop Execution

Loop nodes create sub-executions for each array item:

```python
graph.add_node("loop", None, node_type="loop", array_path="$.items")
graph.add_node("processor", None, node_type="llm")
graph.add_edge("START", "loop")
graph.add_edge("loop", "processor")
```

**Execution Flow:**

1. Loop node extracts array: `[item1, item2, item3]`
2. For each item:
   - Set `context.iteration_context = {index, value, is_first, is_last}`
   - Queue downstream nodes
   - Execute with iteration context
3. Collect all results

### Iteration Context

```python
{
    "index": 0,              # 0-based index
    "value": item,           # Current item
    "is_first": True,        # First iteration?
    "is_last": False         # Last iteration?
}
```

Access in templates:

```python
system_prompt="Process item {{$iteration_index}}: {{$iteration}}"
```

## Conditional Execution

Condition nodes determine which branches execute:

```python
graph.add_node("condition", [
    Condition("positive", check_positive, "handler_a"),
    Condition("negative", check_negative, "handler_b"),
], node_type="condition")
```

**Execution Flow:**

1. Condition node evaluates predicates
2. Marks unfulfilled branches for skipping
3. Executor skips queuing ignored nodes
4. Only fulfilled branches execute

## Performance Considerations

### Parallel Execution

Nodes with no dependencies execute in parallel:

```
START
  |
  A
 / \
B   C  (execute in parallel)
 \ /
  D
```

### Queue Management

Executor maintains efficient queue:
- O(1) enqueue/dequeue
- Tracks dependencies
- Avoids redundant work

### Streaming Overhead

Streaming adds minimal overhead:
- Events yielded via AsyncIterator
- No buffering required
- Real-time feedback

### State Persistence

Persistence costs:
- **MemoryBackend**: ~0ms overhead
- **SQLiteBackend**: ~1-5ms per save

## Best Practices

### 1. Keep Context Minimal

```python
# ✅ Good: Only what's needed
context = ExecutionContext(
    graph_id="test",
    session_id="session-1",
    chat_history=[],
    variables={"user_id": "123"},
    state={}
)

# ❌ Bad: Huge state object
context = ExecutionContext(
    ...,
    state={"huge_data": [...]}  # Store in DB instead
)
```

### 2. Use Appropriate Backend

```python
# Development
backend = MemoryBackend()

# Production
backend = SQLiteBackend("prod.db")
```

### 3. Handle Errors Gracefully

```python
async for event in executor.execute(input, context):
    if event.type == "node_error":
        # Log and handle
        logger.error(f"Node {event.node_id} failed: {event.error}")
```

### 4. Monitor Performance

```python
import time

start = time.time()
async for event in executor.execute(input, context):
    if event.type == "execution_complete":
        elapsed = time.time() - start
        print(f"Execution took {elapsed:.2f}s")
```

## Advanced Topics

### Custom Executors

Extend `Executor` for custom behavior:

```python
class CustomExecutor(Executor):
    async def execute(self, input, context):
        # Custom pre-processing
        
        async for event in super().execute(input, context):
            # Custom event handling
            yield event
```

### Execution Hooks (Future)

Coming soon:

```python
executor.on("node_start", lambda event: ...)
executor.on("node_complete", lambda event: ...)
```

## See Also

- [Graphs](graphs) - Graph structure
- [Nodes](nodes) - Node types
- [Events](events) - Event system
- [State Management Guide](../guides/state-management) - State patterns
