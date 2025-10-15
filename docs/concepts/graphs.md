---
layout: default
title: Graphs
parent: Concepts
nav_order: 1
---

# Graphs

Graphs are the foundation of Mesh. They define the structure and flow of your multi-agent workflows.

## What is a Graph?

A **graph** in Mesh is a **directed acyclic graph (DAG)** composed of:
- **Nodes**: Units of execution (agents, LLMs, tools, etc.)
- **Edges**: Connections between nodes defining execution flow
- **Entry Point**: The starting node for execution

## Graph Structure

```
START (implicit)
  ↓
[entry_node]
  ↓
[node_a]
  ↓
[node_b]
  ↓
END (implicit)
```

### Key Properties

1. **Directed**: Edges have a direction (source → target)
2. **Acyclic**: No cycles (can't loop back to earlier nodes)
3. **Connected**: All nodes must be reachable from START

## Building Graphs

### Programmatic API (StateGraph)

The recommended way to build graphs is using the `StateGraph` builder:

```python
from mesh import StateGraph

graph = StateGraph()

# Add nodes
graph.add_node("agent", my_agent, node_type="agent")
graph.add_node("tool", my_function, node_type="tool")

# Add edges
graph.add_edge("START", "agent")
graph.add_edge("agent", "tool")

# Set entry point
graph.set_entry_point("agent")

# Compile
compiled = graph.compile()
```

### React Flow JSON (Declarative)

Parse Flowise-compatible JSON:

```python
from mesh import ReactFlowParser, NodeRegistry

registry = NodeRegistry()
parser = ReactFlowParser(registry)

flow_json = {
    "nodes": [
        {"id": "agent_1", "type": "agentAgentflow", "data": {...}},
        {"id": "llm_1", "type": "llmAgentflow", "data": {...}}
    ],
    "edges": [
        {"source": "agent_1", "target": "llm_1"}
    ]
}

graph = parser.parse(flow_json)
```

## Graph Lifecycle

```
Build → Validate → Compile → Execute
```

### 1. Build

Add nodes and edges:

```python
graph = StateGraph()
graph.add_node("step1", None, node_type="llm")
graph.add_node("step2", None, node_type="llm")
graph.add_edge("START", "step1")
graph.add_edge("step1", "step2")
```

### 2. Validate

Check for common issues:

```python
graph.set_entry_point("step1")
compiled = graph.compile()  # Validates automatically
```

Validation checks:
- Entry point is set
- No cycles exist
- All nodes are connected
- No orphaned nodes

### 3. Compile

Creates an executable `ExecutionGraph`:

```python
compiled = graph.compile()
# Returns: ExecutionGraph with dependency info
```

### 4. Execute

Run the graph:

```python
from mesh import Executor, ExecutionContext, MemoryBackend

executor = Executor(compiled, MemoryBackend())
context = ExecutionContext(
    graph_id="my-graph",
    session_id="session-1",
    chat_history=[],
    variables={},
    state={}
)

async for event in executor.execute("input", context):
    print(event)
```

## Graph Patterns

### Sequential Flow

```python
# A → B → C
graph.add_node("A", None, node_type="llm")
graph.add_node("B", None, node_type="llm")
graph.add_node("C", None, node_type="llm")
graph.add_edge("START", "A")
graph.add_edge("A", "B")
graph.add_edge("B", "C")
```

### Parallel Branches

```python
# A → [B, C] (both execute after A)
graph.add_node("A", None, node_type="llm")
graph.add_node("B", None, node_type="llm")
graph.add_node("C", None, node_type="llm")
graph.add_edge("START", "A")
graph.add_edge("A", "B")
graph.add_edge("A", "C")
```

### Conditional Branching

```python
from mesh.nodes import Condition

# A → [condition] → [B or C]
graph.add_node("A", None, node_type="llm")
graph.add_node("condition", [
    Condition("success", lambda x: "success" in str(x), "B"),
    Condition("failure", lambda x: "failure" in str(x), "C")
], node_type="condition")
graph.add_node("B", None, node_type="llm")
graph.add_node("C", None, node_type="llm")
graph.add_edge("START", "A")
graph.add_edge("A", "condition")
graph.add_edge("condition", "B")
graph.add_edge("condition", "C")
```

### Multi-Input Nodes

```python
# [A, B] → C (C waits for both A and B)
graph.add_node("A", None, node_type="llm")
graph.add_node("B", None, node_type="llm")
graph.add_node("C", None, node_type="llm")
graph.add_edge("START", "A")
graph.add_edge("START", "B")
graph.add_edge("A", "C")
graph.add_edge("B", "C")  # C waits for both
```

## Execution Model

Mesh uses a **queue-based execution model**:

1. **Initialize queue** with entry point
2. **Dequeue node** and execute
3. **Emit events** (streaming)
4. **Queue children** based on dependencies
5. **Repeat** until queue is empty

### Dependency Resolution

When a node has multiple parents:

```python
# A → C
# B → C (C waits for both A and B)
```

The executor:
1. Tracks received inputs from parents
2. Waits until all parents complete
3. Combines inputs
4. Executes node

## Graph State

State flows through the graph via:

### 1. Node Outputs

Each node produces output that becomes input for children:

```python
# Node A output: {"content": "Hello"}
# Node B receives: {"content": "Hello"}
```

### 2. Shared State

Persistent state across all nodes:

```python
context = ExecutionContext(
    graph_id="my-graph",
    session_id="session-1",
    chat_history=[],
    variables={"user_id": "123"},  # Shared variables
    state={"step_count": 0}  # Shared state
)
```

### 3. Chat History

Accumulated conversation history:

```python
context.chat_history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
```

## Best Practices

### 1. Use Descriptive Node IDs

```python
# ❌ Bad
graph.add_node("n1", None, node_type="llm")
graph.add_node("n2", None, node_type="llm")

# ✅ Good
graph.add_node("analyzer", None, node_type="llm")
graph.add_node("summarizer", None, node_type="llm")
```

### 2. Keep Graphs Simple

```python
# ✅ Good: 3-7 nodes
START → analyze → process → respond → END

# ⚠️ Consider splitting: 20+ nodes
# Too complex, hard to maintain
```

### 3. Validate Early

```python
graph = StateGraph()
graph.add_node("step1", None, node_type="llm")
graph.add_node("step2", None, node_type="llm")
graph.add_edge("START", "step1")
graph.add_edge("step1", "step2")
graph.set_entry_point("step1")

try:
    compiled = graph.compile()  # Validates
except ValueError as e:
    print(f"Graph invalid: {e}")
```

### 4. Use Variables for Flexibility

```python
# Reference previous nodes
graph.add_node("step2", None, node_type="llm",
               system_prompt="Based on {{step1}}, ...")
```

## Common Errors

### "No entry point set"

**Problem:** Forgot to call `set_entry_point()`

**Solution:**
```python
graph.set_entry_point("first_node")
compiled = graph.compile()
```

### "Cycle detected in graph"

**Problem:** Edge creates a cycle

**Solution:** Remove the edge that causes the cycle. Graphs must be acyclic.

### "Orphaned nodes detected"

**Problem:** Node not connected to START

**Solution:** Add edge from START or parent node:
```python
graph.add_edge("START", "orphaned_node")
```

### "Node not found"

**Problem:** Referencing non-existent node in edge

**Solution:** Ensure node exists before adding edge:
```python
graph.add_node("node_a", None, node_type="llm")  # Create first
graph.add_edge("START", "node_a")  # Then reference
```

## Advanced Topics

### Sub-Graphs (Future)

Coming in v2:

```python
# Compose graphs
sub_graph = StateGraph()
# ... build sub_graph

main_graph = StateGraph()
main_graph.add_subgraph("processing", sub_graph)
```

### Dynamic Graphs (Future)

Coming in v2:

```python
# Modify graph during execution
executor.add_node_dynamically("new_node", ...)
```

## See Also

- [Nodes](nodes) - Learn about node types
- [Execution](execution) - Understand execution flow
- [State Management](../guides/state-management) - Managing state
- [Variables](../guides/variables) - Variable resolution
