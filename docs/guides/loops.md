---
layout: default
title: Loops
parent: Guides
nav_order: 2
---

# Loop Guide

Understanding how controlled cycles work in Mesh.

## Overview

Mesh supports controlled cycles (loops) in your graphs, allowing nodes to execute repeatedly until a condition is met or a maximum iteration count is reached. This guide explains how the loop mechanism works and when to use it.

## Key Concept

**The loop_condition is evaluated AFTER the node executes, BEFORE deciding whether to queue it again.**

```
┌─────────────┐
│   Execute   │
│    Node     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Evaluate   │  <── Your loop_condition is called HERE
│  Condition  │
└──────┬──────┘
       │
       ├─ True ──▶ Queue node again (continue loop)
       │
       └─ False ─▶ Don't queue node (exit loop)
```

## Creating Loops

To create a loop, mark an edge as a loop edge and provide controls:

```python
graph.add_edge(
    "node_a",
    "node_b",
    is_loop_edge=True,  # Mark as loop edge
    loop_condition=my_condition,  # Optional: exit condition
    max_iterations=100  # Optional: safety limit
)
```

**Requirements:**
- Must set `is_loop_edge=True`
- Must provide at least one: `loop_condition` or `max_iterations`

## Loop Condition Function

The loop_condition function signature:

```python
def loop_condition(state: Dict, output: Dict) -> bool:
    """
    Args:
        state: The shared state dictionary (context.state)
        output: The output from the node that just executed

    Returns:
        True: Continue looping (queue the node again)
        False: Exit the loop (stop queuing the node)
    """
    return some_check(state, output)
```

**Important:**
- Called AFTER node execution
- Receives both shared state and node output
- Return `True` to continue, `False` to exit
- Should be a pure function (no side effects)

## Execution Flow

Here's what happens during loop execution:

### Step 1: Node Executes

```python
def increment(input: dict) -> dict:
    value = input.get("value", 0)
    new_value = value + 1
    print(f"Executing: {value} -> {new_value}")
    return {"value": new_value}
```

### Step 2: Condition Evaluated

```python
def should_continue(state: dict, output: dict) -> bool:
    value = output.get("value", 0)
    should_continue = value < 5
    print(f"Check: value={value} < 5? {should_continue}")
    return should_continue
```

### Step 3: Decision Made

- If condition returns `True`: Node is queued for another iteration
- If condition returns `False`: Loop exits, node is not queued

### Complete Timeline

```
[Iteration 1]
  → Node executes: value 0 → 1
  → Condition evaluated: value=1 < 5? True
  → Node queued again

[Iteration 2]
  → Node executes: value 1 → 2
  → Condition evaluated: value=2 < 5? True
  → Node queued again

[Iteration 3]
  → Node executes: value 4 → 5
  → Condition evaluated: value=5 < 5? False
  → Loop exits
```

## Common Patterns

### Pattern 1: Loop Until Flag is Set

```python
def process_task(input: dict) -> dict:
    # Do work
    result = do_work(input)
    return {
        "result": result,
        "done": is_finished(result)  # Set flag when done
    }

def should_continue(state, output):
    return not output.get("done", False)

graph.add_node("process", process_task, node_type="tool")
graph.add_edge("START", "process")
graph.add_edge(
    "process",
    "process",
    is_loop_edge=True,
    loop_condition=should_continue,
    max_iterations=50  # Safety limit
)
```

### Pattern 2: Loop Until Counter Reaches Value

```python
def increment_counter(input: dict) -> dict:
    count = input.get("count", 0)
    return {"count": count + 1}

def should_continue(state, output):
    return output.get("count", 0) < 100

graph.add_edge(
    "counter",
    "counter",
    is_loop_edge=True,
    loop_condition=should_continue
)
```

### Pattern 3: Loop Until Quality Threshold Met

```python
def refine_output(input: dict) -> dict:
    # Improve quality
    refined = improve(input)
    quality = calculate_quality(refined)
    return {
        "output": refined,
        "quality_score": quality
    }

def should_continue(state, output):
    quality = output.get("quality_score", 0)
    return quality < 0.95  # Continue until 95% quality

graph.add_edge(
    "refine",
    "refine",
    is_loop_edge=True,
    loop_condition=should_continue,
    max_iterations=10  # Don't refine more than 10 times
)
```

### Pattern 4: Loop Based on Shared State

```python
def process_batch(input: dict, state: dict) -> dict:
    # Process one batch
    results = process(input)

    # Update shared state
    state["total_processed"] = state.get("total_processed", 0) + len(results)

    return {"results": results}

def should_continue(state, output):
    # Check shared state, not just output
    total = state.get("total_processed", 0)
    return total < 1000

graph.add_edge(
    "batch",
    "batch",
    is_loop_edge=True,
    loop_condition=should_continue
)
```

### Pattern 5: Multi-Node Loop

Loop between multiple nodes:

```python
def check(input: dict) -> dict:
    value = input.get("value", 0)
    return {
        "value": value,
        "divisible_by_5": (value % 5) == 0
    }

def increment(input: dict) -> dict:
    value = input.get("value", 0)
    return {"value": value + 1}

graph.add_node("check", check, node_type="tool")
graph.add_node("increment", increment, node_type="tool")

graph.add_edge("START", "check")
graph.add_edge("check", "increment")  # check -> increment

# Loop back: increment -> check
graph.add_edge(
    "increment",
    "check",
    is_loop_edge=True,
    loop_condition=lambda state, output: not output.get("divisible_by_5", False),
    max_iterations=20
)
```

## Max Iterations Only

You can use `max_iterations` without a condition:

```python
# Self-loop that runs exactly 10 times
graph.add_edge(
    "process",
    "process",
    is_loop_edge=True,
    max_iterations=10
)
```

This is useful when you want a fixed number of iterations.

## Loop Exit Conditions

The loop exits when **ANY** of these conditions are met:

1. **loop_condition returns False**
   ```python
   loop_condition=lambda state, output: output.get("count", 0) < 10
   # Exits when count >= 10
   ```

2. **max_iterations is reached**
   ```python
   max_iterations=100
   # Exits after 100 loop iterations
   ```

3. **loop_condition raises an exception**
   ```python
   def my_condition(state, output):
       return output["missing_key"]  # KeyError -> loop exits safely
   ```

## Safety Features

### Automatic Loop Exit

If your condition raises an exception, the loop exits safely:

```python
def risky_condition(state, output):
    # If this raises KeyError, loop exits instead of crashing
    return output["maybe_missing_key"] < 10

# Loop will exit gracefully on exception
```

### Max Iterations Backstop

Always provide `max_iterations` as a safety net:

```python
# Even if condition logic is wrong, max_iterations prevents infinite loops
graph.add_edge(
    "node", "node",
    is_loop_edge=True,
    loop_condition=lambda s, o: True,  # Bug: always true!
    max_iterations=50  # Saved by max_iterations
)
```

### Global Iteration Limit

The executor has a global `max_iterations` limit (default 1000):

```python
executor = Executor(compiled, backend, max_iterations=500)
```

## Complete Example

```python
import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend

def increment(input: dict) -> dict:
    """Increment the value by 1."""
    value = input.get("value", 0)
    new_value = value + 1
    print(f"Execute: {value} -> {new_value}")
    return {"value": new_value}

def should_continue(state: dict, output: dict) -> bool:
    """Continue while value < 5."""
    value = output.get("value", 0)
    result = value < 5
    print(f"Condition: value={value} < 5? {result}")
    return result

async def main():
    # Build graph with loop
    graph = StateGraph()
    graph.add_node("increment", increment, node_type="tool")
    graph.add_edge("START", "increment")
    graph.add_edge(
        "increment",
        "increment",  # Loop back to itself
        is_loop_edge=True,
        loop_condition=should_continue,
        max_iterations=10  # Safety limit
    )
    graph.set_entry_point("increment")

    # Execute
    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="loop-demo",
        session_id="1",
        chat_history=[],
        variables={},
        state={}
    )

    async for event in executor.execute({"value": 0}, context):
        if event.type == "execution_complete":
            print(f"Final: {event.output}")
            print(f"Loop iterations: {context.loop_iterations}")

asyncio.run(main())
```

Output:
```
Execute: 0 -> 1
Condition: value=1 < 5? True
Execute: 1 -> 2
Condition: value=2 < 5? True
Execute: 2 -> 3
Condition: value=3 < 5? True
Execute: 3 -> 4
Condition: value=4 < 5? True
Execute: 4 -> 5
Condition: value=5 < 5? False
Final: {'value': 5}
Loop iterations: {'increment->increment': 4}
```

## Tracking Loop Iterations

Check loop iteration counts from the execution context:

```python
async for event in executor.execute(input, context):
    if event.type == "execution_complete":
        # Dictionary of edge_key -> iteration_count
        print(context.loop_iterations)
        # Output: {'node_a->node_b': 10}
```

The edge key format is `"source_node->target_node"`.

## Common Mistakes

### ❌ Modifying State in Condition

```python
# BAD: Don't modify state in the condition
def bad_condition(state, output):
    state["counter"] += 1  # Don't do this!
    return state["counter"] < 10
```

### ✅ Modify State in Node

```python
# GOOD: Modify state in the node
def my_node(input, state):
    state["counter"] = state.get("counter", 0) + 1
    return {"value": state["counter"]}

def my_condition(state, output):
    return state.get("counter", 0) < 10  # Just read
```

### ❌ Forgetting Loop Controls

```python
# BAD: Loop edge without controls
graph.add_edge("node", "node", is_loop_edge=True)
# Error: Must have loop_condition or max_iterations!
```

### ✅ Always Provide Controls

```python
# GOOD: At least one control mechanism
graph.add_edge("node", "node", is_loop_edge=True, max_iterations=10)
```

## Debugging Tips

### Add Logging to Your Condition

```python
def should_continue(state, output):
    result = output.get("count", 0) < 10
    print(f"[CONDITION] count={output.get('count')}, continue={result}")
    return result
```

### Check Iteration Counts

```python
async for event in executor.execute(input, context):
    if event.type == "node_complete":
        print(f"Loop iterations so far: {context.loop_iterations}")
```

### Use Detailed Examples

See the examples directory for detailed walkthroughs:

```bash
# Detailed explanation with logging
python examples/loop_condition_explained.py

# Visual timing diagram
python examples/loop_timing_diagram.py

# Real-world patterns
python examples/cyclic_graph_example.py
python examples/cyclic_graph_max_iterations.py
```

## See Also

- [Graphs Concept](../concepts/graphs) - Graph structure with loop patterns
- [Quick Start](../quick-start) - Pattern 5: Controlled Cycles
- [Troubleshooting](../troubleshooting) - Cycle detection errors
- Examples:
  - `examples/loop_condition_explained.py` - Detailed walkthrough
  - `examples/loop_timing_diagram.py` - Timing visualization
  - `examples/cyclic_graph_example.py` - Multi-node loop
  - `examples/cyclic_graph_max_iterations.py` - Self-loop
