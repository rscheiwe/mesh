# 03 - Advanced Patterns

This folder contains examples of advanced graph patterns including loops, cycles, and conditional execution.

## Examples

### `condition_node_routing.py`
**Difficulty:** ⭐⭐ Intermediate

Demonstrates **deterministic** conditional branching using ConditionNode - pure logic-based routing without AI.

**Examples include:**
1. Simple true/false routing (age check: adult vs minor)
2. Multiple conditions with all SimpleCondition helpers
3. Custom predicate functions for complex logic

**What you'll learn:**
- True/false branching based on conditions
- All SimpleCondition operations: equals, not_equal, contains, greater_than, less_than, is_empty
- Creating custom predicates for complex logic
- Default routing when no conditions match
- Deterministic mode (condition_routing="deterministic")

**Run:**
```bash
python 03_advanced_patterns/condition_node_routing.py
```

### `condition_node_ai_routing.py`
**Difficulty:** ⭐⭐⭐ Advanced

Demonstrates **AI-driven** conditional branching using ConditionNode - LLM-based intent classification and routing. This matches Flowise's Condition Agent Node behavior.

**Examples include:**
1. Intent classification for customer service routing
2. Hybrid approach: deterministic checks + AI fallback

**What you'll learn:**
- AI mode for ConditionNode (condition_routing="ai")
- Using LLM for intent classification and routing
- Defining scenarios with natural language descriptions
- Hybrid routing: fast deterministic rules + AI for complex cases
- Fallback handling when LLM fails

**Requirements:**
- OPENAI_API_KEY environment variable

**Run:**
```bash
export OPENAI_API_KEY='your-key-here'
python 03_advanced_patterns/condition_node_ai_routing.py
```

### `cyclic_graph_example.py`
**Difficulty:** ⭐⭐⭐ Advanced

Demonstrates a controlled cycle (loop) in a graph where a node can loop back to itself or a previous node until a condition is met.

**Example:** Loop until a number is divisible by 5
- Check if divisible → If no, increment → Check again (loop)
- Check if divisible → If yes, exit

**What you'll learn:**
- Creating cyclic graphs with `is_loop_edge=True`
- Loop conditions: `loop_condition` lambda functions
- Safety limits: `max_iterations`
- Loop iteration tracking in state

**Run:**
```bash
python 03_advanced_patterns/cyclic_graph_example.py
```

### `cyclic_graph_max_iterations.py`
**Difficulty:** ⭐⭐⭐ Advanced

Demonstrates loop safety controls using `max_iterations` to prevent infinite loops.

**What you'll learn:**
- Using `max_iterations` as a safety net
- Loop termination strategies
- Combining conditions with iteration limits

**Run:**
```bash
python 03_advanced_patterns/cyclic_graph_max_iterations.py
```

### `loop_condition_explained.py`
**Difficulty:** ⭐⭐⭐ Advanced

In-depth explanation of loop condition functions and their signatures.

**What you'll learn:**
- Loop condition function signature: `(state, output) -> bool`
- Accessing state vs. output in conditions
- When to use state-based vs. output-based conditions
- Debugging loop conditions

**Run:**
```bash
python 03_advanced_patterns/loop_condition_explained.py
```

### `loop_timing_diagram.py`
**Difficulty:** ⭐⭐⭐ Advanced

Visualizes the execution timing of loops to understand when nodes execute and how loops iterate.

**What you'll learn:**
- Loop execution timeline
- Node execution order in cycles
- Loop iteration counting
- Performance implications

**Run:**
```bash
python 03_advanced_patterns/loop_timing_diagram.py
```

## Key Concepts

### Controlled Cycles

Mesh supports cyclic graphs (loops) with proper controls to prevent infinite loops:

```python
# Mark an edge as a loop edge
graph.add_edge(
    "increment",
    "check",
    is_loop_edge=True,           # Required for cycles
    loop_condition=condition_fn,  # Function returning bool
    max_iterations=100            # Safety limit
)
```

### Loop Conditions

Loop conditions determine whether to continue looping:

```python
def should_continue_loop(state: Dict, output: Dict) -> bool:
    """Return True to continue loop, False to exit."""
    return output.get("divisible", False) == False
```

**Function signature:**
- `state`: Full execution state (all nodes)
- `output`: Output from the source node
- Returns `bool`: True to continue loop, False to exit

### Safety Controls

**Required:** Every loop edge must have at least one of:
- `loop_condition`: Function to determine continuation
- `max_iterations`: Maximum number of times the loop can execute

**Best practice:** Use both for safety:
```python
loop_condition=my_condition,  # Logical exit condition
max_iterations=100            # Safety net
```

### Loop Iteration Tracking

Mesh automatically tracks loop iterations:
```python
context.loop_iterations[node_id]  # Current iteration count
```

## When to Use Loops

✅ **Good use cases:**
- Retry logic with conditions
- Iterative refinement (improve until quality threshold met)
- Search/exploration (try until found)
- Polling (check until ready)

❌ **Not recommended:**
- Simple data iteration (use `LoopNode` instead)
- Fixed number of steps (use explicit nodes)
- State machines (use `ConditionNode` instead)

## Important Notes

### Uncontrolled Cycles are Errors

```python
# ❌ ERROR: Uncontrolled cycle
graph.add_edge("A", "B")
graph.add_edge("B", "A")  # No is_loop_edge or controls

# ✅ CORRECT: Controlled cycle
graph.add_edge("A", "B")
graph.add_edge(
    "B",
    "A",
    is_loop_edge=True,
    max_iterations=10
)
```

### Exit Edges

Provide an exit path from the loop:
```python
# Loop edge (continue cycling)
graph.add_edge("check", "increment", is_loop_edge=True, ...)

# Exit edge (break out of loop)
graph.add_edge("check", "END", loop_condition=lambda s, o: o["done"])
```

## Next Steps

- **04_event_translation:** Learn about event translation and step tracking
- **05_integrations:** See how to integrate Mesh with web frameworks
