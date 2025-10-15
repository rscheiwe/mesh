---
layout: default
title: Nodes
parent: Core Concepts
nav_order: 2
---

# Nodes

Nodes are the building blocks of Mesh workflows. Each node performs a specific task and can emit streaming events.

## Node Types

Mesh provides 7 core node types:

| Type | Purpose | Key Features |
|------|---------|--------------|
| **StartNode** | Graph entry point | Implicit, auto-created |
| **EndNode** | Graph exit point | Implicit, optional |
| **AgentNode** | Vel/OpenAI agents | Streaming, auto-detection |
| **LLMNode** | Direct LLM calls | Streaming, simple |
| **ToolNode** | Python functions | Sync/async support |
| **ConditionNode** | Branching logic | Multiple conditions |
| **LoopNode** | Array iteration | JSONPath selection |

## 1. StartNode

The entry point to your graph. Always implicit (auto-created).

**Usage:**

```python
graph = StateGraph()
graph.add_node("first", None, node_type="llm")
graph.add_edge("START", "first")  # "START" is automatic
graph.set_entry_point("first")
```

**Key Points:**
- Don't create manually
- Always reference as `"START"` in edges
- Sets up initial execution context

## 2. EndNode

The exit point of your graph. Optional and implicit.

**Usage:**

```python
graph.add_node("last", None, node_type="llm")
graph.add_edge("last", "END")  # Optional
```

**Key Points:**
- Automatically added if nodes have no children
- Triggers `execution_complete` event
- Returns final output

## 3. AgentNode

Wraps Vel or OpenAI Agents SDK agents with automatic detection and streaming.

**With Vel:**

```python
from vel import Agent as VelAgent

vel_agent = VelAgent(
    id="assistant",
    model={"provider": "openai", "name": "gpt-4"},
)

graph.add_node("agent", vel_agent, node_type="agent")
```

**With OpenAI Agents SDK:**

```python
from agents import Agent

openai_agent = Agent(
    name="Assistant",
    instructions="You are helpful"
)

# Vel translation by default
graph.add_node("agent", openai_agent, node_type="agent")

# Or use native events
graph.add_node("agent", openai_agent, node_type="agent",
               use_native_events=True)
```

**Key Features:**
- Auto-detects agent type (Vel vs OpenAI)
- Token-by-token streaming
- Event translation (Vel format by default)
- Chat history management
- System prompt override:

```python
graph.add_node("agent", my_agent, node_type="agent",
               system_prompt="Custom prompt: {{$question}}")
```

## 4. LLMNode

Direct LLM calls without agent framework. Simpler but less powerful than AgentNode.

**Usage:**

```python
graph.add_node("llm", None, node_type="llm",
               model="gpt-4",
               system_prompt="You are a helpful assistant")
```

**Parameters:**
- `model`: OpenAI model name (required)
- `system_prompt`: System message (optional, supports variables)
- `temperature`: Creativity (0-2, default: 1.0)
- `max_tokens`: Output limit (default: None)

**Example with Variables:**

```python
graph.add_node("llm", None, node_type="llm",
               model="gpt-4",
               system_prompt="Analyze: {{$question}}. Context: {{previous_node}}")
```

**When to Use:**
- Quick LLM calls without tools
- Simple text generation
- No need for conversation history
- No tool calling required

## 5. ToolNode

Execute arbitrary Python functions as tools.

**Basic Usage:**

```python
def my_tool(input: dict) -> dict:
    query = input.get("query", "")
    # Your logic here
    return {"result": f"Processed: {query}"}

graph.add_node("tool", my_tool, node_type="tool")
```

**Async Tools:**

```python
async def fetch_data(input: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(input["url"])
        return {"data": response.json()}

graph.add_node("fetcher", fetch_data, node_type="tool")
```

**With Configuration:**

```python
def multiply(input: dict, multiplier: int = 2) -> dict:
    value = input.get("value", 0)
    return {"result": value * multiplier}

graph.add_node("tool", multiply, node_type="tool",
               config={"bindings": {"multiplier": 3}})
```

**Key Features:**
- Supports sync and async functions
- Automatic parameter injection
- Error handling with retries
- Access to execution context:

```python
def tool_with_context(input: dict, context: ExecutionContext) -> dict:
    # Access session_id, variables, state, etc.
    return {"session": context.session_id}
```

## 6. ConditionNode

Conditional branching with multiple output paths.

**Usage:**

```python
from mesh.nodes import Condition

def check_sentiment(output: dict) -> bool:
    return "positive" in output.get("content", "").lower()

def check_negative(output: dict) -> bool:
    return "negative" in output.get("content", "").lower()

graph.add_node("condition", [
    Condition("positive", check_sentiment, "positive_handler"),
    Condition("negative", check_negative, "negative_handler"),
], node_type="condition", default_target="neutral_handler")

# Add handlers
graph.add_node("positive_handler", None, node_type="llm")
graph.add_node("negative_handler", None, node_type="llm")
graph.add_node("neutral_handler", None, node_type="llm")

# Connect
graph.add_edge("START", "analyzer")
graph.add_edge("analyzer", "condition")
graph.add_edge("condition", "positive_handler")
graph.add_edge("condition", "negative_handler")
graph.add_edge("condition", "neutral_handler")
```

**Condition Object:**

```python
Condition(
    name="condition_name",           # Identifier
    predicate=lambda x: bool,        # Function returning True/False
    target_node="target_node_id"     # Where to route if True
)
```

**Key Features:**
- Multiple conditions per node
- Default fallback path
- Predicates can be any callable
- Unfulfilled branches are skipped

**Advanced Example:**

```python
def is_long_text(output: dict) -> bool:
    content = output.get("content", "")
    return len(content) > 1000

def is_short_text(output: dict) -> bool:
    content = output.get("content", "")
    return len(content) <= 100

graph.add_node("router", [
    Condition("long", is_long_text, "summarizer"),
    Condition("short", is_short_text, "expander"),
], node_type="condition", default_target="normal_handler")
```

## 7. LoopNode

Iterate over arrays and execute downstream nodes for each item.

**Usage:**

```python
graph.add_node("loop", None, node_type="loop",
               array_path="$.items",
               max_iterations=100)

graph.add_node("processor", None, node_type="llm",
               model="gpt-4",
               system_prompt="Process item: {{$iteration}}")

graph.add_edge("START", "loop")
graph.add_edge("loop", "processor")
```

**Parameters:**
- `array_path`: JSONPath to array in input (required)
- `max_iterations`: Maximum loop count (default: 100)

**JSONPath Examples:**

```python
# Top-level array
array_path="$"  # Input: [1, 2, 3]

# Nested field
array_path="$.data.items"  # Input: {"data": {"items": [...]}}

# Array in result
array_path="$.results[*]"  # Input: {"results": [...]}
```

**Iteration Variables:**

Access current iteration data:

```python
# {{$iteration}} - Current item value
# {{$iteration_index}} - Current index (0-based)
# {{$iteration_first}} - True if first item
# {{$iteration_last}} - True if last item

graph.add_node("processor", None, node_type="llm",
               system_prompt="""
               Item #{{{$iteration_index}}}: {{$iteration}}
               {% if $iteration_first %}This is the first item{% endif %}
               """)
```

**Example Workflow:**

Input:
```json
{
  "items": [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
  ]
}
```

Graph:
```python
graph.add_node("loop", None, node_type="loop", array_path="$.items")
graph.add_node("greet", None, node_type="llm",
               system_prompt="Greet {{$iteration.name}}, age {{$iteration.age}}")
graph.add_edge("START", "loop")
graph.add_edge("loop", "greet")
```

## Node Configuration

All nodes support common configuration:

### Retry Logic

```python
graph.add_node("flaky_tool", my_function, node_type="tool",
               config={
                   "retry": {
                       "max_retries": 3,
                       "delay": 1.0  # seconds
                   }
               })
```

### Metadata

```python
graph.add_node("llm", None, node_type="llm",
               metadata={"description": "Analyzer node", "version": "1.0"})
```

## Node Execution

Each node implements the `execute()` method:

```python
async def execute(
    self,
    input: Any,
    context: ExecutionContext
) -> NodeResult:
    # Node logic here
    return NodeResult(
        output={"content": "..."},
        state={"key": "value"},
        chat_history=[...],
        metadata={...}
    )
```

**NodeResult Fields:**
- `output`: Data passed to child nodes
- `state`: Updates to shared state
- `chat_history`: Conversation updates
- `metadata`: Extra information
- `loop_to_node`: For loop nodes
- `max_loops`: Loop limit

## Best Practices

### 1. Choose the Right Node Type

```python
# ✅ Use AgentNode for complex interactions with tools
graph.add_node("agent", vel_agent, node_type="agent")

# ✅ Use LLMNode for simple text generation
graph.add_node("llm", None, node_type="llm", model="gpt-4")

# ✅ Use ToolNode for custom logic
graph.add_node("tool", my_function, node_type="tool")
```

### 2. Keep Nodes Focused

```python
# ✅ Good: Single responsibility
graph.add_node("analyzer", None, node_type="llm",
               system_prompt="Analyze sentiment")
graph.add_node("summarizer", None, node_type="llm",
               system_prompt="Summarize results")

# ❌ Bad: Trying to do too much
graph.add_node("do_everything", None, node_type="llm",
               system_prompt="Analyze, summarize, and respond")
```

### 3. Use Variables

```python
# ✅ Good: Reference previous nodes
graph.add_node("step2", None, node_type="llm",
               system_prompt="Based on {{step1}}, ...")

# ❌ Bad: Hardcoded values
graph.add_node("step2", None, node_type="llm",
               system_prompt="Based on the analysis, ...")
```

### 4. Handle Errors

```python
# ✅ Good: Retries configured
graph.add_node("api_call", fetch_data, node_type="tool",
               config={"retry": {"max_retries": 3}})

# Add error handling
graph.add_node("error_handler", None, node_type="llm")
```

## Common Patterns

### Pattern: Analyze → Process → Respond

```python
graph.add_node("analyzer", None, node_type="llm", model="gpt-4")
graph.add_node("processor", process_func, node_type="tool")
graph.add_node("responder", None, node_type="llm", model="gpt-4")

graph.add_edge("START", "analyzer")
graph.add_edge("analyzer", "processor")
graph.add_edge("processor", "responder")
```

### Pattern: Agent with Tool

```python
graph.add_node("agent", my_agent, node_type="agent")
graph.add_node("tool", my_tool, node_type="tool")
graph.add_edge("START", "agent")
graph.add_edge("agent", "tool")
```

### Pattern: Conditional Routing

```python
graph.add_node("classifier", None, node_type="llm")
graph.add_node("router", conditions, node_type="condition")
graph.add_node("handler_a", None, node_type="llm")
graph.add_node("handler_b", None, node_type="llm")

graph.add_edge("START", "classifier")
graph.add_edge("classifier", "router")
graph.add_edge("router", "handler_a")
graph.add_edge("router", "handler_b")
```

## See Also

- [Graphs](graphs) - Graph structure
- [Execution](execution) - How nodes execute
- [Events](events) - Node event emission
- [Streaming Guide](../guides/streaming) - Streaming patterns
- [Variables Guide](../guides/variables) - Variable resolution
