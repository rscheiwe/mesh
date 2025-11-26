# Manual Graph Creation Patterns

This guide shows how to use Mesh as an imported Python package for manual graph creation.

## Pattern 1: Define Tools Inline

```python
from mesh import StateGraph
from mesh.nodes import ToolNode

# Define tool function
def my_tool(input: str) -> dict:
    return {"result": f"Processed: {input}"}

# Create graph
graph = StateGraph()

# Add tool node
tool_node = ToolNode(id="my_tool", tool_fn=my_tool)
graph.add_node("my_tool", tool_node, node_type="tool")

graph.add_edge("START", "my_tool")
graph.set_entry_point("my_tool")

compiled = graph.compile()
```

## Pattern 2: Import Tools from Separate File

**my_tools.py:**
```python
def calculate_sum(a: int, b: int) -> dict:
    """Add two numbers."""
    return {"result": a + b}

def fetch_data(url: str, context) -> dict:
    """Fetch data from API."""
    import requests
    response = requests.get(url)
    return {"data": response.json()}

async def async_processor(input: str, context) -> dict:
    """Process data asynchronously."""
    await asyncio.sleep(1)
    return {"processed": input.upper()}
```

**main.py:**
```python
from mesh import StateGraph
from mesh.nodes import ToolNode
from my_tools import calculate_sum, fetch_data, async_processor

graph = StateGraph()

# Add tools
graph.add_node("sum", ToolNode(id="sum", tool_fn=calculate_sum))
graph.add_node("fetch", ToolNode(id="fetch", tool_fn=fetch_data))
graph.add_node("process", ToolNode(id="process", tool_fn=async_processor))

# Connect
graph.add_edge("START", "sum")
graph.add_edge("sum", "fetch")
graph.add_edge("fetch", "process")

graph.set_entry_point("sum")
compiled = graph.compile()
```

## Pattern 3: Tool with Bindings (Pre-configured Parameters)

```python
from mesh.nodes import ToolNode

def multiply(a: int, b: int) -> dict:
    return {"result": a * b}

# Pre-configure parameters via bindings
tool_node = ToolNode(
    id="multiply_by_10",
    tool_fn=multiply,
    config={"bindings": {"b": 10}}  # b is always 10
)

# When executed, only 'a' needs to be provided via input
```

## Pattern 4: Tool with Context Access

```python
def smart_tool(input: str, context) -> dict:
    """Tool that accesses execution context."""
    # Access previous node outputs
    prev_output = context.get_node_output("previous_node")

    # Access global variables
    api_key = context.variables.get("api_key")

    # Access chat history
    history = context.chat_history

    return {"result": f"Processed with context: {input}"}
```

## Pattern 5: Combined with AgentNode (No Registry)

```python
from vel import Agent as VelAgent
from mesh import StateGraph
from mesh.nodes import AgentNode, ToolNode

# Create Vel Agent inline
vel_agent = VelAgent(
    id="my_agent",
    model={
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
)

# Create tool
def helper_tool(input: str) -> dict:
    return {"data": f"Helper: {input}"}

# Build graph
graph = StateGraph()

agent_node = AgentNode(
    id="agent_0",
    agent=vel_agent,
    system_prompt="You are helpful"
)

tool_node = ToolNode(id="helper", tool_fn=helper_tool)

graph.add_node("agent_0", agent_node, node_type="agent")
graph.add_node("helper", tool_node, node_type="tool")

graph.add_edge("START", "helper")
graph.add_edge("helper", "agent_0")

graph.set_entry_point("helper")
compiled = graph.compile()
```

## Key Points

1. **No Registry Needed**: For manual graphs, you don't need a registry
2. **Direct Function Passing**: Pass functions directly to `ToolNode(tool_fn=...)`
3. **Import from Files**: Organize tools in separate modules
4. **Sync or Async**: Both sync and async functions work
5. **Context Access**: Tools can access `context` for previous outputs, variables, etc.
6. **Bindings**: Pre-configure parameters via `config={"bindings": {...}}`

## When to Use Each Pattern

- **Inline**: Quick scripts, testing, simple tools
- **Imported**: Production code, reusable tools, team projects
- **Bindings**: Tools with default parameters
- **Context**: Tools that need access to graph state or previous outputs

## React Flow vs Manual

| Feature | React Flow (UI) | Manual (Python) |
|---------|----------------|-----------------|
| Tool Source | DB (`toolUuid`) or Registry | Python function |
| Agent Source | Inline config or Registry | Vel Agent instance |
| Use Case | Visual workflow building | Code-based automation |
| Registration | Optional (DB-backed) | Not needed |
| Flexibility | UI-driven | Full Python control |

Both approaches work seamlessly - choose based on your use case!
