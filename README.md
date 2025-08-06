<div align="center">
<img src="https://github.com/user-attachments/assets/ea0a0014-cdff-446c-a500-5cef964c6e0e" width="200" height="200">
<h1>Mesh</h1>

</div>


Mesh is a Python framework for building and orchestrating AI agent graphs across multiple GenAI providers (OpenAI, Anthropic, Google, etc.), utilizing tools, agent, memory, and other node types. It provides a flexible, extensible architecture for creating complex AI workflows with state management and multiple compilation strategies.

## Features

- **Provider Agnostic**: Works with OpenAI, Anthropic, Google, and custom providers
- **Graph-Based Architecture**: Build complex AI workflows as directed graphs
- **Rich Node Types**: LLM nodes, Agent nodes with tools, control flow nodes, and more
- **State Management**: Built-in state tracking and persistence
- **Compilation Strategies**: Static and dynamic graph compilation
- **Parallel Execution**: Automatic detection and execution of parallel nodes
- **Type Safety**: Full type hints and Pydantic models
- **Extensible**: Easy to add custom nodes and providers
- **Loop Protection**: Built-in max_loops parameter prevents infinite loops
- **Flexible Terminal Nodes**: Any node without outgoing edges is automatically terminal
- **Event Streaming**: Real-time observability into graph execution

## Installation

### Using UV (Recommended - Fast)

UV is a fast Python package installer and resolver written in Rust.

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install mesh
uv pip install mesh

# Install with specific providers
uv pip install mesh[openai]      # OpenAI support
uv pip install mesh[anthropic]   # Anthropic support
uv pip install mesh[google]      # Google support
uv pip install mesh[all]         # All providers

# For development
uv pip install -r requirements-dev.txt
```

### Using pip (Traditional)

```bash
pip install mesh

# With specific providers
pip install mesh[openai]      # OpenAI support
pip install mesh[anthropic]   # Anthropic support
pip install mesh[google]      # Google support
pip install mesh[all]         # All providers + dev dependencies
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mesh.git
cd mesh

# Using UV (fast)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from mesh import Graph, Edge
from mesh.nodes import LLMNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.compilation import StaticCompiler, GraphExecutor

async def main():
    # Create a graph
    graph = Graph()

    # Add nodes
    # Nodes without incoming edges automatically become starting nodes
    llm = LLMNode(config=LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        api_key="your-api-key"  # Or use env var OPENAI_API_KEY
    ))

    graph.add_node(llm)
    
    # llm is both the starting node (no incoming edges) and terminal node (no outgoing edges)

    # Compile and execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    executor = GraphExecutor()
    result = await executor.execute(
        compiled,
        initial_input={"prompt": "Hello, AI!"}
    )

    print(result.get_final_output())

asyncio.run(main())
```

## Graph Validation

Mesh automatically validates your graph structure to catch common errors:

### Validation Rules

1. **No disconnected nodes** - All nodes must be reachable
2. **No cycles** - The graph must be a DAG (except for explicit LOOP edges)
3. **Terminal node restrictions** - Only certain node types can be terminal nodes

### Handling Validation Errors

```python
from mesh import Graph
from mesh.nodes import CustomFunctionNode, LLMNode

graph = Graph()

# This will cause a validation error - CustomFunctionNode cannot be terminal
process = CustomFunctionNode(lambda data, state: {"result": data})
graph.add_node(process)  # No outgoing edges = terminal node

# Validate the graph
errors = graph.validate()
if errors:
    for error in errors:
        print(f"Validation error: {error}")
    # Output: "Node CustomFunctionNode cannot be a terminal node..."

# Fix by adding a valid terminal node
llm = LLMNode(config=LLMConfig(...))
graph.add_node(llm)
graph.add_edge(Edge(process.id, llm.id))  # Now process is not terminal

# Validation will also happen during compilation
try:
    compiled = await compiler.compile(graph)
except ValueError as e:
    print(f"Compilation failed: {e}")
```

## Streaming and Sync/Async Support

Mesh provides flexible execution modes for different use cases:

### Execution Modes

1. **Async + Non-streaming** (default):

   ```python
   LLMNode(config=LLMConfig(
       stream=False,  # default
       use_async=True,  # default
   ))
   ```

2. **Async + Streaming**:

   ```python
   LLMNode(config=LLMConfig(
       stream=True,
       use_async=True,
   ))
   ```

3. **Sync + Non-streaming**:

   ```python
   LLMNode(config=LLMConfig(
       stream=False,
       use_async=False,
   ))
   ```

4. **Sync + Streaming**:
   ```python
   LLMNode(config=LLMConfig(
       stream=True,
       use_async=False,
   ))
   ```

### Streaming Example

```python
# Custom node to handle streaming
class StreamingLLMNode(LLMNode):
    async def _execute_impl(self, input_data, state=None):
        messages = self._prepare_messages(input_data, state)
        provider = await self._get_provider()

        stream = await provider.chat_completion(
            messages=messages,
            model=self.llm_config.model,
            stream=True,
        )

        # Process chunks as they arrive
        full_response = []
        async for chunk in stream:
            print(chunk.content, end="", flush=True)
            full_response.append(chunk.content)

        return {"response": "".join(full_response)}
```

### Sync Usage (for non-async environments)

```python
# Direct provider usage for sync environments
from mesh.providers.openai_provider import OpenAIProvider, OpenAIConfig

provider = OpenAIProvider(OpenAIConfig())
response = provider.chat_completion_sync(
    messages=messages,
    model="gpt-3.5-turbo",
    stream=False,
)
print(response.content)
```

## API Key Configuration

Mesh supports multiple ways to configure API keys for maximum flexibility:

### 1. Environment Variables (Recommended for Development)

```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

```python
# The provider will automatically use the environment variable
llm = LLMNode(config=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-3.5-turbo"
    # No api_key needed - uses OPENAI_API_KEY env var
))
```

### 2. Explicit Configuration (Recommended for Production)

```python
# Pass API key directly to the node
llm = LLMNode(config=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-3.5-turbo",
    api_key=get_api_key_from_vault("openai")
))
```

### 3. Multiple Providers in Same Graph

```python
# Different nodes can use different providers and API keys
openai_node = LLMNode(config=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key=openai_key
))

anthropic_node = LLMNode(config=LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-opus",
    api_key=anthropic_key
))
```

### 4. Custom Endpoints (Azure, Local LLMs)

```python
# Configure for Azure OpenAI or custom endpoints
azure_llm = LLMNode(config=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-35-turbo",
    api_key=azure_key,
    api_base="https://myresource.openai.azure.com/",
    provider_options={
        "api_version": "2024-02-01"
    }
))
```

## Node Types

### Node Hierarchy

The mesh framework uses inheritance to create specialized nodes:

```
BaseNode (abstract base)
├── LLMNode (LLM capabilities)
│   └── AgentNode (LLM + tools)
├── ToolNode (function execution)
│   └── MultiToolNode (parallel tools)
├── ConditionalNode (control flow)
├── LoopNode (iteration)
├── CustomFunctionNode (custom logic)
├── HTTPNode (HTTP requests)
└── HumanInputNode (user interaction)
```

### Core Nodes

- **BaseNode**: Base class for custom nodes

**Node Role Detection**:
- **Starting Nodes**: Any node without incoming edges is automatically treated as a starting node
- **Terminal Nodes**: Any node without outgoing edges is automatically treated as a terminal node
  - **Important**: Only certain node types can be terminal nodes:
    - ✅ `LLMNode` - Can be terminal (produces final output)
    - ✅ `AgentNode` - Can be terminal (produces final response)
    - ✅ `ToolNode` - Can be terminal (produces final result)
    - ✅ `MultiToolNode` - Can be terminal (produces final results)
    - ❌ `CustomFunctionNode` - Cannot be terminal (intermediate processing)
    - ❌ `ConditionalNode` - Cannot be terminal (control flow)
    - ❌ `LoopNode` - Cannot be terminal (control flow)
    - ❌ `HumanInputNode` - Cannot be terminal (input gathering)
    - ❌ `HTTPNode` - Cannot be terminal (data fetching)

This design ensures that graphs always terminate with nodes that produce meaningful output rather than intermediate processing steps.

### AI Nodes

#### LLMNode

Direct LLM provider endpoints for text generation:

```python
# Simple LLM for text generation
llm = LLMNode(config=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-3.5-turbo",
    api_key="sk-...",  # or use env var
    temperature=0.7,
    max_tokens=150,
    system_prompt="You are a helpful assistant"
))
```

#### AgentNode

Extends LLMNode with tool capabilities:

```python
# Agent with tools (inherits all LLMConfig options)
agent = AgentNode(config=AgentConfig(
    # All LLMConfig fields are available
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="sk-...",
    temperature=0,

    # Additional agent-specific fields
    tools=[calculator_tool, database_tool],
    max_iterations=5,
    tool_choice="auto"  # or "none" or specific tool name
))
```

**Key Point**: `AgentNode` inherits from `LLMNode`, so it has all LLM capabilities plus tool support. Both nodes share the same API key configuration and provider management.

### Tool Execution Nodes

#### ToolNode

Executes a function and provides its result as context to downstream nodes:

```python
# Fetch data before AI processing
fetch_data = ToolNode(
    tool_func=fetch_user_profile,
    config=ToolNodeConfig(
        tool_name="fetch_user",
        output_key="user_data",
        store_in_state=True
    ),
    extract_args=lambda input_data: {"user_id": input_data["user_id"]}
)

# The result is automatically passed to the next node
llm = LLMNode(config=LLMConfig(...))

# Connect: fetch_data -> llm
# LLM receives: {"user_data": <fetched_data>, ...other_input_data}
```

#### MultiToolNode

Executes multiple tools in parallel or sequence:

```python
# Execute multiple data fetches in parallel
data_fetcher = MultiToolNode(
    tools=[
        ("user_profile", fetch_user_profile),
        ("recommendations", get_recommendations),
        ("analytics", fetch_analytics)
    ],
    parallel=True  # Execute all at once
)
```

**Key Differences**:

- **ToolNode**: Always executes its function, provides results as context
- **AgentNode**: AI decides whether/when to use tools
- **CustomFunctionNode**: Simple function execution without context passing

### Control Flow

- **ConditionalNode**: Branch based on conditions
- **LoopNode**: Iterate with configurable conditions

### Utility Nodes

- **CustomFunctionNode**: Execute custom Python functions
- **HTTPNode**: Make HTTP requests
- **HumanInputNode**: Get human input during execution

## When to Use Each Node Type

### Use ToolNode when you need:

- Pre-fetch data before AI processing
- Execute deterministic functions as part of the workflow
- Gather context from databases, APIs, or files
- Transform or preprocess data for downstream nodes
- Run multiple data fetches in parallel (MultiToolNode)

### Use LLMNode when you need:

- Simple text generation (stories, summaries, translations)
- Question answering without external data
- Text transformation (formatting, style changes)
- Classification or sentiment analysis
- Any task that can be completed with just the LLM's training data

### Use AgentNode when you need:

- AI to decide whether/when to use tools
- Dynamic tool selection based on the query
- Multi-step reasoning with tool use
- Complex workflows where tool usage isn't predetermined

### Example Comparison:

```python
# Task: "Generate email for user_id=123"
# ✅ Use ToolNode + LLMNode - deterministic data fetch, then generation
fetch_user = ToolNode(tool_func=get_user_data)
email_writer = LLMNode(config=LLMConfig(...))
# Connect: fetch_user -> email_writer

# Task: "Write a poem about Python"
# ✅ Use LLMNode - creative task, no external data needed
poet = LLMNode(config=LLMConfig(
    model="gpt-3.5-turbo",
    temperature=0.9
))

# Task: "Answer questions about our products (may need DB lookup)"
# ✅ Use AgentNode - AI decides if/when to query database
assistant = AgentNode(config=AgentConfig(
    model="gpt-3.5-turbo",
    tools=[database_tool, calculator_tool]
))
```

## Event Streaming and Observability

Mesh provides real-time event streaming for complete visibility into graph execution:

### Event Types

```python
from mesh.core.events import EventType

# Available event types:
EventType.GRAPH_START      # Graph execution started
EventType.GRAPH_END        # Graph execution completed
EventType.NODE_START       # Node execution started
EventType.NODE_END         # Node execution completed
EventType.TOOL_START       # Tool execution started
EventType.TOOL_END         # Tool execution completed
EventType.STREAM_CHUNK     # LLM streaming content chunk
EventType.NODE_ERROR       # Error in node execution
```

### Basic Event Streaming

```python
from mesh.compilation import StreamingGraphExecutor

# Create streaming executor
executor = StreamingGraphExecutor()

# Stream events as they occur
async for event in executor.execute_streaming(compiled_graph):
    print(f"[{event.timestamp}] {event.type}: {event.node_name}")

    # Access event data
    if event.type == EventType.TOOL_START:
        print(f"  Tool: {event.data['tool_name']}")
        print(f"  Args: {event.data['tool_args']}")
```

### Event Handlers

```python
# Custom event handler
class MyEventHandler:
    async def handle_event(self, event):
        if event.type == EventType.NODE_ERROR:
            await send_alert(f"Error in {event.node_name}: {event.data['error']}")

# Add handler to executor
executor = StreamingGraphExecutor()
executor.add_event_handler(MyEventHandler().handle_event)
```

### Event Collection and Analysis

```python
from mesh.compilation import EventCollector

# Collect events for analysis
collector = EventCollector()
executor.add_event_handler(collector)

# Execute graph
async for event in executor.execute_streaming(compiled_graph):
    pass  # Events are automatically collected

# Analyze events
node_events = collector.get_node_events("node_123")
tool_events = collector.get_events_by_type(EventType.TOOL_END)

# Calculate execution times
for event in collector.get_events_by_type(EventType.NODE_END):
    print(f"{event.node_name}: {event.data['execution_time']:.2f}s")
```

### Real-Time Monitoring Example

```python
# Monitor parallel execution
async for event in executor.execute_streaming(compiled_graph):
    if event.type == EventType.NODE_START:
        print(f"🟢 Started: {event.node_name}")
    elif event.type == EventType.NODE_END:
        status = "✅" if event.data['success'] else "❌"
        print(f"{status} Completed: {event.node_name} ({event.data['execution_time']:.2f}s)")
    elif event.type == EventType.TOOL_START:
        print(f"🔧 Tool: {event.data['tool_name']} starting...")
    elif event.type == EventType.STREAM_CHUNK:
        print(event.data['content'], end='', flush=True)
```

### Event Data Structure

Each event contains:

- `type`: The event type (EventType enum)
- `timestamp`: When the event occurred
- `node_id`: ID of the related node (if applicable)
- `node_name`: Name of the related node (if applicable)
- `data`: Event-specific data dictionary
- `metadata`: Additional metadata

### Integration with Streaming LLMs

When using streaming LLMs, content chunks are emitted as events:

```python
# Configure LLM for streaming
llm = LLMNode(config=LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-3.5-turbo",
    stream=True  # Enable streaming
))

# Receive stream chunks as events
async for event in executor.execute_streaming(compiled_graph):
    if event.type == EventType.STREAM_CHUNK:
        print(event.data['content'], end='', flush=True)
```

## Loop Handling and Cycles

Mesh supports graph cycles with built-in loop protection:

### Max Loops Configuration

```python
# Global max loops for the executor
executor = GraphExecutor(max_loops=100)  # Default: 100

# Override per execution
result = await executor.execute(
    compiled_graph,
    initial_input=data,
    max_loops=50  # Override for this execution
)
```

### Creating Loops

```python
from mesh.core.edge import EdgeType

# Create a loop structure
counter = CustomFunctionNode(lambda data, state: {"count": data.get("count", 0) + 1})
condition = ConditionalNode(
    condition=lambda data, state: data.get("count", 0) < 10,
    true_output=lambda data: data,
    false_output=lambda data: {"count": data.get("count"), "done": True}
)

# Loop back edge
graph.add_edge(Edge(
    condition.id,
    counter.id,
    edge_type=EdgeType.LOOP,
    condition=lambda data: not data.get("done", False)
))
```

### Terminal Nodes

Any node without outgoing edges is automatically a terminal node:

```python
# These are all terminal nodes:
final_llm = LLMNode(config=config)       # No outgoing edges
result_tool = ToolNode(tool_func=save)   # No outgoing edges
output = CustomFunctionNode(format_output) # No outgoing edges

# The graph can have multiple terminal nodes
graph.add_edge(Edge(decision.id, final_llm.id))    # Terminal path 1
graph.add_edge(Edge(decision.id, result_tool.id))  # Terminal path 2
```

## State Management

Mesh provides comprehensive state management:

```python
from mesh.state import GraphState

# State is automatically passed through nodes
state = GraphState()

# Nodes can read/write state
await state.set("key", "value")
value = await state.get("key")

# Persist state
await state.save_to_file(Path("state.json"))
state = await GraphState.load_from_file(Path("state.json"))
```

## Compilation Strategies

### Static Compilation

Pre-compiles the entire graph for predictable execution:

```python
from mesh.compilation import StaticCompiler

compiler = StaticCompiler()
compiled = await compiler.compile(graph)
```

### Dynamic Compilation

Builds execution path at runtime based on conditions:

```python
from mesh.compilation import DynamicCompiler

compiler = DynamicCompiler()
compiled = await compiler.compile(graph)
```


## Creating Custom Nodes

```python
from mesh.nodes.base import BaseNode

class MyCustomNode(BaseNode):
    async def _execute_impl(self, input_data, state):
        # Your custom logic here
        result = input_data["value"] * 2
        return {"result": result}
```

## Agent Tools

Create tools for agent nodes:

```python
from mesh.nodes.agent import Tool, ToolResult

class DatabaseTool(Tool):
    def __init__(self):
        super().__init__(
            name="database",
            description="Query the database"
        )

    async def execute(self, query: str) -> ToolResult:
        # Execute database query
        results = await db.query(query)
        return ToolResult(success=True, output=results)
```

## Examples

See the `examples/` directory for comprehensive examples:

### Basic Examples

- `simple_graph.py`: Basic graph with conditional routing
- `agent_graph.py`: Agent with multiple tools
- `openai_example.py`: Using OpenAI provider with tools

### Advanced Examples

- `tool_node_example.py`: Using ToolNode for data preprocessing
- `llm_vs_agent_nodes.py`: Comparison of LLMNode and AgentNode
- `api_key_configuration.py`: Different ways to configure API keys
- `streaming_example.py`: Streaming and sync/async execution modes
- `event_streaming_example.py`: Real-time graph execution monitoring
- `feedback_loop_example.py`: Feedback loop with conditional routing

## Key Features Summary

### Execution Modes

- **Standard Execution**: Simple, synchronous result retrieval
- **Streaming Execution**: Real-time event streaming for observability
- **Async/Sync Support**: Both async and sync APIs for flexibility
- **Parallel Execution**: Automatic parallel node execution when possible

### Observability

- **Event Streaming**: Complete visibility into graph execution
- **Performance Metrics**: Execution time tracking per node
- **Error Tracking**: Detailed error events with context
- **Tool Monitoring**: Track tool invocations and results

### Comparison: Standard vs Streaming Execution

```python
# Standard Execution
executor = GraphExecutor()
result = await executor.execute(compiled_graph, input_data)
print(result.get_final_output())

# Streaming Execution (with events)
executor = StreamingGraphExecutor()
async for event in executor.execute_streaming(compiled_graph, input_data):
    # Real-time visibility into execution
    print(f"{event.type}: {event.node_name}")
```

## Architecture

Mesh follows a modular architecture:

```
mesh/
├── core/          # Core graph components (Node, Edge, Graph, Events)
├── nodes/         # Node implementations
├── state/         # State management
├── compilation/   # Compilation strategies and executors
├── providers/     # Provider-specific implementations
└── utils/         # Utility functions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
