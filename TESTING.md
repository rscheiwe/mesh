# Testing Mesh Locally

This guide covers how to test the mesh framework on your local machine.

## Quick Start

### 1. Basic Installation Test

```bash
# Clone the repository
git clone <your-repo-url>
cd mesh

# Create virtual environment with UV (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### 2. Run Basic Tests

```bash
# Run the test setup script
python test_setup.py

# This will test:
# - Basic graph creation and execution
# - ToolNode functionality
# - Event streaming
# - Conditional flow
# - State management
# - Mock LLM nodes
```

## Manual Testing

### Test 1: Basic Graph (No API Keys Required)

```python
import asyncio
from mesh import Graph, Edge
from mesh.nodes import CustomFunctionNode
from mesh.compilation import StaticCompiler, GraphExecutor

async def test_basic():
    # Create a simple graph
    graph = Graph()
    
    # Node without incoming edges automatically becomes the starting node
    double = CustomFunctionNode(lambda data, state: {"result": data["value"] * 2})
    
    graph.add_node(double)
    # double is both starting node (no incoming edges) and terminal node (no outgoing edges)
    
    # Compile and execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    
    executor = GraphExecutor()
    result = await executor.execute(compiled, initial_input={"value": 21})
    
    print(f"Result: {result.get_final_output()}")  # Should print: {'result': 42}

asyncio.run(test_basic())
```

### Test 2: Event Streaming

```python
import asyncio
from mesh import Graph, Edge
from mesh.nodes import ToolNode
from mesh.nodes.tool import ToolNodeConfig
from mesh.compilation import StaticCompiler, StreamingGraphExecutor

async def fetch_data(id: str):
    # Simulate API call
    await asyncio.sleep(0.1)
    return {"id": id, "data": f"Data for {id}"}

async def test_streaming():
    graph = Graph()
    
    # Node without incoming edges automatically becomes the starting node
    tool = ToolNode(
        tool_func=fetch_data,
        config=ToolNodeConfig(tool_name="fetcher"),
        extract_args=lambda data: {"id": data["id"]}
    )
    
    graph.add_node(tool)
    # tool is both starting node (no incoming edges) and terminal node (no outgoing edges)
    
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    
    # Stream events
    executor = StreamingGraphExecutor()
    async for event in executor.execute_streaming(compiled, {"id": "test123"}):
        print(f"{event.type}: {event.node_name}")

asyncio.run(test_streaming())
```

### Test 3: With OpenAI (Requires API Key)

```python
import asyncio
import os
from mesh import Graph, Edge
from mesh.nodes import LLMNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.compilation import StaticCompiler, GraphExecutor

async def test_openai():
    # Set your API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    graph = Graph()
    
    # Node without incoming edges automatically becomes the starting node
    llm = LLMNode(config=LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=50
    ))
    
    graph.add_node(llm)
    # llm is both starting node (no incoming edges) and terminal node (no outgoing edges)
    
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    
    executor = GraphExecutor()
    result = await executor.execute(
        compiled,
        initial_input={"prompt": "Say hello in 3 words"}
    )
    
    print(result.get_final_output())

# Only run if you have an API key
# asyncio.run(test_openai())
```

## Testing with UV

### Install and Test

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install mesh with all dependencies
uv pip install -e ".[all]"

# Run tests
python test_setup.py
```

### Run Specific Examples

```bash
# Test basic graph
python examples/simple_graph.py

# Test tool nodes
python examples/tool_node_example.py

# Test event streaming
python examples/event_streaming_example.py

# Test with OpenAI (requires API key)
export OPENAI_API_KEY="your-key"
python examples/openai_example.py
```

## Testing Checklist

- [ ] **Installation**: Package installs without errors
- [ ] **Imports**: All modules import correctly
- [ ] **Basic Graph**: Can create and execute simple graphs
- [ ] **ToolNode**: Functions execute and pass results
- [ ] **Event Streaming**: Events emit during execution
- [ ] **State Management**: State persists across nodes
- [ ] **Conditional Flow**: Branches execute correctly
- [ ] **Parallel Execution**: Multiple nodes run concurrently
- [ ] **Error Handling**: Errors are caught and reported
- [ ] **LLM Integration**: Works with API keys (if available)

## Common Issues

### Import Errors

```bash
# Make sure you're in the mesh directory
cd /path/to/mesh

# Install in development mode
pip install -e .
```

### Missing Dependencies

```bash
# Install all optional dependencies
pip install -e ".[all]"

# Or specific ones
pip install -e ".[openai]"
```

### API Key Issues

```python
# Option 1: Environment variable
os.environ["OPENAI_API_KEY"] = "sk-..."

# Option 2: In code
llm = LLMNode(config=LLMConfig(
    api_key="sk-...",
    # ... other config
))
```

## Unit Tests (Future)

To run the full test suite (when implemented):

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mesh

# Run specific test file
pytest tests/test_graph.py

# Run in verbose mode
pytest -v
```

## Performance Testing

```python
import time
import asyncio
from mesh.compilation import StreamingGraphExecutor, EventCollector

async def performance_test():
    # ... create graph ...
    
    executor = StreamingGraphExecutor()
    collector = EventCollector()
    executor.add_event_handler(collector)
    
    start_time = time.time()
    
    async for event in executor.execute_streaming(compiled, input_data):
        pass
    
    end_time = time.time()
    
    # Analyze performance
    print(f"Total time: {end_time - start_time:.2f}s")
    
    for event in collector.get_events_by_type(EventType.NODE_END):
        print(f"{event.node_name}: {event.data['execution_time']:.3f}s")
```

## Next Steps

1. **Write Unit Tests**: Create proper pytest tests in `tests/` directory
2. **Integration Tests**: Test with real API providers
3. **Performance Benchmarks**: Measure execution times
4. **Load Testing**: Test with many parallel nodes
5. **Error Scenarios**: Test error handling and recovery