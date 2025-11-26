---
layout: default
title: Nodes
parent: Concepts
nav_order: 2
---

# Nodes

Nodes are the building blocks of Mesh workflows. Each node performs a specific task and can emit streaming events.

## Node Types

Mesh provides 8 core node types:

| Type | Purpose | Key Features |
|------|---------|--------------|
| **StartNode** | Graph entry point | Implicit, auto-created |
| **EndNode** | Graph exit point | Implicit, optional |
| **AgentNode** | Vel/OpenAI agents | Streaming, auto-detection |
| **LLMNode** | Direct LLM calls | Streaming, simple |
| **ToolNode** | Python functions | Sync/async support |
| **RAGNode** | Document retrieval | Vector search, context enrichment |
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

## 6. RAGNode

Retrieve documents from vector stores for context enrichment in LLM prompts.

**Basic Usage:**

```python
from mesh.nodes import RAGNode

# Create RAG node
rag_node = RAGNode(
    id="rag_0",
    query_template="{{$question}}",  # What to search for
    top_k=5,                          # Number of documents
    similarity_threshold=0.7,         # Minimum score
    file_id="uuid-123",              # Filter to specific file
)

# Inject retriever (dependency injection pattern)
rag_node.set_retriever(retriever)

graph.add_node("rag_0", rag_node, node_type="rag")
```

**Parameters:**
- `id`: Node identifier (e.g., `"rag_0"`)
- `query_template`: Search query with variable support (default: `"{{$question}}"`)
- `top_k`: Number of documents to retrieve (default: 5)
- `similarity_threshold`: Minimum similarity score 0.0-1.0 (default: 0.7)
- `file_id`: UUID of specific file to search (optional)
- `folder_uuid`: UUID of folder to search across (optional)
- `retriever_type`: Type of vector store - `"postgres"` or `"chroma"` (default: `"postgres"`)

### Query Template - The Search Query

The **Query Template** determines what text gets embedded and searched against your vector database chunks.

**Default: User's Question**
```python
query_template="{{$question}}"  # Searches based on user input
```

Example flow:
```
User asks: "What cats are good in hot weather?"
           ↓
Query Template: {{$question}}
           ↓
Resolved: "What cats are good in hot weather?"
           ↓
Generate embedding: [0.123, 0.456, ...] (1536 dims)
           ↓
Search postgres: Returns top 5 similar chunks
```

**Dynamic Queries: Reference Previous Nodes**
```python
# Search based on LLM's refined query
query_template="{{llm_0.output}}"

# Combine multiple inputs
query_template="Find docs about {{$question}} related to {{analyzer.topic}}"

# Use tool output
query_template="{{query_refiner.refined_query}}"
```

**Static Queries: Fixed Search**
```python
# Always search for specific topic
query_template="product specifications and features"

# Domain-specific search
query_template="customer support policies"
```

### Output Structure

RAGNode outputs a dictionary with:

```python
{
    "formatted": "<CONTEXT>...formatted docs with metadata...</CONTEXT>",
    "documents": [
        {
            "id": "uuid",
            "document_id": "file-uuid",
            "content": "...",
            "page_number": 5,
            "heading": "Section Title",
            "similarity": 0.89,
            "file_title": "Document.pdf",
            "folder_uuid": "folder-uuid"
        },
        ...
    ],
    "query": "resolved query text",
    "num_results": 5
}
```

Access in downstream nodes:
- `{{rag_0.output.formatted}}` - Pre-formatted context block for LLM prompts
- `{{rag_0.output.documents}}` - Raw document array for custom processing
- `{{rag_0.output.query}}` - The resolved query that was searched

### Complete RAG Flow

```python
from mesh import StateGraph
from mesh.nodes import RAGNode

# 1. Create graph
graph = StateGraph()

# 2. Add RAG node
rag_node = RAGNode(
    id="rag_0",
    query_template="{{$question}}",
    top_k=5,
    similarity_threshold=0.7,
    file_id="file-uuid-123",  # Search specific file
)

# 3. Add LLM that uses retrieved context
graph.add_node("llm", None, node_type="llm",
               model="gpt-4",
               system_prompt="""
               Answer using this context:
               {{rag_0.output.formatted}}

               Question: {{$question}}
               """)

# 4. Connect nodes
graph.add_edge("START", "rag_0")
graph.add_edge("rag_0", "llm")

# 5. Inject retriever (before execution)
rag_node.set_retriever(my_retriever)

# 6. Execute
result = await graph.run(input="What cats are good in hot weather?")
```

### Retriever Setup

RAGNode uses **dependency injection** for the retriever:

```python
# Example retriever interface
class MyRetriever:
    async def search_file(self, query: str, file_id: str,
                         similarity_threshold: float, limit: int):
        # Generate embedding
        embedding = await generate_embedding(query)

        # Query vector database
        results = query_pgvector(embedding, file_id, similarity_threshold, limit)
        return results

    async def search_folder(self, query: str, folder_uuid: str,
                           similarity_threshold: float, limit: int):
        # Search across folder
        ...

# Inject into node
retriever = MyRetriever()
rag_node.set_retriever(retriever)
```

**In React Flow Graphs:**

When using React Flow parser, inject retrievers after parsing:

```python
from mesh import ReactFlowParser
from mesh.nodes import RAGNode

# Parse graph
graph = parser.parse(react_flow_json)

# Inject retriever into all RAG nodes
retriever = MyRetriever()
for node in graph.nodes.values():
    if isinstance(node, RAGNode):
        node.set_retriever(retriever)

# Execute
await graph.run(input="...")
```

### File vs Folder Search

**File Search** - Search within specific document:
```python
RAGNode(
    id="rag_0",
    file_id="550e8400-e29b-41d4-a716-446655440000",  # Specific file UUID
    top_k=5
)
```

**Folder Search** - Search across all documents in folder:
```python
RAGNode(
    id="rag_0",
    folder_uuid="abc-123-def",  # Folder UUID
    top_k=10  # Aggregates top results across all files
)
```

**Note:** Use either `file_id` OR `folder_uuid`, not both.

### Multi-Source RAG

Search multiple knowledge bases by using multiple RAG nodes:

```python
# Search product docs
rag_products = RAGNode(
    id="rag_products",
    query_template="{{$question}}",
    file_id="products-file-id",
    top_k=3
)

# Search support tickets
rag_support = RAGNode(
    id="rag_support",
    query_template="{{$question}}",
    file_id="support-file-id",
    top_k=3
)

# LLM uses both sources
graph.add_node("llm", None, node_type="llm",
               system_prompt="""
               Product context: {{rag_products.output.formatted}}
               Support context: {{rag_support.output.formatted}}

               Question: {{$question}}
               """)

# Connect
graph.add_edge("START", "rag_products")
graph.add_edge("START", "rag_support")  # Parallel retrieval
graph.add_edge("rag_products", "llm")
graph.add_edge("rag_support", "llm")
```

### Query Refinement Pattern

Use an LLM to refine the search query before retrieval:

```python
# Step 1: Refine query
graph.add_node("refiner", None, node_type="llm",
               model="gpt-4",
               system_prompt="Rewrite as a search query: {{$question}}")

# Step 2: Search with refined query
rag_node = RAGNode(
    id="rag_0",
    query_template="{{refiner.output}}",  # Use LLM's refined query
    top_k=5
)

# Step 3: Answer with context
graph.add_node("answerer", None, node_type="llm",
               system_prompt="Answer using: {{rag_0.output.formatted}}")

# Connect
graph.add_edge("START", "refiner")
graph.add_edge("refiner", "rag_0")
graph.add_edge("rag_0", "answerer")
```

### When to Use RAGNode

**✅ Use RAGNode when:**
- You need to ground LLM responses in specific documents
- Working with large knowledge bases (docs, tickets, articles)
- Need up-to-date information from your own data
- Want to cite sources in LLM responses
- Building Q&A systems over documentation

**❌ Don't use RAGNode when:**
- Information fits in LLM context window
- No vector database available
- Real-time web search needed (use tool with API instead)
- Documents change too frequently for embeddings

## 7. ConditionNode

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

## 8. LoopNode

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
