# 05 - Integrations

This folder contains examples of integrating Mesh with web frameworks, parsers, and external systems.

## Examples

### `fastapi_with_registry.py` ⭐ RECOMMENDED
**Difficulty:** ⭐⭐⭐ Advanced

**The recommended pattern** for integrating Mesh with a React Flow frontend.

Features:
- NodeRegistry for managing agent instances
- Agents registered at startup
- Frontend references agents by string ID
- React Flow JSON with agent references
- SSE streaming for real-time responses
- Endpoints for listing available agents/tools

**What you'll learn:**
- Agent registry pattern
- Declarative graph execution
- Frontend-backend integration
- Runtime agent resolution

**Prerequisites:**
```bash
pip install 'mesh[server,vel,agents]'
```

**Run:**
```bash
python 05_integrations/fastapi_with_registry.py
```

**See also:** `AGENT_REGISTRY_PATTERN.md` for architecture details

---

### `fastapi_server.py`
**Difficulty:** ⭐⭐⭐ Advanced

A simpler FastAPI server example (without registry) that:
- Exposes Mesh graphs via REST API
- Streams responses using Server-Sent Events (SSE)
- Handles state persistence with SQLite
- Provides health checks and error handling

**What you'll learn:**
- Building a web API with Mesh
- SSE streaming for real-time responses
- State management across HTTP requests
- Production patterns (health checks, error handling)

**Prerequisites:**
```bash
pip install 'mesh[server]'  # Installs fastapi, sse-starlette, uvicorn
```

**Run:**
```bash
python 05_integrations/fastapi_server.py
```

**Test:**
```bash
# Health check
curl http://localhost:8000/health

# Execute graph
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test-123"}'
```

### `react_flow_parse.py`
**Difficulty:** ⭐⭐⭐ Advanced

Demonstrates parsing React Flow JSON (Flowise-compatible format) into Mesh execution graphs.

**What you'll learn:**
- Loading graphs from JSON specifications
- Declarative graph definition (alternative to programmatic)
- React Flow node/edge format
- Visual graph editor compatibility

**Run:**
```bash
python 05_integrations/react_flow_parse.py
```

**JSON Format:**
```json
{
  "nodes": [
    {"id": "1", "type": "agent", "data": {...}},
    {"id": "2", "type": "llm", "data": {...}}
  ],
  "edges": [
    {"source": "1", "target": "2"}
  ]
}
```

## Integration Guides

### `AGENT_REGISTRY_PATTERN.md` ⭐ START HERE

**Comprehensive architecture guide** for React Flow + Mesh integration:
- Why the registry pattern is needed
- Frontend-backend separation
- Runtime agent resolution
- Complete code examples
- API endpoint documentation
- Troubleshooting guide

**This is the recommended approach for production applications.**

### `VEL_INTEGRATION.md`

Comprehensive guide for integrating Vel agents with Mesh:
- Setting up Vel SDK
- Agent configuration patterns
- Event handling with Vel agents
- Multi-turn conversations
- State persistence

### `OPENAI_AGENTS_INTEGRATION.md`

Guide for integrating OpenAI Agents SDK:
- OpenAI agent setup
- Tool configuration
- Event translation
- Best practices

### `frontend_example.tsx`

Complete React + TypeScript example showing:
- Fetching available agents from backend
- Building React Flow graphs with agent references
- Executing graphs with SSE streaming
- Real-time event handling
- TypeScript types for Mesh API

## Key Concepts

### FastAPI + Mesh

**Streaming Pattern:**
```python
@app.post("/execute")
async def execute_graph(request: Request):
    # Create executor
    executor = Executor(graph, backend)

    # Stream events as SSE
    async def event_generator():
        async for event in executor.execute(input, context):
            yield {
                "event": event.type,
                "data": json.dumps(event.to_dict())
            }

    return EventSourceResponse(event_generator())
```

**Benefits:**
- ✅ Real-time streaming to frontend
- ✅ Server-Sent Events (SSE) for one-way streaming
- ✅ Compatible with AI SDK UI components
- ✅ Standard HTTP/REST patterns

### React Flow JSON

**Programmatic vs. Declarative:**

```python
# Programmatic (code)
graph = StateGraph()
graph.add_node("agent", agent, node_type="agent")
graph.add_edge("START", "agent")

# Declarative (JSON)
json_spec = {
    "nodes": [{"id": "agent", "type": "agent", ...}],
    "edges": [{"source": "START", "target": "agent"}]
}
graph = parse_react_flow_json(json_spec)
```

**Use Cases:**
- ✅ Visual graph editors (Flowise, React Flow)
- ✅ Dynamic graph generation
- ✅ Graph templates and sharing
- ✅ Non-programmer graph creation

### State Persistence

**Production Pattern:**
```python
# Use SQLite backend for persistence
from mesh.backends import SQLiteBackend

backend = SQLiteBackend("production.db")
executor = Executor(graph, backend)

# State persists across requests
context = ExecutionContext(
    session_id="user-123",  # Same session = same state
    state=await backend.load("user-123") or {}
)
```

**Benefits:**
- ✅ Multi-turn conversations
- ✅ User-specific state
- ✅ Survives server restarts
- ✅ Scalable (can switch to Redis, PostgreSQL)

## Production Deployment

### Environment Variables

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MESH_ENV=production
MESH_LOG_LEVEL=INFO
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat()
    }
```

### Error Handling

```python
try:
    async for event in executor.execute(input, context):
        yield event
except NodeExecutionError as e:
    yield ExecutionEvent(
        type=EventType.EXECUTION_ERROR,
        error=str(e)
    )
```

## Frontend Integration

### React + AI SDK

```typescript
import { useChat } from '@ai-sdk/react'

function ChatComponent() {
  const { messages, isLoading } = useChat({
    api: 'http://localhost:8000/execute',
  })

  // Messages update in real-time as events stream
}
```

### Vanilla JavaScript

```javascript
const eventSource = new EventSource('/execute')

eventSource.addEventListener('token', (e) => {
  const data = JSON.parse(e.data)
  console.log(data.content)  // Stream tokens
})
```

## Sample Flows

The `flows/` directory contains sample React Flow JSON specifications:
- `sample_flow.json` - Example multi-node graph
- Add your own flows here!

## Next Steps

- **06_visualization:** Learn how to visualize your graphs
- **Production Deployment:** Deploy to cloud platforms (AWS, GCP, Azure)
