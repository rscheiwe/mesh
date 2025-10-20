# Agent Registry Pattern for React Flow Integration

This document explains the **recommended architecture** for integrating Mesh with a React frontend using React Flow for graph visualization.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          React Flow Graph Editor                     │  │
│  │                                                       │  │
│  │  [Agent: research_agent] → [Agent: writer_agent]    │  │
│  │           ↓                                          │  │
│  │      "research_agent"  (string reference)           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           │ HTTP POST /api/execute          │
│                           │ { flow: {...}, input: "..." }   │
│                           ↓                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              NodeRegistry                             │  │
│  │                                                       │  │
│  │  "research_agent" → VelAgent(...)     ← Instantiated│  │
│  │  "writer_agent"   → OpenAIAgent(...)  ← Python      │  │
│  │  "calculator"     → calculate_fn(...)  ← Objects    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          ReactFlowParser                             │  │
│  │                                                       │  │
│  │  1. Parse React Flow JSON                           │  │
│  │  2. Resolve "research_agent" → registry lookup      │  │
│  │  3. Create AgentNode with actual agent instance     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Mesh Executor                        │  │
│  │                                                       │  │
│  │  Execute graph with real agent instances            │  │
│  │  Stream events back to frontend (SSE)               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Separation of Concerns

**Frontend (React):**
- Visual graph editor (React Flow)
- References agents by **string ID**
- Sends React Flow JSON to backend
- Receives streaming events

**Backend (FastAPI):**
- Owns actual agent **Python instances**
- Registers agents in NodeRegistry
- Parses React Flow JSON
- Resolves string references → agent instances
- Executes graphs with Mesh

### 2. The Problem This Solves

**Problem:** React Flow JSON can't contain Python objects (agents are classes/instances)

**Solution:** Use string references in JSON, resolve at runtime via registry

```json
// Frontend sends this:
{
  "nodes": [{
    "id": "agent_1",
    "data": {
      "inputs": {
        "agent": "research_agent"  // ← String reference!
      }
    }
  }]
}
```

```python
# Backend resolves this:
agent_ref = "research_agent"
agent = registry.get_agent(agent_ref)  # ← Actual Python object!
node = AgentNode(id="agent_1", agent=agent)
```

### 3. Workflow

#### Backend Startup

```python
# 1. Create registry
registry = NodeRegistry()

# 2. Instantiate and register agents
research_agent = VelAgent(id="research", model={...})
registry.register_agent("research_agent", research_agent)

coding_agent = OpenAIAgent(name="Coder", instructions="...")
registry.register_agent("coding_agent", coding_agent)

# 3. Register tools
def calculate(input):
    return {"result": ...}

registry.register_tool("calculator", calculate)
```

#### Runtime Execution

```python
# 4. Receive React Flow JSON from frontend
flow_json = request.json()["flow"]  # Contains string references

# 5. Parse with registry
parser = ReactFlowParser(registry)
graph = parser.parse(flow_json)  # Resolves "research_agent" → actual agent

# 6. Execute
executor = Executor(graph, backend)
async for event in executor.execute(input, context):
    yield event  # Stream back to frontend
```

## Implementation

### Backend Setup (`fastapi_with_registry.py`)

```python
from fastapi import FastAPI
from mesh import NodeRegistry, ReactFlowParser, Executor

app = FastAPI()
registry = NodeRegistry()

@app.on_event("startup")
async def startup():
    # Register all your agents
    registry.register_agent("research_agent", research_agent)
    registry.register_agent("coding_agent", coding_agent)
    registry.register_tool("calculator", calculate_fn)

@app.get("/api/agents")
async def list_agents():
    """Frontend fetches available agents."""
    return {"agents": [
        {"id": name, "name": name.title()}
        for name in registry.list_agents()
    ]}

@app.post("/api/execute")
async def execute(request: dict):
    """Execute graph from React Flow JSON."""
    parser = ReactFlowParser(registry)
    graph = parser.parse(request["flow"])

    executor = Executor(graph, backend)

    async def stream():
        async for event in executor.execute(request["input"], context):
            yield f"data: {event.to_json()}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
```

### Frontend Usage

```typescript
// 1. Fetch available agents on mount
const [agents, setAgents] = useState([])

useEffect(() => {
  fetch('/api/agents')
    .then(r => r.json())
    .then(data => setAgents(data.agents))
}, [])

// 2. User drags agent onto canvas
function addAgentNode(agentId: string) {
  const node = {
    id: `agent_${Date.now()}`,
    type: 'agentAgentflow',
    data: {
      inputs: {
        agent: agentId  // ← String reference!
      }
    }
  }
  setNodes([...nodes, node])
}

// 3. Execute graph
async function executeGraph() {
  const response = await fetch('/api/execute', {
    method: 'POST',
    body: JSON.stringify({
      flow: { nodes, edges },  // React Flow JSON
      input: userInput
    })
  })

  // Handle SSE stream
  const reader = response.body.getReader()
  // ... parse SSE events
}
```

## API Endpoints

### `GET /api/agents`

Returns list of registered agents for frontend dropdowns.

**Response:**
```json
{
  "agents": [
    {
      "id": "research_agent",
      "name": "Research Agent",
      "type": "vel",
      "description": "Vel agent instance"
    },
    {
      "id": "coding_agent",
      "name": "Coding Agent",
      "type": "openai",
      "description": "OpenAI agent instance"
    }
  ]
}
```

### `GET /api/tools`

Returns list of registered tools.

**Response:**
```json
{
  "tools": [
    {
      "id": "calculator",
      "name": "Calculator",
      "description": "Calculate mathematical expressions"
    }
  ]
}
```

### `POST /api/execute`

Execute graph with streaming.

**Request:**
```json
{
  "flow": {
    "nodes": [
      {
        "id": "agent_1",
        "type": "agentAgentflow",
        "data": {
          "inputs": {
            "agent": "research_agent",
            "systemPrompt": "Research: {{$question}}"
          }
        }
      }
    ],
    "edges": [...]
  },
  "input": "What is quantum computing?",
  "session_id": "session-123"
}
```

**Response:** SSE stream
```
data: {"type":"node_start","node_id":"agent_1"}

data: {"type":"token","content":"Quantum"}

data: {"type":"token","content":" computing"}

data: {"type":"node_complete","node_id":"agent_1"}

data: {"type":"execution_complete","output":"..."}
```

### `POST /api/execute-sync`

Execute graph without streaming (simpler but less responsive).

**Response:**
```json
{
  "success": true,
  "output": "...",
  "events": [...],
  "session_id": "session-123"
}
```

## Benefits of This Architecture

### ✅ Security
- Backend controls what agents are available
- Frontend can't inject arbitrary code
- Agent configurations stay server-side

### ✅ Simplicity
- Frontend just uses string IDs
- No need to serialize/deserialize agent configs
- Standard React Flow JSON format

### ✅ Flexibility
- Easy to add new agents (just register them)
- Can update agent configs without frontend changes
- Support multiple agent frameworks (Vel, OpenAI, custom)

### ✅ Performance
- Agents instantiated once at startup
- No overhead creating agents per request
- Connection pooling, caching work seamlessly

## Running the Example

### 1. Start Backend

```bash
# Install dependencies
pip install 'mesh[vel,agents,server]'

# Set environment variables
export OPENAI_API_KEY=sk-...

# Run server
python examples/05_integrations/fastapi_with_registry.py
```

Backend starts at `http://localhost:8000`

### 2. Test with cURL

```bash
# List agents
curl http://localhost:8000/api/agents

# Execute graph
curl -X POST http://localhost:8000/api/execute \
  -H "Content-Type: application/json" \
  -d @examples/05_integrations/flows/agent_flow.json
```

### 3. Frontend Integration

See `frontend_example.tsx` for a complete React implementation.

```bash
npm install @xyflow/react eventsource-parser
```

## Advanced Patterns

### Dynamic Agent Creation

You can extend the pattern to create agents from configuration:

```python
@app.post("/api/agents/create")
async def create_agent(config: dict):
    """Create and register agent from config."""
    agent = create_agent_from_config(config)
    registry.register_agent(config["id"], agent)
    return {"success": True, "id": config["id"]}
```

Then frontend can create agents on-the-fly.

### Agent Templates

Pre-define agent templates:

```python
AGENT_TEMPLATES = {
    "research": {
        "framework": "vel",
        "model": {"provider": "openai", "model": "gpt-4o"},
        "tools": ["web_search"]
    },
    "coding": {
        "framework": "openai",
        "instructions": "You are an expert programmer"
    }
}

@app.get("/api/agent-templates")
async def list_templates():
    return {"templates": AGENT_TEMPLATES}
```

### Per-User Agents

Store agents per user session:

```python
user_registries: Dict[str, NodeRegistry] = {}

@app.post("/api/execute")
async def execute(request: dict, user_id: str):
    if user_id not in user_registries:
        user_registries[user_id] = create_user_registry(user_id)

    registry = user_registries[user_id]
    # ... use user-specific registry
```

## Troubleshooting

### Agent Not Found

**Error:** `Agent 'my_agent' not registered`

**Solution:** Make sure agent is registered in startup:
```python
@app.on_event("startup")
async def startup():
    registry.register_agent("my_agent", agent_instance)
```

### Import Errors

**Error:** `ImportError: No module named 'vel'`

**Solution:** Install optional dependencies:
```bash
pip install 'mesh[vel]'  # For Vel agents
pip install 'mesh[agents]'  # For OpenAI agents
```

### Graph Parse Errors

**Error:** `Failed to parse graph: Agent node 'agent_1' missing 'agent' reference`

**Solution:** Ensure React Flow JSON includes agent reference:
```json
{
  "data": {
    "inputs": {
      "agent": "research_agent"  // ← Must match registered ID
    }
  }
}
```

## See Also

- `fastapi_with_registry.py` - Complete backend implementation
- `frontend_example.tsx` - React frontend example
- `flows/agent_flow.json` - Sample React Flow JSON
- `react_flow_parse.py` - Simple parser example
- `mesh/parsers/react_flow.py` - Parser implementation
- `mesh/utils/registry.py` - Registry implementation

## Summary

The **Agent Registry Pattern** provides a clean architecture for integrating Mesh with React Flow:

1. **Backend** owns and instantiates agent instances
2. **Registry** maps string IDs to Python objects
3. **Frontend** references agents by string ID in React Flow JSON
4. **Parser** resolves references at runtime
5. **Executor** runs graphs with real agent instances

This keeps the frontend simple while giving the backend full control over agent lifecycle and configuration.
