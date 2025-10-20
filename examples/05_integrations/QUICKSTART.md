# Quickstart: React Flow + Mesh Integration

Fast setup guide for integrating Mesh with a React Flow frontend.

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install 'mesh[server,vel,agents]'
```

### 2. Create Your Backend

```python
# my_backend.py
from fastapi import FastAPI
from mesh import NodeRegistry
from vel import Agent as VelAgent

app = FastAPI()
registry = NodeRegistry()

@app.on_event("startup")
async def startup():
    # Register your agents
    agent = VelAgent(id="my_agent", model={"provider": "openai", "model": "gpt-4o"})
    registry.register_agent("my_agent", agent)

# Copy endpoints from fastapi_with_registry.py
# - GET /api/agents
# - GET /api/tools
# - POST /api/execute
```

### 3. Frontend: Fetch Agents

```typescript
// Fetch available agents on mount
useEffect(() => {
  fetch('http://localhost:8000/api/agents')
    .then(r => r.json())
    .then(data => setAgents(data.agents))
}, [])
```

### 4. Frontend: Reference Agents in React Flow

```typescript
const node = {
  id: 'agent_1',
  type: 'agentAgentflow',
  data: {
    inputs: {
      agent: 'my_agent',  // 👈 String reference to backend agent!
      systemPrompt: 'Process: {{$question}}'
    }
  }
}
```

### 5. Frontend: Execute Graph

```typescript
const response = await fetch('http://localhost:8000/api/execute', {
  method: 'POST',
  body: JSON.stringify({
    flow: { nodes, edges },  // Your React Flow JSON
    input: 'User question here'
  })
})

// Parse SSE stream
const reader = response.body.getReader()
// ... stream events
```

## Key Points

### ✅ DO

**Backend:**
- ✅ Create agents at startup
- ✅ Register in NodeRegistry
- ✅ Use string IDs for references

**Frontend:**
- ✅ Fetch agents list from `/api/agents`
- ✅ Reference agents by string ID
- ✅ Send React Flow JSON to `/api/execute`

### ❌ DON'T

**Backend:**
- ❌ Don't create agents per request (slow!)
- ❌ Don't expose agent internals to frontend
- ❌ Don't skip registration

**Frontend:**
- ❌ Don't try to send agent instances in JSON
- ❌ Don't hardcode agent configurations
- ❌ Don't create agents client-side

## Complete Example

See `fastapi_with_registry.py` for a complete, working implementation.

## Testing

```bash
# 1. Start backend
python examples/05_integrations/fastapi_with_registry.py

# 2. Test endpoints
./examples/05_integrations/test_api.sh

# 3. Check docs
open http://localhost:8000/docs
```

## Architecture

```
React Flow (Frontend)
    ↓
  "my_agent" (string)
    ↓
FastAPI Endpoint
    ↓
NodeRegistry.get_agent("my_agent")
    ↓
Actual Agent Instance
    ↓
Mesh Executor
    ↓
SSE Stream → Frontend
```

## Troubleshooting

**"Agent 'X' not registered"**
→ Make sure you called `registry.register_agent("X", agent)` at startup

**"Module 'vel' not found"**
→ Run `pip install 'mesh[vel]'`

**CORS errors**
→ Add CORS middleware to FastAPI (see example)

**Streaming doesn't work**
→ Check nginx buffering is disabled (`X-Accel-Buffering: no`)

## Next Steps

1. Read `AGENT_REGISTRY_PATTERN.md` for full architecture
2. Check `frontend_example.tsx` for React implementation
3. See `flows/agent_flow.json` for React Flow JSON format
4. Explore advanced patterns in README.md

## Common Patterns

### Multiple Agents

```python
registry.register_agent("researcher", research_agent)
registry.register_agent("writer", writer_agent)
registry.register_agent("qa", qa_agent)
```

### With Tools

```python
def my_tool(input):
    return {"result": ...}

registry.register_tool("my_tool", my_tool)
```

### Per-User Agents

```python
user_registries = {}

@app.post("/api/execute")
async def execute(request: dict, user_id: str):
    if user_id not in user_registries:
        user_registries[user_id] = create_user_registry(user_id)

    registry = user_registries[user_id]
    # ... use user-specific registry
```

## Resources

- **Full Example:** `fastapi_with_registry.py`
- **Architecture:** `AGENT_REGISTRY_PATTERN.md`
- **Frontend:** `frontend_example.tsx`
- **Docs:** Mesh CLAUDE.md
