---
layout: default
title: Troubleshooting
nav_order: 9
---

# Troubleshooting

Common issues and solutions.

## Installation Issues

### "No module named 'mesh'"

**Problem:** Mesh is not installed.

**Solution:**
```bash
pip install mesh
# or for development
pip install -e ".[all]"
```

### "No module named 'vel'" or "No module named 'agents'"

**Problem:** Optional dependencies not installed.

**Solution:**
```bash
# For Vel SDK
pip install "mesh[vel]"

# For OpenAI Agents SDK
pip install "mesh[agents]"

# For all features
pip install "mesh[all]"
```

## Configuration Issues

### "Illegal header value" or API Key Errors

**Problem:** API key not set or invalid.

**Solution:**

1. Check `.env` file exists:
```bash
ls -la .env
```

2. Verify content:
```bash
# Should contain:
OPENAI_API_KEY=sk-...
```

3. Load environment variables:
```python
from mesh.utils import load_env
load_env()
```

4. Check no extra spaces/quotes:
```bash
# ❌ Wrong
OPENAI_API_KEY="sk-..."

# ✅ Correct
OPENAI_API_KEY=sk-...
```

### "Connection refused" (Database)

**Problem:** Database not available.

**Solution:** Use MemoryBackend for development:
```python
from mesh.backends import MemoryBackend
backend = MemoryBackend()
```

## Graph Issues

### "Graph validation failed: No entry point"

**Problem:** Forgot to set entry point.

**Solution:**
```python
graph.set_entry_point("first_node")
compiled = graph.compile()
```

### "Cycle detected in graph"

**Problem:** Edges create an uncontrolled cycle (no loop controls).

**Solution:** Mark cycle edges as controlled loops with `is_loop_edge=True` and add controls.

```python
# ❌ Wrong: Uncontrolled cycle
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge("C", "A")  # Error: uncontrolled cycle!

# ✅ Correct: Controlled loop with max_iterations
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge(
    "C",
    "A",
    is_loop_edge=True,
    max_iterations=10
)

# ✅ Correct: Controlled loop with condition
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge(
    "C",
    "A",
    is_loop_edge=True,
    loop_condition=lambda state, output: not output.get("done", False),
    max_iterations=50  # Optional safety limit
)
```

### "Orphaned nodes detected"

**Problem:** Node not connected to START.

**Solution:**
```python
graph.add_edge("START", "orphaned_node")
# or
graph.add_edge("parent_node", "orphaned_node")
```

### "Node not found: START"

**Problem:** Trying to create a node called "START".

**Solution:** Don't create START node. It's implicit. Just reference it:
```python
# ❌ Wrong
graph.add_node("START", ...)

# ✅ Correct
graph.add_edge("START", "first_node")
```

## Execution Issues

### No Streaming Output

**Problem:** Not checking correct event type or not flushing.

**Solution:**
```python
# ✅ Correct
async for event in executor.execute(input, context):
    if event.type == "token":
        print(event.content, end="", flush=True)  # flush=True!
```

### Import Errors for Vel/OpenAI

**Problem:** Vel or OpenAI Agents SDK not installed but trying to use event translation.

**Solution:**

```python
# ✅ Option 1: Install dependencies
pip install "mesh[vel]"
pip install "mesh[agents]"

# ✅ Option 2: Use native events
graph.add_node("agent", agent, node_type="agent",
               use_native_events=True)
```

### "Event translator not available"

**Problem:** Trying to use Vel translation without Vel installed.

**Solution:**

Mesh automatically falls back to native events. If you want translation:
```bash
pip install "mesh[vel]"
```

## Event Translation Issues

### Different Events Than Expected

**Problem:** Using native events instead of Vel-translated.

**Check:**
```python
# Are you using use_native_events=True?
graph.add_node("agent", agent, node_type="agent",
               use_native_events=False)  # Default: Vel translation
```

### "AttributeError: translate"

**Problem:** Old Vel version or wrong import.

**Solution:**
```bash
# Update Vel
pip install --upgrade vel

# Correct import
from vel import get_openai_agents_translator
```

## Performance Issues

### Slow Execution

**Check:**
1. Using appropriate backend
2. Not blocking event loop
3. Network latency

**Solutions:**
```python
# Use memory backend for development
from mesh.backends import MemoryBackend
backend = MemoryBackend()

# Ensure async/await properly used
async for event in executor.execute(input, context):
    # Don't block here
    pass
```

### High Memory Usage

**Problem:** Large state objects or chat history.

**Solution:**
```python
# Limit chat history
context.chat_history = context.chat_history[-10:]  # Keep last 10

# Use SQLite for large state
from mesh.backends import SQLiteBackend
backend = SQLiteBackend("mesh.db")
```

## Testing Issues

### Tests Fail with API Errors

**Problem:** No API key in test environment.

**Solution:**

1. Mock API calls:
```python
@pytest.fixture
def mock_openai():
    with patch('openai.ChatCompletion.create'):
        yield
```

2. Or use test key in `.env.test`

### AsyncIO Errors

**Problem:** Not using pytest-asyncio.

**Solution:**
```bash
pip install pytest-asyncio
```

```python
import pytest

@pytest.mark.asyncio
async def test_execution():
    async for event in executor.execute(...):
        pass
```

## Still Having Issues?

1. **Check Examples:** See [examples/](https://github.com/rscheiwe/mesh/tree/main/examples)
2. **Read Docs:** Visit [full documentation](https://rscheiwe.github.io/mesh)
3. **Check GitHub Issues:** [Open issues](https://github.com/rscheiwe/mesh/issues)
4. **Create Issue:** [Report a bug](https://github.com/rscheiwe/mesh/issues/new)

## Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run your code
async for event in executor.execute(input, context):
    print(f"Event: {event.type} - Node: {event.node_id}")
```

## Common Warnings

### "Vel not available, falling back to native events"

**Meaning:** Vel SDK not installed, using native events instead.

**Fix:** `pip install "mesh[vel]"` or ignore if you want native events.

### "State backend not persisting"

**Meaning:** Using MemoryBackend which doesn't persist.

**Fix:** Use SQLiteBackend for production:
```python
from mesh.backends import SQLiteBackend
backend = SQLiteBackend("mesh.db")
```
