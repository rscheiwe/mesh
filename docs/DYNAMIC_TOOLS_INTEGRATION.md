# Dynamic Tools Integration - Mesh + Vel v0.3.0+

**Date:** November 25, 2025
**Status:** Complete

## Overview

Mesh now fully supports Vel's dynamic tools upgrade (v0.3.0+), enabling instance-scoped tools without global registration. This aligns with modern best practices and eliminates global state pollution.

## What Changed

### Before (Deprecated Pattern)

```python
# ❌ OLD: Global registry
from vel.tools import register_tool, ToolSpec

tool = ToolSpec(...)
register_tool(tool)  # Global registration required
agent = VelAgent(tools=['tool_name'])  # String reference
```

### After (New Pattern)

```python
# ✅ NEW: Instance-scoped
from vel.tools import ToolSpec

tool = ToolSpec.from_function(my_function)  # Auto-schema generation
agent = VelAgent(tools=[tool])  # Direct instance
```

## Implementation in Mesh

### 1. Programmatic API (StateGraph)

**Pattern:** Tools configured at agent instantiation, then agent wrapped in AgentNode.

```python
from vel import Agent as VelAgent
from vel.tools import ToolSpec
from mesh import StateGraph

# Define and wrap tools
def get_weather(city: str) -> dict:
    return {"temp": 72, "condition": "sunny"}

weather_tool = ToolSpec.from_function(get_weather)

# Create agent WITH tools
agent = VelAgent(
    id="assistant",
    model={"provider": "openai", "model": "gpt-4o"},
    tools=[weather_tool],  # Tools part of agent config
)

# Wrap pre-configured agent in AgentNode
graph = StateGraph()
graph.add_node("agent", agent, node_type="agent")
```

**Key Point:** AgentNode does NOT add tools - it wraps an agent that already has tools.

### 2. Declarative API (React Flow JSON)

**Pattern:** Tools defined inline as Python code, automatically wrapped in ToolSpec.

```json
{
  "id": "agent_0",
  "type": "agentAgentflow",
  "data": {
    "inputs": {
      "provider": "openai",
      "modelName": "gpt-4o",
      "systemPrompt": "You are a helpful assistant with tools.",
      "tools": [
        {
          "code": "def get_weather(city: str) -> dict:\n    '''Get weather for a city.'''\n    return {'temp': 72, 'condition': 'sunny'}",
          "name": "get_weather",
          "description": "Optional override for docstring"
        }
      ]
    }
  }
}
```

**Implementation:**
- ReactFlowParser extracts tool code from JSON
- Executes code to get function
- Wraps in ToolSpec using `ToolSpec.from_function()`
- Passes tools array to VelAgent constructor

**Code Location:** `mesh/parsers/react_flow.py`
- Lines 257-273: Agent creation with tools
- Lines 288-363: `_create_tools_from_config()` method

## Tool vs ToolNode

### AgentNode with Tools (LLM-Controlled)

Use when the LLM decides when to call tools:

```python
agent = VelAgent(tools=[weather_tool])
graph.add_node("agent", agent, node_type="agent")
```

The agent uses Vel's runtime to:
1. Decide if tool is needed
2. Generate tool arguments
3. Execute tool
4. Process result

### ToolNode (Mesh-Orchestrated)

Use when Mesh controls tool execution in a deterministic flow:

```python
graph.add_node("weather", get_weather, node_type="tool")
graph.add_edge("agent", "weather")  # Always called after agent
```

Mesh executes the tool and passes result to next node.

## Benefits

1. **No Global State**
   - Each agent has its own tools
   - Better for testing (no cleanup needed)
   - Better for multi-tenancy (no tool name collisions)

2. **Type Safety**
   - No string magic (`tools=['name']` → `tools=[tool_spec]`)
   - IDE autocomplete works
   - Refactoring safe

3. **Runtime Creation**
   - Tools can be created at runtime
   - Perfect for UI-based tool builders
   - No app restart needed

4. **Auto-Schema Generation**
   - Schemas extracted from type hints
   - Less boilerplate (3 lines vs 15+ lines)
   - Single source of truth

## Examples

### Programmatic Example

See: `examples/test_agent_with_dynamic_tools.py`

Demonstrates:
- Defining tool functions
- Wrapping in ToolSpec
- Creating agent with tools
- Executing with tool calls
- Multiple tools on same agent

### React Flow Example

See: `examples/test_reactflow_agent_with_inline_tools.py`

Demonstrates:
- Inline tool definition in JSON
- Automatic ToolSpec wrapping
- Tool execution via Vel runtime
- Multiple inline tools

## Testing

Both examples include full execution tests showing:
- Tool call events (`tool-input-available`)
- Tool output events (`tool-output-available`)
- Multi-step execution (LLM → tool → LLM)
- Streaming responses

## Backwards Compatibility

Old patterns still work but emit deprecation warnings:

```python
# Still works, but warns
register_tool(tool)  # DeprecationWarning
agent = VelAgent(tools=['tool_name'])  # DeprecationWarning
```

**Removal Timeline:**
- v0.3.0 (current): Warnings
- v1.x: Warnings continue
- v2.0: Breaking changes (removal)

## Migration Checklist

- [x] ReactFlow parser supports inline tools
- [x] Tools wrapped in ToolSpec automatically
- [x] Tools passed to VelAgent constructor
- [x] Programmatic API examples created
- [x] Declarative API examples created
- [x] Documentation updated (CLAUDE.md)
- [x] No dependency on global registry

## Related Files

**Core Implementation:**
- `mesh/parsers/react_flow.py` - ReactFlow parser with inline tools
- `mesh/nodes/agent.py` - AgentNode (no changes needed)

**Examples:**
- `examples/test_agent_with_dynamic_tools.py` - Programmatic API
- `examples/test_reactflow_agent_with_inline_tools.py` - Declarative API

**Documentation:**
- `CLAUDE.md` - Section 8: Dynamic Tools Integration
- `docs/DYNAMIC_TOOLS_INTEGRATION.md` - This file

**Vel SDK Reference:**
- `/Users/richard.s/vel/features/summaries/DYNAMIC_TOOLS_UPGRADE.md` - Vel upgrade details
- `/Users/richard.s/vel/vel/agent.py` - Agent constructor (lines 29-186)
- `/Users/richard.s/vel/vel/tools/registry.py` - ToolSpec.from_function()

## Summary

Mesh is now fully compatible with Vel v0.3.0+ dynamic tools upgrade. Tools work seamlessly in both programmatic and declarative APIs, with no global registration required. The implementation follows best practices and provides a clean, type-safe developer experience.
