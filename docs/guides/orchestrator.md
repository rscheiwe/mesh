---
layout: default
title: Orchestrator & Multi-Agent Delegation
parent: Guides
nav_order: 5
---

# Orchestrator & Multi-Agent Delegation

The **OrchestratorNode** enables LLM-driven delegation to sub-agents at runtime. Instead of static graph edges determining execution order, an orchestrator agent dynamically decides which sub-agents to call based on the task at hand.

## Overview

### The Problem

Traditional graph-based workflows have static execution paths. If you connect Agent A → Agent B → Agent C, they always execute in that order. But many real-world scenarios require dynamic delegation:

- "Research this topic" → calls Research Agent
- "Analyze these numbers" → calls Analysis Agent
- "Write some code" → calls Coding Agent
- Complex tasks may require multiple agents in varying order

### The Solution: Orchestrator Pattern

The OrchestratorNode creates an orchestrator agent that:
1. Receives the user's task
2. Discovers sub-agents from connected AgentFlowNodes (via graph edges)
3. Creates callable tools from each sub-agent
4. Decides which sub-agent(s) to invoke based on descriptions
5. Synthesizes results into a coherent response

## Visual Canvas Pattern

In the React Flow canvas, connect AgentFlowNodes as children of the OrchestratorNode:

```
                        ┌─→ AgentFlowNode (Researcher)
StartNode → Orchestrator ─┼─→ AgentFlowNode (Analyst)
                        └─→ AgentFlowNode (Writer)
```

The orchestrator automatically discovers connected AgentFlowNodes as sub-agents. **No dropdown selection needed** - just draw the edges!

### How It Works

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│         Orchestrator Node               │
│  ┌─────────────────────────────────┐    │
│  │   Orchestrator LLM (e.g. GPT-4) │    │
│  │   "Which agent should handle?"  │    │
│  └─────────────────────────────────┘    │
│              │                          │
│    ┌─────────┼─────────┐   Discovered   │
│    ▼         ▼         ▼   from edges   │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │Flow A│ │Flow B│ │Flow C│  (ToolSpec)  │
│ │(Mesh)│ │(Mesh)│ │(Mesh)│              │
│ └──────┘ └──────┘ └──────┘              │
│              │                          │
│              ▼                          │
│       Synthesize Results                │
└─────────────────────────────────────────┘
    │
    ▼
Final Response
```

## Key Design Decision: Edge-Based Discovery

Sub-agents are discovered from **graph edges**, not from a configuration dropdown. This provides:

1. **Visual Clarity**: You can see the agent team structure directly in the canvas
2. **Composition**: Each sub-agent is a saved AgentFlow that can be any Mesh graph:
   - `START → Agent → END` (simple agent flow)
   - `START → Tool → LLM → END` (tool + reasoning)
   - `START → RAG → Agent → END` (retrieval-augmented)
   - Any composition of Mesh nodes
3. **Reusability**: The same AgentFlow can be used in multiple orchestrators

## Configuration

### OrchestratorNode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | string | required | Node identifier |
| `provider` | string | "openai" | LLM provider (openai, anthropic, gemini) |
| `model_name` | string | "gpt-4o" | Model for orchestration decisions |
| `instruction` | string | "" | Instructions for the orchestrator LLM |
| `temperature` | float | 0.3 | Sampling temperature (lower = more deterministic) |
| `result_mode` | string | "synthesize" | How to handle outputs |
| `max_iterations` | int | 5 | Maximum sub-agent calls |
| `show_sub_agent_events` | bool | True | Stream events from sub-agents |
| `event_mode` | string | "full" | Event emission mode |

### Sub-Agent Configuration (AgentFlowNode)

Each AgentFlowNode connected to the orchestrator should have:

| Field | Description |
|-------|-------------|
| `flowUuid` | UUID of the saved agent flow |
| `flowName` | Display name (becomes tool name for the orchestrator LLM) |
| `flowDescription` | Description used by orchestrator LLM to decide when to call this agent |

The **description is critical** - it's what the orchestrator LLM uses to decide when to call each sub-agent.

### Result Modes

| Mode | Behavior |
|------|----------|
| `synthesize` | Orchestrator combines all sub-agent outputs into a coherent response |
| `stream_through` | Stream sub-agent outputs directly to user as they complete |
| `raw` | Return structured output with `orchestrator_response` and `sub_agent_outputs` |

## Usage Examples

### Example 1: Research & Analysis Team

In the canvas:
1. Add an **OrchestratorNode** with instruction:
```
You are a research team coordinator.

Analyze the user's request and delegate to the appropriate specialist:
- For gathering information, use the researcher agent
- For analyzing data or numbers, use the analyst agent
- For writing summaries, use the writer agent

Complex requests may require multiple agents. Synthesize their outputs
into a coherent final response.
```

2. Connect three **AgentFlowNodes** as children:
   - "Researcher" - description: "Gathers comprehensive information on topics through web search and document analysis"
   - "Analyst" - description: "Analyzes data, identifies patterns, and provides statistical insights"
   - "Writer" - description: "Writes clear, well-structured summaries and reports"

3. Connect `StartNode → Orchestrator → [Researcher, Analyst, Writer]`

### Example 2: Customer Support Router

```
StartNode → Orchestrator → Technical Support AgentFlow
                        → Billing Support AgentFlow
                        → General Support AgentFlow
```

Orchestrator instruction:
```
You are a customer support router.

Analyze the customer's issue and route to the appropriate specialist:
- Technical issues → Technical Support Agent
- Billing questions → Billing Agent
- General inquiries → General Support Agent

Always acknowledge the customer's concern before delegating.
```

## React Flow UI Integration

In the mesh-app UI, the Orchestrator node appears in the Agents category with a distinct fuchsia color.

### Node Definition

```typescript
{
  type: "orchestratorAgentflow",
  name: "orchestrator",
  label: "Orchestrator",
  description: "LLM-driven delegation to sub-agents",
  icon: "Network",
  category: "Agents",
  color: "#c026d3",  // fuchsia-600
  inputs: [
    { name: "provider", type: "options", ... },
    { name: "modelName", type: "string", ... },
    { name: "instruction", type: "string", rows: 6, ... },
    { name: "resultMode", type: "options", ... },
    // ... other inputs
  ],
  outputs: ["output"]
}
```

### Connecting Sub-Agents

Simply drag edges from the Orchestrator node to AgentFlowNodes. The orchestrator will automatically discover all connected AgentFlowNodes at runtime and create tools from them.

## How It Works Internally

### 1. Parsing Phase

When the ReactFlowParser encounters an orchestrator with connected AgentFlowNodes:
- It identifies which AgentFlowNodes are children of orchestrators
- Those nodes are NOT expanded inline (normal AgentFlowNodes get expanded)
- Instead, they become `SubAgentNode` placeholders with flow metadata

### 2. Sub-Agent Discovery

When the orchestrator executes, it discovers sub-agents:

```python
def _discover_sub_agents(self) -> List[Dict[str, Any]]:
    """Discover sub-agents from connected SubAgentNodes."""
    sub_agents = []
    child_ids = self._graph.get_children(self.id)

    for child_id in child_ids:
        child_node = self._graph.get_node(child_id)
        if isinstance(child_node, SubAgentNode):
            info = child_node.get_info()
            sub_agents.append(info.to_dict())

    return sub_agents
```

### 3. Tool Creation

Each discovered sub-agent becomes a ToolSpec:

```python
tool = ToolSpec(
    name=tool_name,  # sanitized from flowName
    description=description,  # from flowDescription
    input_schema={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": f"Task or query to delegate to {name}"
            }
        },
        "required": ["message"],
    },
    handler=handler,  # Executes the flow via Mesh Executor
)
```

### 4. Execution

The orchestrator streams its response, calling sub-agents as needed:

```python
async for event in orchestrator.run_stream({"message": message}, session_id):
    if event["type"] == "text-delta":
        # Orchestrator is reasoning/responding
        full_response += event["delta"]

    elif event["type"] == "tool-input-available":
        # Orchestrator is calling a sub-agent

    elif event["type"] == "tool-output-available":
        # Sub-agent completed, result available
        sub_agent_outputs.append(event["output"])
```

## Event Streaming

### Event Flow

```
1. data-mesh-node-start {node_id: "orchestrator_0", node_type: "orchestrator"}
2. text-delta {delta: "I'll analyze your request..."}
3. tool-input-available {toolName: "researcher", input: {...}}
4. [Sub-agent events if showSubAgentEvents=true]
   - data-mesh-node-start {node_id: "orchestrator_0.START"}
   - text-delta {delta: "Researching..."}
   - data-mesh-node-complete {node_id: "orchestrator_0.agent_0"}
5. tool-output-available {output: {response: "..."}}
6. text-delta {delta: "Based on the research..."}
7. data-mesh-node-complete {node_id: "orchestrator_0"}
```

### Event Prefixing

Sub-agent events are prefixed with the orchestrator's node ID:
- `orchestrator_0.START`
- `orchestrator_0.agent_0`
- `orchestrator_0.llm_0`

This allows the UI to distinguish which events come from which sub-agent.

## Comparison: Orchestrator vs Sequential Flows

| Aspect | Sequential Agent Flows | Orchestrator Node |
|--------|------------------------|-------------------|
| **Control Flow** | Static graph edges | LLM decides at runtime |
| **Execution** | All connected flows execute | Only relevant agents execute |
| **Selection** | Designer chooses at build-time | LLM chooses based on task |
| **Parallelism** | Defined by graph structure | LLM can call multiple or iterate |
| **Results** | Each node outputs independently | Orchestrator synthesizes |
| **Sub-Agent Config** | ~~Dropdown selection~~ | **Graph edges** |

## Best Practices

### 1. Write Clear Descriptions

The orchestrator LLM uses descriptions to decide which sub-agent to call. Be specific:

```
# Good
"Performs deep research on topics using web search, academic papers, and document analysis"

# Too vague
"Research agent"
```

### 2. Provide Clear Instructions

Tell the orchestrator when to use each sub-agent:

```
For gathering information → use researcher
For analyzing data → use analyst
For complex tasks → use multiple agents in sequence
Always synthesize outputs into a coherent response.
```

### 3. Set Appropriate Limits

Use `max_iterations` to prevent runaway delegation:

```python
max_iterations=5  # Reasonable limit for most tasks
```

### 4. Choose the Right Result Mode

- `synthesize`: Best for most use cases—orchestrator provides coherent response
- `stream_through`: Good for interactive UIs where you want immediate feedback
- `raw`: Best for programmatic access to all outputs

### 5. Consider Event Visibility

- `show_sub_agent_events=True`: Shows full execution for transparency
- `show_sub_agent_events=False`: Cleaner UX, sub-agents as "black boxes"

## Inspiration

This pattern is inspired by:
- [CloudShip Station](https://github.com/cloudshipai/station) - Hierarchical agent delegation for incident response
- OpenAI's Swarm pattern - Agent handoffs and tool-based delegation
- Vel's `Agent.as_tool()` - Exposing agents as callable tools

## Related

- [Nodes Reference](../concepts/nodes.md) - All node types
- [Event Modes](../concepts/event-modes.md) - Controlling event streaming
- [Variables](../concepts/variables.md) - Using `{{node.output}}` in instructions
