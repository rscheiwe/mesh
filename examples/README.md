# Mesh Examples

This directory contains progressively complex examples of using Mesh, organized from beginner to advanced.

## Getting Started

### Prerequisites

```bash
# Basic installation
pip install mesh-agent

# With all optional dependencies
pip install 'mesh[vel,agents,server]'

# Set up environment
cp .env.example .env
# Add your API keys to .env
```

### Quick Start

Start with the simplest example:
```bash
python 01_single_node_basics/llm_node.py
```

## Examples Organization

Examples are organized into numbered folders that progress from simple to advanced:

### 01 - Single Node Basics ⭐
**Difficulty:** Beginner

The fundamentals: single-node graphs demonstrating core Mesh functionality.

**Examples:**
- `llm_node.py` - Direct LLM call without agent framework
- `vel_agent_node.py` - Single Vel agent
- `openai_agent_node.py` - Single OpenAI agent

**Learn:** Graph basics, streaming, event handling

[→ Go to 01_single_node_basics](01_single_node_basics/)

---

### 02 - Multi-Node Graphs ⭐⭐
**Difficulty:** Intermediate

Connect nodes together to create workflows.

**Examples:**
- `agent_chain.py` - Agent → Agent sequential chain
- `tool_to_llm.py` - Tool → LLM workflow

**Learn:** Node connections, variable resolution, execution order

[→ Go to 02_multi_node_graphs](02_multi_node_graphs/)

---

### 03 - Advanced Patterns ⭐⭐⭐
**Difficulty:** Advanced

Loops, cycles, and conditional execution.

**Examples:**
- `cyclic_graph_example.py` - Controlled cycles
- `cyclic_graph_max_iterations.py` - Loop safety limits
- `loop_condition_explained.py` - Loop condition deep dive
- `loop_timing_diagram.py` - Loop execution timeline

**Learn:** Loop edges, conditions, iteration limits, safety controls

[→ Go to 03_advanced_patterns](03_advanced_patterns/)

---

### 04 - Event Translation ⭐⭐⭐
**Difficulty:** Advanced

Understanding event translation and step-level tracking.

**Examples:**
- `event_translation_comparison.py` - Native vs translated events
- `EVENT_TRANSLATION.md` - Comprehensive documentation

**Learn:** Event systems, translation architecture, step tracking, TranslatorOrchestrator

[→ Go to 04_event_translation](04_event_translation/)

---

### 05 - Integrations ⭐⭐⭐
**Difficulty:** Advanced

Web frameworks, parsers, and external systems.

**Examples:**
- `fastapi_server.py` - Production FastAPI server with SSE
- `react_flow_parse.py` - JSON graph specification
- Integration guides for Vel and OpenAI

**Learn:** REST APIs, SSE streaming, JSON parsers, state persistence

[→ Go to 05_integrations](05_integrations/)

---

### 06 - Visualization ⭐⭐
**Difficulty:** Intermediate

Graph visualization using Mermaid diagrams.

**Examples:**
- `graph_visualization_example.py` - Basic visualization
- `01-05_*.py` - Various graph pattern visualizations

**Learn:** Graph visualization, Mermaid syntax, debugging visually

[→ Go to 06_visualization](06_visualization/)

---

## Learning Path

### Beginner Path (⭐)
1. Start with `01_single_node_basics/llm_node.py`
2. Try `01_single_node_basics/vel_agent_node.py` or `openai_agent_node.py`
3. Understand single-node execution
4. Learn basic event handling

### Intermediate Path (⭐⭐)
1. Complete Beginner Path
2. Work through `02_multi_node_graphs/`
3. Learn variable resolution and node connections
4. Explore `06_visualization/` to see graph structures

### Advanced Path (⭐⭐⭐)
1. Complete Intermediate Path
2. Study `03_advanced_patterns/` for loops and cycles
3. Dive into `04_event_translation/` for event systems
4. Build production apps with `05_integrations/`

## Common Patterns

### Single Node
```python
graph = StateGraph()
graph.add_node("node", agent, node_type="agent")
graph.add_edge("START", "node")
graph.set_entry_point("node")
```

### Sequential Chain
```python
graph.add_node("node1", agent1, node_type="agent")
graph.add_node("node2", agent2, node_type="agent")
graph.add_edge("START", "node1")
graph.add_edge("node1", "node2")
```

### Tool Integration
```python
graph.add_node("tool", None, node_type="tool", function=my_function)
graph.add_node("llm", None, node_type="llm",
    system_prompt="Process: {{tool.output}}")
graph.add_edge("tool", "llm")
```

### Controlled Loop
```python
graph.add_edge("check", "process")
graph.add_edge(
    "process",
    "check",
    is_loop_edge=True,
    loop_condition=lambda s, o: not o["done"],
    max_iterations=10
)
```

## Event Handling

All examples demonstrate event-driven streaming:

```python
async for event in executor.execute(input, context):
    if event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "node_complete":
        print(f"\n✓ {event.node_id} complete")
    elif event.type == "execution_complete":
        print("\n✓ Done!")
```

## Troubleshooting

### Import Errors
```bash
# Make sure mesh is installed
pip install mesh-agent

# Or install from local source
pip install -e .
```

### Missing Dependencies
```bash
# Install optional dependencies
pip install 'mesh[vel]'        # For Vel examples
pip install 'mesh[agents]'     # For OpenAI Agents examples
pip install 'mesh[server]'     # For FastAPI examples
pip install 'mesh[viz]'        # For visualization examples
```

### API Keys
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

### Import Path Issues
Make sure you're running from the repository root:
```bash
cd /path/to/mesh
python examples/01_single_node_basics/llm_node.py
```

## Contributing Examples

Want to add an example? Guidelines:
1. Place in the appropriate difficulty folder
2. Include docstring explaining what it demonstrates
3. Add entry to folder's README.md
4. Use clear variable names
5. Include error handling
6. Test with real API keys

## Resources

- **Documentation:** [Mesh Docs](https://github.com/rscheiwe/mesh)
- **Vel SDK:** [Vel Docs](https://rscheiwe.github.io/vel)
- **CLAUDE.md:** Comprehensive developer guide at repository root
- **Issues:** [GitHub Issues](https://github.com/rscheiwe/mesh/issues)

## Next Steps

1. Choose your starting point based on experience level
2. Work through examples in order within each folder
3. Experiment by modifying examples
4. Build your own graphs!

Happy coding! 🚀
