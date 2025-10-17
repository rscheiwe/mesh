# 02 - Multi-Node Graphs

This folder contains examples of connecting multiple nodes together to create more complex workflows.

## Examples

### `agent_chain.py`
**Difficulty:** ⭐⭐ Intermediate

Demonstrates a sequential chain of two LLM nodes:
1. **Analyzer** - Extracts key information from user input
2. **Responder** - Crafts a response based on the analysis

**What you'll learn:**
- Connecting multiple nodes in sequence
- Passing output from one node to another
- Variable resolution syntax `{{node_id.content}}`
- Node execution order

**Run:**
```bash
python 02_multi_node_graphs/agent_chain.py
```

### `tool_to_llm.py`
**Difficulty:** ⭐⭐ Intermediate

Demonstrates Tool node → LLM node workflow:
1. **Tool Node** - Executes Python function (calculates statistics)
2. **LLM Node** - Interprets tool output in natural language

**What you'll learn:**
- Using `ToolNode` to execute Python functions
- Accessing tool outputs with `{{tool_id.output}}`
- Combining computation with language understanding
- Tool function patterns

**Run:**
```bash
python 02_multi_node_graphs/tool_to_llm.py
```

## Key Concepts

### Variable Resolution

Access output from previous nodes using the `{{node_id}}` syntax:

```python
system_prompt="Based on this analysis: {{analyzer.content}}"
```

**Available variables:**
- `{{node_id}}` - Full output of a node
- `{{node_id.content}}` - Text content from agent/LLM nodes
- `{{node_id.output}}` - Structured output from tool nodes
- `{{$question}}` - Original user input
- `{{$vars.key}}` - Global variables

### Node Execution Order

Mesh uses **queue-based execution** with dependency tracking:
1. Nodes execute when all dependencies are met
2. Output flows to downstream nodes automatically
3. Multiple paths can execute in parallel

### Edge Connections

```python
graph.add_edge("source", "target")
```

This creates a directed edge from `source` → `target`. The target node receives the source's output as input.

## Next Steps

Ready for more advanced patterns?
- **03_advanced_patterns:** Loops, cycles, and conditional branching
- **04_event_translation:** Understanding event translation and step tracking
