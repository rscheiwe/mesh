# 01 - Single Node Basics

This folder contains the simplest examples of Mesh: single-node graphs that demonstrate the core functionality.

## Examples

### `llm_node.py`
**Difficulty:** ⭐ Beginner

A basic LLM node that makes a direct call to an LLM provider (OpenAI GPT-4) without using an agent framework.

**What you'll learn:**
- How to create a simple Mesh graph
- Using `LLMNode` for direct LLM calls
- Token-by-token streaming
- Basic event handling

**Run:**
```bash
python 01_single_node_basics/llm_node.py
```

### `vel_agent_node.py`
**Difficulty:** ⭐ Beginner

A single Vel agent node demonstrating Mesh integration with the Vel agent framework.

**What you'll learn:**
- Using Vel agents in Mesh graphs
- Native Vel event handling
- Multi-turn conversations with state persistence

**Prerequisites:**
```bash
pip install 'mesh[vel]'
```

**Run:**
```bash
python 01_single_node_basics/vel_agent_node.py
```

### `openai_agent_node.py`
**Difficulty:** ⭐ Beginner

A single OpenAI Agents SDK node demonstrating integration with OpenAI's agent framework.

**What you'll learn:**
- Using OpenAI Agents SDK in Mesh
- Agent auto-detection
- Vel event translation (automatic)

**Prerequisites:**
```bash
pip install openai-agents
```

**Run:**
```bash
python 01_single_node_basics/openai_agent_node.py
```

## Key Concepts

- **Single Node Graphs:** The simplest graph structure with just one node
- **Agent vs LLM Nodes:** Agents have tools and multi-step capabilities, LLMs are simpler direct calls
- **Streaming:** All examples demonstrate real-time token streaming
- **Event Types:** NODE_START, TOKEN, NODE_COMPLETE, EXECUTION_COMPLETE

## Next Steps

Once you're comfortable with single nodes, move on to:
- **02_multi_node_graphs:** Connect multiple nodes together
- **03_advanced_patterns:** Loops, cycles, and conditional branching
