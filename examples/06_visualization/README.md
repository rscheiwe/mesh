# Mesh Visualization Examples

This directory contains examples demonstrating Mesh's Mermaid diagram visualization capabilities.

> **Note**: The visualization API design is inspired by [Pydantic AI's graph visualization](https://ai.pydantic.dev/graph/), adapted for Mesh's unique node types and workflow patterns.

## Examples

### 01_cyclic_graph.py
**Pattern**: Controlled Loop/Cycle

Demonstrates:
- Cyclic graph with loop edges
- Loop condition visualization
- Termination points
- Tool nodes

**Graph**: A "divisible by 5" checker that loops until the condition is met.

```bash
python examples/visualization_examples/01_cyclic_graph.py
```

### 02_sequential_workflow.py
**Pattern**: Linear Pipeline

Demonstrates:
- Sequential workflow with no branching
- Multiple node types (LLM, Tool)
- Color-coded node styling
- Simple start-to-end flow

**Graph**: Analyzer → Processor → Formatter pipeline.

```bash
python examples/visualization_examples/02_sequential_workflow.py
```

### 03_conditional_branching.py
**Pattern**: Conditional Routing

Demonstrates:
- Condition nodes (diamond shape)
- Branching based on conditions
- Multiple execution paths
- Sentiment analysis routing

**Graph**: Sentiment analyzer that routes to different handlers.

```bash
python examples/visualization_examples/03_conditional_branching.py
```

### 04_multi_agent_workflow.py
**Pattern**: Multi-Stage Pipeline

Demonstrates:
- Complex multi-node workflows
- Sequential tool nodes
- Multi-stage processing
- Real-world workflow patterns

**Graph**: Input processing → Content generation → Enhancement → Validation → Formatting.

```bash
python examples/visualization_examples/04_multi_agent_workflow.py
```

### 05_mixed_node_types.py
**Pattern**: Multi-Type Workflow (COLOR SHOWCASE)

Demonstrates:
- **All node type colors in one graph**
- Agent nodes (RED)
- LLM nodes (BLUE)
- Tool nodes (GREEN)
- Condition nodes (YELLOW)
- Complete color palette visualization

**Graph**: Tool preprocessing → Condition router → [LLM or Agent] → Tool postprocessing.

```bash
python examples/visualization_examples/05_mixed_node_types.py
```

**This example is perfect for seeing all node type colors at once!**

## Node Type Colors

The visualizations use different colors for each node type:

- **START**: Dark gray filled circle
- **END**: Red circle (termination point)
- **Agent**: Red rounded rectangle
- **LLM**: Blue rounded rectangle
- **Tool**: Green rounded rectangle
- **Condition**: Yellow diamond
- **Loop**: Purple rounded rectangle

## Output

All examples save their visualizations to `mesh/visualizations/` by default. Each visualization includes:

- Graph title
- Styled nodes (color-coded by type)
- Directional arrows
- Loop edge labels (for cyclic graphs)
- Clear start and end points

## Usage

Each example can be run independently and will:
1. Print the Mermaid flowchart code
2. Save a PNG visualization to `mesh/visualizations/`
3. Explain the graph pattern and flow

You can modify the examples to visualize your own graph structures!
