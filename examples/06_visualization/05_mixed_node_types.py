"""Example: Mixed Node Types Visualization

This example demonstrates all different node types with distinct colors:
- Agent nodes (RED) - for intelligent agent processing
- LLM nodes (BLUE) - for direct LLM calls (simulated)
- Tool nodes (GREEN) - for utility functions
- Condition nodes (YELLOW) - for decision points

Shows the full color palette of the visualization system.

Note: This example uses pre-built node instances for visualization purposes.
"""

from mesh import StateGraph
from mesh.nodes.condition import Condition
from mesh.nodes.agent import AgentNode
from mesh.nodes.llm import LLMNode
from mesh.nodes.tool import ToolNode


class MockVelAgent:
    """Mock Vel agent for visualization demo."""
    def __init__(self, name):
        self.agent_id = name
        self.__class__.__module__ = "vel.agent"

    async def run_stream(self, message, context=None):
        """Mock streaming method."""
        yield {"type": "token", "content": "mock response"}


def preprocess_input(input: dict) -> dict:
    """Tool: Preprocess the input data."""
    text = input.get("text", "")
    return {"processed_text": text.lower(), "word_count": len(text.split())}


def classify_complexity(input: dict) -> dict:
    """Tool: Classify input complexity."""
    word_count = input.get("word_count", 0)
    complexity = "complex" if word_count > 10 else "simple"
    return {"complexity": complexity, "processed_text": input.get("processed_text", "")}


def postprocess_output(input: dict) -> dict:
    """Tool: Format final output."""
    content = input.get("content", "")
    return {"final_output": f"=== RESULT ===\n{content}\n=============="}


def main():
    print("=" * 70)
    print("Mixed Node Types Visualization Example")
    print("=" * 70)
    print()

    # Build a workflow with mixed node types
    graph = StateGraph()

    # 1. Tool node (GREEN) - preprocessing
    tool_preprocessor = ToolNode(id="preprocessor", tool_fn=preprocess_input)
    graph.add_node("preprocessor", tool_preprocessor)

    # 2. Tool node (GREEN) - classification
    tool_classifier = ToolNode(id="classifier", tool_fn=classify_complexity)
    graph.add_node("classifier", tool_classifier)

    # 3. Condition node (YELLOW) - routing based on complexity
    conditions = [
        Condition(
            name="simple",
            predicate=lambda output: output.get("complexity") == "simple",
            target_node="simple_llm"
        ),
        Condition(
            name="complex",
            predicate=lambda output: output.get("complexity") == "complex",
            target_node="complex_agent"
        )
    ]
    graph.add_node("complexity_router", conditions, node_type="condition")

    # 4a. LLM node (BLUE) - for simple cases (mock, won't actually execute)
    # Creating node directly to avoid API key requirement for visualization
    from mesh.nodes.llm import LLMNode
    mock_llm = LLMNode.__new__(LLMNode)
    mock_llm.id = "simple_llm"
    mock_llm.config = {}
    graph._nodes["simple_llm"] = mock_llm

    # 4b. Agent node (RED) - for complex cases
    mock_agent = MockVelAgent("complex_processor")
    agent_node = AgentNode(id="complex_agent", agent=mock_agent)
    graph.add_node("complex_agent", agent_node)

    # 5. Tool node (GREEN) - post-processing
    tool_postprocessor = ToolNode(id="postprocessor", tool_fn=postprocess_output)
    graph.add_node("postprocessor", tool_postprocessor)

    # Create edges
    graph.add_edge("preprocessor", "classifier")
    graph.add_edge("classifier", "complexity_router")
    graph.add_edge("complexity_router", "simple_llm")
    graph.add_edge("complexity_router", "complex_agent")
    graph.add_edge("simple_llm", "postprocessor")
    graph.add_edge("complex_agent", "postprocessor")

    # Set entry point
    graph.set_entry_point("preprocessor")

    print("Step 1: Generate Mermaid Code")
    print("-" * 70)
    print()

    # Generate Mermaid code
    mermaid_code = graph.mermaid_code(title="Mixed Node Types")
    print(mermaid_code)
    print()

    print("=" * 70)
    print("Step 2: Save Visualization as PNG")
    print("-" * 70)
    print()

    # Save as PNG
    output_path = graph.save_visualization(
        title="mixed_node_types",
        image_format="png"
    )

    print(f"✓ Visualization saved to: {output_path}")
    print()

    print("=" * 70)
    print("Graph Pattern: Mixed Node Types")
    print("=" * 70)
    print()
    print("Flow:")
    print("  START → preprocessor (GREEN tool)")
    print("       → classifier (GREEN tool)")
    print("       → complexity_router (YELLOW condition)")
    print("          ├─→ simple_llm (BLUE llm)")
    print("          └─→ complex_agent (RED agent)")
    print("       → postprocessor (GREEN tool)")
    print("       → END")
    print()
    print("Color Legend:")
    print("  🔴 RED (Agent):     Intelligent agent processing")
    print("  🔵 BLUE (LLM):      Direct LLM calls")
    print("  🟢 GREEN (Tool):    Utility functions")
    print("  🟡 YELLOW (Condition): Decision points")
    print()
    print("This visualization shows all node type colors at once,")
    print("making it easy to identify the role of each component.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
