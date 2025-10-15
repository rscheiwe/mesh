"""Example: Sequential Workflow Visualization

This example demonstrates a simple sequential workflow with multiple node types:
- Tool nodes for processing steps
- Shows clear linear flow from start to end

Shows how different processing stages are visualized in a pipeline.
"""

from mesh import StateGraph


def analyze_text(input: dict) -> dict:
    """Analyze the input text."""
    text = input.get("text", "")
    return {"analyzed": f"Analysis of: {text}"}


def process_analysis(input: dict) -> dict:
    """Process the analysis results."""
    analyzed = input.get("analyzed", "")
    return {"processed": f"Processed: {analyzed}"}


def format_output(input: dict) -> dict:
    """Format the final output."""
    processed = input.get("processed", "")
    return {"formatted": f"=== RESULT ===\n{processed}\n============="}


def main():
    print("=" * 70)
    print("Sequential Workflow Visualization Example")
    print("=" * 70)
    print()

    # Build a sequential workflow
    graph = StateGraph()

    # Add nodes - all tool nodes for simplicity
    graph.add_node("analyzer", analyze_text, node_type="tool")
    graph.add_node("processor", process_analysis, node_type="tool")
    graph.add_node("formatter", format_output, node_type="tool")

    # Create sequential flow
    graph.add_edge("analyzer", "processor")
    graph.add_edge("processor", "formatter")

    # Set entry point
    graph.set_entry_point("analyzer")

    print("Step 1: Generate Mermaid Code")
    print("-" * 70)
    print()

    # Generate Mermaid code
    mermaid_code = graph.mermaid_code(title="Sequential Workflow")
    print(mermaid_code)
    print()

    print("=" * 70)
    print("Step 2: Save Visualization as PNG")
    print("-" * 70)
    print()

    # Save as PNG
    output_path = graph.save_visualization(
        title="sequential_workflow",
        image_format="png"
    )

    print(f"✓ Visualization saved to: {output_path}")
    print()

    print("=" * 70)
    print("Graph Pattern: Sequential")
    print("=" * 70)
    print()
    print("Flow: START → analyzer (Tool) → processor (Tool) → formatter (Tool) → END")
    print()
    print("This pattern shows a linear pipeline with no branching or loops.")
    print("Tool nodes are styled in green, showing a clear processing pipeline.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
