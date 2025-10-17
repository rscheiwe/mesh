"""Example: Multi-Stage Workflow Visualization

This example demonstrates a complex workflow with multiple processing stages:
- Input processing
- Content generation
- Enhancement
- Validation
- Final formatting

Shows how multi-stage pipelines are visualized clearly.
"""

from mesh import StateGraph


def extract_keywords(input: dict) -> dict:
    """Extract keywords from text."""
    text = input.get("text", "")
    keywords = ["workflow", "agent", "visualization"]
    return {"text": text, "keywords": keywords}


def generate_content(input: dict) -> dict:
    """Generate content based on keywords."""
    keywords = input.get("keywords", [])
    content = f"Generated content using: {', '.join(keywords)}"
    return {"content": content}


def enhance_content(input: dict) -> dict:
    """Enhance and polish the content."""
    content = input.get("content", "")
    enhanced = f"Enhanced: {content}"
    return {"content": enhanced}


def validate_output(input: dict) -> dict:
    """Validate the generated output."""
    content = input.get("content", "")
    is_valid = len(content) > 10
    return {"content": content, "valid": is_valid}


def format_final(input: dict) -> dict:
    """Format the validated content."""
    content = input.get("content", "")
    return {"final": f"=== FINAL OUTPUT ===\n{content}\n==================="}


def main():
    print("=" * 70)
    print("Multi-Stage Workflow Visualization Example")
    print("=" * 70)
    print()

    # Build a complex multi-node workflow
    graph = StateGraph()

    # Stage 1: Input processing
    graph.add_node("input_processor", extract_keywords, node_type="tool")

    # Stage 2: Content generation
    graph.add_node("content_generator", generate_content, node_type="tool")

    # Stage 3: Enhancement
    graph.add_node("content_enhancer", enhance_content, node_type="tool")

    # Stage 4: Validation
    graph.add_node("validator", validate_output, node_type="tool")

    # Stage 5: Final formatting
    graph.add_node("formatter", format_final, node_type="tool")

    # Create workflow edges
    graph.add_edge("input_processor", "content_generator")
    graph.add_edge("content_generator", "content_enhancer")
    graph.add_edge("content_enhancer", "validator")
    graph.add_edge("validator", "formatter")

    # Set entry point
    graph.set_entry_point("input_processor")

    print("Step 1: Generate Mermaid Code")
    print("-" * 70)
    print()

    # Generate Mermaid code
    mermaid_code = graph.mermaid_code(title="Multi-Agent Workflow")
    print(mermaid_code)
    print()

    print("=" * 70)
    print("Step 2: Save Visualization as PNG")
    print("-" * 70)
    print()

    # Save as PNG
    output_path = graph.save_visualization(
        title="multi_agent_workflow",
        image_format="png"
    )

    print(f"✓ Visualization saved to: {output_path}")
    print()

    print("=" * 70)
    print("Graph Pattern: Multi-Stage Pipeline")
    print("=" * 70)
    print()
    print("Flow:")
    print("  START → input_processor")
    print("       → content_generator")
    print("       → content_enhancer")
    print("       → validator")
    print("       → formatter")
    print("       → END")
    print()
    print("This pattern shows a multi-stage pipeline with 5 processing stages.")
    print("Each green node represents a distinct processing step in the workflow.")
    print()
    print("The visualization clearly shows the flow and node type distribution.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
