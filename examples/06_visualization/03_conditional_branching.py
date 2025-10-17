"""Example: Conditional Branching Visualization

This example demonstrates conditional branching in a graph:
- Condition node evaluates input
- Routes to different paths based on condition
- Shows diamond-shaped condition nodes in visualization
"""

from mesh import StateGraph
from mesh.nodes.condition import Condition


def analyze_sentiment(input: dict) -> dict:
    """Analyze sentiment of the input."""
    text = input.get("text", "")
    # Simple sentiment logic for demo
    sentiment = "positive" if "good" in text.lower() or "great" in text.lower() else "negative"
    return {"text": text, "sentiment": sentiment}


def handle_positive(input: dict) -> dict:
    """Handle positive sentiment."""
    return {"response": "Thank you for the positive feedback!"}


def handle_negative(input: dict) -> dict:
    """Handle negative sentiment."""
    return {"response": "We're sorry to hear that. How can we improve?"}


def main():
    print("=" * 70)
    print("Conditional Branching Visualization Example")
    print("=" * 70)
    print()

    # Build a branching workflow
    graph = StateGraph()

    # Add nodes
    graph.add_node("sentiment_analyzer", analyze_sentiment, node_type="tool")

    # Add condition node with explicit conditions
    conditions = [
        Condition(
            name="positive",
            predicate=lambda output: output.get("sentiment") == "positive",
            target_node="positive_handler"
        ),
        Condition(
            name="negative",
            predicate=lambda output: output.get("sentiment") == "negative",
            target_node="negative_handler"
        )
    ]

    graph.add_node("sentiment_condition", conditions, node_type="condition")
    graph.add_node("positive_handler", handle_positive, node_type="tool")
    graph.add_node("negative_handler", handle_negative, node_type="tool")

    # Create branching flow
    graph.add_edge("sentiment_analyzer", "sentiment_condition")
    graph.add_edge("sentiment_condition", "positive_handler")
    graph.add_edge("sentiment_condition", "negative_handler")

    # Set entry point
    graph.set_entry_point("sentiment_analyzer")

    print("Step 1: Generate Mermaid Code")
    print("-" * 70)
    print()

    # Generate Mermaid code
    mermaid_code = graph.mermaid_code(title="Conditional Branching")
    print(mermaid_code)
    print()

    print("=" * 70)
    print("Step 2: Save Visualization as PNG")
    print("-" * 70)
    print()

    # Save as PNG
    output_path = graph.save_visualization(
        title="conditional_branching",
        image_format="png"
    )

    print(f"✓ Visualization saved to: {output_path}")
    print()

    print("=" * 70)
    print("Graph Pattern: Conditional Branching")
    print("=" * 70)
    print()
    print("Flow:")
    print("  START → sentiment_analyzer")
    print("         → sentiment_condition (decision point)")
    print("            ├─→ positive_handler → END")
    print("            └─→ negative_handler → END")
    print()
    print("The condition node (yellow diamond) evaluates the sentiment and")
    print("routes to different handlers based on the result.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
