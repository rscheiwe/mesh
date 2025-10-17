"""Example: Graph Visualization with Mermaid Diagrams

This example demonstrates Mesh's Mermaid visualization capabilities.
It shows how to:
1. Generate Mermaid flowchart code from a graph
2. Save the graph as a PNG diagram
3. Visualize different node types and loop edges
"""

from mesh import StateGraph


def check_divisible_by_5(input: dict) -> dict:
    """Check if value is divisible by 5."""
    value = input.get("value", 0)
    is_divisible = (value % 5) == 0
    return {"value": value, "divisible": is_divisible}


def increment(input: dict) -> dict:
    """Increment the value by 1."""
    value = input.get("value", 0)
    new_value = value + 1
    return {"value": new_value}


def main():
    print("=" * 70)
    print("Graph Visualization Example: Mermaid Diagrams")
    print("=" * 70)
    print()

    # Build a cyclic graph (loop until divisible by 5)
    graph = StateGraph()

    # Add nodes
    graph.add_node("check", check_divisible_by_5, node_type="tool")
    graph.add_node("increment", increment, node_type="tool")

    # Add edges (no need to manually add START edge, compile() does this)
    # Edge: check -> increment (when NOT divisible)
    graph.add_edge(
        "check",
        "increment",
    )

    # Loop edge: increment -> check (creates cycle)
    graph.add_edge(
        "increment",
        "check",
        is_loop_edge=True,
        max_iterations=20,
    )

    # Set entry point (compile() will connect START to this node)
    graph.set_entry_point("check")

    print("Step 1: Generate Mermaid Code")
    print("-" * 70)
    print()

    # Generate Mermaid code
    mermaid_code = graph.mermaid_code(title="Divisible By 5 Graph")
    print(mermaid_code)
    print()

    print("=" * 70)
    print("Step 2: Save Visualization as PNG")
    print("-" * 70)
    print()

    # Save as PNG
    output_path = graph.save_visualization(
        title="fives_graph",
        image_format="png"
    )

    print(f"✓ Visualization saved to: {output_path}")
    print()

    print("=" * 70)
    print("Step 3: Alternative - Save with Custom Path")
    print("-" * 70)
    print()

    # Save with custom path
    custom_path = "/Users/richard.s/mesh/examples/my_graph.png"
    output_path_2 = graph.save_visualization(
        output_path=custom_path,
        title="Custom Fives Graph",
        image_format="png"
    )

    print(f"✓ Custom visualization saved to: {output_path_2}")
    print()

    print("=" * 70)
    print("Visualization Features")
    print("=" * 70)
    print()
    print("Node Type Styling:")
    print("  - START: Dark gray filled circle")
    print("  - END: Red double circle (target)")
    print("  - Agent: Red rounded rectangle")
    print("  - LLM: Blue rounded rectangle")
    print("  - Tool: Green rounded rectangle")
    print("  - Condition: Yellow diamond")
    print("  - Loop: Purple rounded rectangle")
    print()
    print("Edge Styling:")
    print("  - Regular edges: Simple arrows")
    print("  - Loop edges: Labeled with 'loop' or max iterations")
    print()
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
