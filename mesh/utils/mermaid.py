"""Mermaid diagram visualization utilities for Mesh graphs.

This module provides functions to generate Mermaid flowchart diagrams from
ExecutionGraph instances and save them as images using the mermaid.ink API.

Inspired by Pydantic AI's graph visualization approach:
https://ai.pydantic.dev/graph/

The API design and mermaid.ink integration pattern follows Pydantic Graph's
excellent implementation, adapted for Mesh's node types and workflow patterns.
"""

import base64
import urllib.parse
from typing import Optional, TYPE_CHECKING
from pathlib import Path

import httpx

if TYPE_CHECKING:
    from mesh.core.graph import ExecutionGraph


def generate_mermaid_code(
    graph: "ExecutionGraph",
    title: Optional[str] = None,
    direction: str = "TD",
) -> str:
    """Generate Mermaid flowchart code from an ExecutionGraph.

    Args:
        graph: The ExecutionGraph to visualize
        title: Optional title to display above the diagram
        direction: Flowchart direction - "TD" (top-down) or "LR" (left-right)

    Returns:
        String containing Mermaid flowchart code

    Example:
        >>> mermaid_code = generate_mermaid_code(compiled_graph, title="My Graph")
        >>> print(mermaid_code)
    """
    lines = []

    # Add title if provided
    if title:
        lines.append(f"---")
        lines.append(f"title: {title}")
        lines.append(f"---")

    # Start flowchart
    lines.append(f"flowchart {direction}")

    # Check if we need to add a synthetic END node
    # In cyclic graphs, the END should come from nodes that are targets of loop edges
    # because those are the nodes that evaluate the exit condition
    ending_nodes = []

    # First, identify nodes that are targets of loop edges (decision points)
    loop_targets = set()
    for edge in graph.edges:
        if edge.is_loop_edge:
            loop_targets.add(edge.target)

    # For nodes that are loop targets, they can exit the loop, so add END
    for node_id in loop_targets:
        node_type = graph.nodes[node_id].__class__.__name__.replace("Node", "").lower()
        if node_type not in ["end", "start"]:
            ending_nodes.append(node_id)

    # If no loop targets, fall back to finding nodes with no forward edges
    if not ending_nodes:
        for node_id in graph.nodes.keys():
            node_type = graph.nodes[node_id].__class__.__name__.replace("Node", "").lower()
            if node_type == "end":
                continue
            has_non_loop_children = any(
                edge.source == node_id and not edge.is_loop_edge
                for edge in graph.edges
            )
            if not has_non_loop_children and node_type != "start":
                ending_nodes.append(node_id)

    # Add synthetic END node if there are ending nodes
    add_end_node = len(ending_nodes) > 0 and "END" not in graph.nodes

    # Generate node definitions with styling based on type
    for node_id, node in graph.nodes.items():
        node_type = node.__class__.__name__.replace("Node", "").lower()

        # Map node types to Mermaid shapes and labels
        if node_type == "start":
            # Filled circle for START
            lines.append(f"    {node_id}((({node_id})))")
        elif node_type == "end":
            # Double circle (target/bullseye) for END
            lines.append(f"    {node_id}((({node_id})))")
        elif node_type == "condition":
            # Diamond for condition nodes
            lines.append(f'    {node_id}{{{{{node_type}: {node_id}}}}}')
        else:
            # Rounded rectangle for agent, llm, tool, loop
            lines.append(f'    {node_id}[{node_type}: {node_id}]')

    # Add synthetic END node if needed (use circle-in-circle for target/bullseye effect)
    if add_end_node:
        lines.append(f"    END(((END)))")

    # Generate edges
    for edge in graph.edges:
        source = edge.source
        target = edge.target

        # Add arrow with optional label for loop edges
        if edge.is_loop_edge:
            # Show loop edges with special styling
            if edge.max_iterations:
                lines.append(f'    {source} -->|max: {edge.max_iterations}| {target}')
            else:
                lines.append(f'    {source} -->|loop| {target}')
        else:
            # Regular edge
            lines.append(f'    {source} --> {target}')

    # Add edges from ending nodes to synthetic END
    if add_end_node:
        for ending_node in ending_nodes:
            lines.append(f'    {ending_node} --> END')

    # Add styling with classDef
    lines.append("")
    lines.append("    %% Node type styling")
    lines.append("    classDef startNode fill:#2d3748,stroke:#2d3748,color:#fff")
    lines.append("    classDef endNode fill:#c53030,stroke:#c53030,color:#fff")
    lines.append("    classDef agentNode fill:#e53e3e,stroke:#c53030,color:#fff")
    lines.append("    classDef llmNode fill:#3182ce,stroke:#2c5282,color:#fff")
    lines.append("    classDef toolNode fill:#38a169,stroke:#2f855a,color:#fff")
    lines.append("    classDef conditionNode fill:#d69e2e,stroke:#b7791f,color:#fff")
    lines.append("    classDef loopNode fill:#805ad5,stroke:#6b46c1,color:#fff")

    # Apply classes to nodes
    lines.append("")
    for node_id, node in graph.nodes.items():
        node_type = node.__class__.__name__.replace("Node", "").lower()
        lines.append(f"    class {node_id} {node_type}Node")

    # Apply END class if we added it
    if add_end_node:
        lines.append(f"    class END endNode")

    return "\n".join(lines)


def save_mermaid_image(
    mermaid_code: str,
    output_path: str,
    image_format: str = "png",
    theme: str = "default",
    background_color: str = "white",
) -> str:
    """Save Mermaid diagram as an image using mermaid.ink API.

    Args:
        mermaid_code: Mermaid diagram code to render
        output_path: Path where image should be saved
        image_format: Output format - "png", "svg", or "pdf"
        theme: Mermaid theme - "default", "dark", "forest", "neutral"
        background_color: Background color for the diagram

    Returns:
        Path to saved image file

    Raises:
        httpx.HTTPError: If image generation fails
        ValueError: If invalid format specified

    Example:
        >>> code = generate_mermaid_code(graph, title="My Graph")
        >>> path = save_mermaid_image(code, "output.png", format="png")
    """
    # Validate format
    valid_formats = ["png", "svg", "pdf"]
    if image_format not in valid_formats:
        raise ValueError(f"Invalid format: {image_format}. Must be one of {valid_formats}")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Encode Mermaid code for URL
    # mermaid.ink expects base64-encoded diagram
    encoded = base64.b64encode(mermaid_code.encode("utf-8")).decode("utf-8")

    # Build mermaid.ink URL
    # Different formats use different endpoints
    if image_format == "svg":
        url = f"https://mermaid.ink/svg/{encoded}"
    elif image_format == "pdf":
        url = f"https://mermaid.ink/pdf/{encoded}"
    else:  # png
        # PNG uses /img endpoint with type parameter
        url = f"https://mermaid.ink/img/{encoded}"

    # Build query parameters
    params = {}
    if image_format == "png":
        params["type"] = "png"
    if theme != "default":
        params["theme"] = theme
    if background_color != "white":
        params["bgColor"] = background_color

    # Fetch image from mermaid.ink
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

        # Save image to file
        with open(output_file, "wb") as f:
            f.write(response.content)

    return str(output_file.absolute())


def get_default_visualization_dir() -> Path:
    """Get the default directory for saving visualizations.

    Returns:
        Path to mesh/visualizations/ directory
    """
    # Get mesh package root
    mesh_root = Path(__file__).parent.parent
    vis_dir = mesh_root / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    return vis_dir
