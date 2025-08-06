"""Simplified example showing feedback loop with smart edges - no final node."""

import asyncio
from typing import Any

from mesh import Graph
from mesh.compilation import StaticCompiler
from mesh.compilation.smart_executor import SmartGraphExecutor
from mesh.core import NodeConfig
from mesh.core.edge import Edge
from mesh.core.smart_edge import FeedbackLoopEdge
from mesh.nodes import CustomFunctionNode
from mesh.state import GraphState


async def main() -> None:
    """Feedback loop where feedback node is terminal when approved."""

    print("=== Smart Edge Feedback Loop (No Final Node) ===\n")

    graph = Graph()

    # Mock writer node
    def write_func(data: dict, state: Any) -> dict:
        # Track iteration properly
        iteration = data.get("iteration", 1)
        if data.get("feedback"):
            # This is a loop iteration
            iteration += 1

        # Get the prompt from initial input or previous text
        prompt = data.get("prompt", "")
        text = data.get("text", prompt)

        # Apply feedback if present
        if data.get("feedback"):
            text = f"v{iteration}: {text} (improved based on: {data['feedback']})"
        else:
            text = f"v{iteration}: Initial draft - {text}"

        return {"text": text, "iteration": iteration}

    writer = CustomFunctionNode(write_func, config=NodeConfig(name="Writer"))

    # Mock feedback node - this is terminal when approved
    def feedback_func(data: dict, state: Any) -> dict:
        iteration = data.get("iteration", 1)
        text = data.get("text", "")

        # Simulate feedback that approves after 3 iterations
        if iteration >= 3:
            # When approved, this becomes the final output
            return {
                "status": "approved",
                "final_text": text,
                "total_iterations": iteration,
                "message": "APPROVED! The text looks great now.",
            }
        else:
            return {
                "status": "needs_revision",
                "feedback": f"Make it better (feedback #{iteration})",
                "text": text,
                "iteration": iteration,
            }

    feedback = CustomFunctionNode(feedback_func, config=NodeConfig(name="Feedback"))

    # Build graph - much simpler!
    graph.add_node(writer)
    graph.add_node(feedback)

    # Writer node has no incoming edges initially, so it becomes a starting node automatically
    graph.add_edge(Edge(writer.id, feedback.id))

    # Smart feedback loop edge - only active when not approved
    feedback_loop = FeedbackLoopEdge(
        source_node_id=feedback.id,
        target_node_id=writer.id,
        approval_keywords=["APPROVED", "approved"],
        approval_field="status",  # Check the status field
        max_iterations=5,
    )
    graph.add_edge(feedback_loop)

    # No edge from feedback when approved - it's terminal!

    # Execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    executor = SmartGraphExecutor()
    state = GraphState()

    result = await executor.execute(
        compiled, initial_input={"prompt": "Write about smart edges"}, state=state
    )

    # Show execution flow
    print("Execution flow:")
    for node_id, output in result.outputs.items():
        node_name = output.metadata.get("node_name", node_id)
        if node_name == "Writer":
            print(f"  📝 {node_name}: {output.data.get('text', output.data)}")
        elif node_name == "Feedback":
            status = output.data.get("status", "unknown")
            if status == "approved":
                print(f"  ✅ {node_name}: {output.data.get('message', 'Approved')}")
            else:
                print(
                    f"  📋 {node_name}: {output.data.get('feedback', 'Needs revision')}"
                )

    # The final output is from the feedback node itself
    final_output = result.get_final_output()
    print("\n=== Final Output ===")
    if isinstance(final_output, dict):
        # If we get the data directly
        final = final_output
    elif hasattr(final_output, "data"):
        # If we get a NodeOutput object
        final = final_output.data
    else:
        final = {}

    print(f"Status: {final.get('status', 'unknown')}")
    print(f"Final text: {final.get('final_text', 'N/A')}")
    print(f"Total iterations: {final.get('total_iterations', 0)}")


async def real_world_example():
    """More realistic example with actual LLM-like behavior."""

    print("\n\n=== Real-World Style Example ===\n")

    from mesh.nodes import AgentNode
    from mesh.nodes.agent import AgentConfig
    from mesh.nodes.llm import LLMProvider

    graph = Graph()

    # Email writer
    write_email = AgentNode(
        config=AgentConfig(
            name="EmailWriter",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-api-key",
            system_prompt="Write professional emails. Improve based on any feedback provided.",
        )
    )

    # Email reviewer - terminal when approved
    review_email = AgentNode(
        config=AgentConfig(
            name="EmailReviewer",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-api-key",
            system_prompt="""Review emails. 
        If excellent, respond with: APPROVED: [final email text]
        Otherwise, provide specific feedback for improvement.""",
        )
    )

    # Build graph
    graph.add_node(write_email)
    graph.add_node(review_email)

    # WriteEmail node has no incoming edges, so it becomes a starting node automatically
    graph.add_edge(Edge(write_email.id, review_email.id))

    # Smart feedback loop
    graph.add_edge(
        FeedbackLoopEdge(
            source_node_id=review_email.id,
            target_node_id=write_email.id,
            approval_keywords=["APPROVED"],
            max_iterations=3,
        )
    )

    # ReviewEmail has no outgoing edges when approved - it's the terminal node

    print("Graph structure:")
    print("Start -> EmailWriter -> EmailReviewer")
    print("         ^                    |")
    print("         |____(if not approved)")
    print("\nEmailReviewer is terminal when email is approved.")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(real_world_example())
