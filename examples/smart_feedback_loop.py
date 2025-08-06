"""Example: Feedback loop using smart edges without a decision node."""

import asyncio

from mesh import Graph
from mesh.compilation import StaticCompiler
from mesh.compilation.smart_executor import SmartGraphExecutor
from mesh.core import NodeConfig
from mesh.core.edge import Edge, EdgeType
from mesh.core.smart_edge import FeedbackLoopEdge, create_feedback_loop
from mesh.nodes import AgentNode, CustomFunctionNode
from mesh.nodes.agent import AgentConfig
from mesh.nodes.llm import LLMProvider
from mesh.state import GraphState


async def main():
    """Feedback loop without explicit decision node."""

    print("=== Smart Edge Feedback Loop ===\n")

    # Create graph
    graph = Graph()

    # WriteEmail agent
    write_email = AgentNode(
        config=AgentConfig(
            name="WriteEmail",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-api-key",
            system_prompt="""You are an email writer. 
Write professional emails based on the prompt.
If you receive feedback, incorporate it to improve the email.""",
            max_iterations=1,
        )
    )

    # Feedback agent
    feedback = AgentNode(
        config=AgentConfig(
            name="Feedback",
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="your-api-key",
            system_prompt="""You are a feedback provider. 
Review emails and provide constructive feedback.
If the email is excellent, respond with 'APPROVED'.""",
            max_iterations=1,
        )
    )

    # Add nodes to graph
    graph.add_node(write_email)
    graph.add_node(feedback)

    # WriteEmail node has no incoming edges, so it becomes a starting node automatically

    # WriteEmail -> Feedback
    graph.add_edge(Edge(write_email.id, feedback.id))

    # Feedback -> WriteEmail (smart feedback loop)
    # This edge handles the approval check internally
    feedback_loop = create_feedback_loop(
        feedback_node_id=feedback.id,
        write_node_id=write_email.id,
        approval_keywords=["APPROVED", "LOOKS GOOD", "PERFECT"],
        max_iterations=5,
    )
    graph.add_edge(feedback_loop)

    # No edge from feedback when approved - it becomes terminal!

    # Compile and execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    # Use smart executor
    executor = SmartGraphExecutor(max_loops=10)

    # Track execution
    iteration_count = 0

    async def track_iterations(event):
        nonlocal iteration_count
        if event["type"] == "node_started" and event["node_name"] == "WriteEmail":
            iteration_count += 1
            print(f"📝 Writing email (iteration {iteration_count})")
        elif event["type"] == "node_completed" and event["node_name"] == "Feedback":
            print("📋 Feedback provided")

    executor.add_event_handler(track_iterations)

    # Execute with initial input
    state = GraphState()
    result = await executor.execute(
        compiled,
        initial_input={
            "prompt": "Write an email announcing our new AI ethics guidelines"
        },
        state=state,
    )

    # Show final result - it's from the feedback node when approved
    final = result.get_final_output()
    print("\n=== Final Result ===")
    print(f"✅ Email approved after {iteration_count} iterations")
    if isinstance(final, dict):
        print(f"Final response: {final.get('response', final)}")
    else:
        print(f"Final output: {final}")


async def advanced_example():
    """More complex example with multiple feedback types."""

    print("\n\n=== Advanced Smart Edge Example ===\n")

    from mesh.core.smart_edge import RouterFunction, SmartEdge

    graph = Graph()

    writer = AgentNode(config=AgentConfig(name="Writer"))
    reviewer = AgentNode(config=AgentConfig(name="Reviewer"))

    # Multiple output nodes based on review
    minor_revision = CustomFunctionNode(
        lambda d, s: {"result": "Minor revisions needed", "data": d},
        config=NodeConfig(name="MinorRevision"),
    )
    major_revision = CustomFunctionNode(
        lambda d, s: {"result": "Major revisions needed", "data": d},
        config=NodeConfig(name="MajorRevision"),
    )
    approved = CustomFunctionNode(
        lambda d, s: {"result": "Approved!", "data": d},
        config=NodeConfig(name="Approved"),
    )

    # Add nodes
    for node in [writer, reviewer, minor_revision, major_revision, approved]:
        graph.add_node(node)

    # Writer node has no incoming edges, so it becomes a starting node automatically
    graph.add_edge(Edge(writer.id, reviewer.id))

    # Smart routing based on review
    def review_router(data, state):
        """Route based on review content."""
        response = str(data.get("response", "")).upper()

        if "APPROVED" in response:
            return "approved"
        elif "MAJOR" in response or "SIGNIFICANT" in response:
            return "major"
        else:
            return "minor"

    # Multi-conditional smart edge from reviewer
    routing_edge = SmartEdge(
        source_node_id=reviewer.id,
        target_node_id="",  # Not used for multi-conditional
        edge_type="multi_conditional",
        router=RouterFunction(review_router),
        targets={
            "approved": approved.id,
            "major": major_revision.id,
            "minor": minor_revision.id,
        },
    )

    # For now, use separate edges (multi-conditional would need graph support)
    graph.add_edge(
        Edge(
            reviewer.id,
            approved.id,
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda d: "APPROVED" in str(d.get("response", "")).upper(),
        )
    )
    graph.add_edge(
        Edge(
            reviewer.id,
            major_revision.id,
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda d: any(
                w in str(d.get("response", "")).upper()
                for w in ["MAJOR", "SIGNIFICANT"]
            ),
        )
    )
    graph.add_edge(
        Edge(
            reviewer.id,
            minor_revision.id,
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda d: not any(
                w in str(d.get("response", "")).upper()
                for w in ["APPROVED", "MAJOR", "SIGNIFICANT"]
            ),
        )
    )

    # Loop back from revisions to writer
    graph.add_edge(
        FeedbackLoopEdge(
            minor_revision.id,
            writer.id,
            approval_keywords=["DONE"],  # Never matches, so it loops once
            max_iterations=1,
        )
    )
    graph.add_edge(
        FeedbackLoopEdge(
            major_revision.id,
            writer.id,
            approval_keywords=["DONE"],  # Never matches, so it loops once
            max_iterations=1,
        )
    )

    # Execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    executor = SmartGraphExecutor()
    result = await executor.execute(
        compiled, initial_input={"prompt": "Write about smart edges"}
    )

    print("Execution complete!")


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment for advanced example
    # asyncio.run(advanced_example())
