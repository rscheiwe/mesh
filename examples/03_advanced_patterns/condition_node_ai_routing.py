"""AI-Driven Conditional Routing with ConditionNode

This example demonstrates AI mode for ConditionNode - using an LLM to classify input
and route to appropriate handlers. This matches Flowise's Condition Agent Node.

The same ConditionNode supports both:
- Deterministic routing (condition_routing="deterministic") - rule-based
- AI routing (condition_routing="ai") - LLM-based classification

NOTE: This example requires OPENAI_API_KEY to be set in your environment.
"""

import asyncio
import os
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend


# ============================================================================
# Example: Intent Classification Router
# ============================================================================


async def process_input(input: dict) -> dict:
    """Extract user message from input."""
    if "input" in input:
        actual_input = input["input"]
    else:
        actual_input = input
    return {"user_message": actual_input.get("message", "")}


async def sales_handler(input: dict) -> dict:
    """Handle sales inquiries."""
    return {"response": "Routing to Sales Team...", "department": "sales"}


async def support_handler(input: dict) -> dict:
    """Handle support requests."""
    return {"response": "Routing to Technical Support...", "department": "support"}


async def billing_handler(input: dict) -> dict:
    """Handle billing questions."""
    return {"response": "Routing to Billing Department...", "department": "billing"}


async def general_handler(input: dict) -> dict:
    """Handle general inquiries."""
    return {"response": "Routing to General Inquiries...", "department": "general"}


async def example_ai_intent_router():
    """Example: AI-driven intent classification and routing."""
    print("\n" + "=" * 70)
    print("Example: AI Intent Router - Customer Service")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not found in environment")
        print("   Set your API key to run this example:")
        print("   export OPENAI_API_KEY='your-key-here'\n")
        return

    graph = StateGraph()

    # Input processor
    graph.add_node("process_input", process_input, node_type="tool")

    # AI-driven condition node
    # The LLM classifies user intent and routes to the appropriate handler
    graph.add_node(
        "intent_classifier",
        None,  # No conditions needed for AI mode
        node_type="condition",
        condition_routing="ai",  # Enable AI mode
        model="gpt-4",
        instructions="Classify the customer's message into one of the following categories based on their intent",
        scenarios=[
            {
                "name": "sales",
                "description": "Questions about products, pricing, features, purchasing, or product comparisons",
                "target": "sales_handler",
            },
            {
                "name": "support",
                "description": "Technical issues, bugs, troubleshooting, how-to questions, or feature requests",
                "target": "support_handler",
            },
            {
                "name": "billing",
                "description": "Payment issues, invoices, refunds, subscription management, or pricing plans",
                "target": "billing_handler",
            },
        ],
        default_target="general_handler",  # Fallback for unclear cases
    )

    # Handler nodes
    graph.add_node("sales_handler", sales_handler, node_type="tool")
    graph.add_node("support_handler", support_handler, node_type="tool")
    graph.add_node("billing_handler", billing_handler, node_type="tool")
    graph.add_node("general_handler", general_handler, node_type="tool")

    # Connect: START → process → intent_classifier → [handlers]
    graph.add_edge("START", "process_input")
    graph.add_edge("process_input", "intent_classifier")
    graph.add_edge("intent_classifier", "sales_handler")
    graph.add_edge("intent_classifier", "support_handler")
    graph.add_edge("intent_classifier", "billing_handler")
    graph.add_edge("intent_classifier", "general_handler")

    graph.set_entry_point("process_input")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())

    # Test cases
    test_cases = [
        {
            "name": "Sales inquiry",
            "message": "What are the pricing plans for the enterprise tier?",
        },
        {
            "name": "Support request",
            "message": "My dashboard is showing an error when I try to export data",
        },
        {
            "name": "Billing question",
            "message": "I was charged twice this month and need a refund",
        },
        {
            "name": "General inquiry",
            "message": "What are your business hours?",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        print(f"User: \"{test_case['message']}\"")

        context = ExecutionContext(
            graph_id="ai_router",
            session_id=test_case["name"],
            chat_history=[],
            variables={},
            state={},
        )

        async for event in executor.execute({"message": test_case["message"]}, context):
            if event.type == "node_complete" and event.node_id == "intent_classifier":
                classification = event.output.get("classification")
                scenario = event.output.get("scenario")
                print(f"✓ LLM classified as: {classification} → routing to {scenario}")

            elif event.type == "node_complete" and event.node_id.endswith("_handler"):
                print(f"✓ Handler response: {event.output.get('response')}")


# ============================================================================
# Example: Hybrid Routing (Deterministic + AI Fallback)
# ============================================================================


async def example_hybrid_routing():
    """Example: Use deterministic rules first, AI as fallback for complex cases."""
    print("\n" + "=" * 70)
    print("Example: Hybrid Routing - Deterministic + AI Fallback")
    print("=" * 70)
    print("\nThis demonstrates using deterministic rules for clear cases,")
    print("and falling back to AI for nuanced classification.\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found - skipping AI example\n")
        return

    from mesh.nodes import SimpleCondition

    graph = StateGraph()

    graph.add_node("process_input", process_input, node_type="tool")

    # First layer: Deterministic checks for obvious cases
    graph.add_node(
        "quick_filter",
        [
            SimpleCondition.contains("user_message", "refund", "billing_handler"),
            SimpleCondition.contains("user_message", "invoice", "billing_handler"),
            SimpleCondition.contains("user_message", "error", "support_handler"),
            SimpleCondition.contains("user_message", "bug", "support_handler"),
        ],
        node_type="condition",
        condition_routing="deterministic",
        default_target="ai_classifier",  # If no deterministic match → use AI
    )

    # Second layer: AI classifier for nuanced cases
    graph.add_node(
        "ai_classifier",
        None,
        node_type="condition",
        condition_routing="ai",
        model="gpt-4",
        instructions="For messages that don't clearly match billing or support keywords, classify the intent",
        scenarios=[
            {"name": "sales", "description": "Product questions", "target": "sales_handler"},
            {"name": "support", "description": "Technical help", "target": "support_handler"},
            {"name": "billing", "description": "Payment/billing", "target": "billing_handler"},
        ],
        default_target="general_handler",
    )

    # Handlers
    graph.add_node("sales_handler", sales_handler, node_type="tool")
    graph.add_node("support_handler", support_handler, node_type="tool")
    graph.add_node("billing_handler", billing_handler, node_type="tool")
    graph.add_node("general_handler", general_handler, node_type="tool")

    # Connect
    graph.add_edge("START", "process_input")
    graph.add_edge("process_input", "quick_filter")
    graph.add_edge("quick_filter", "billing_handler")  # Deterministic route
    graph.add_edge("quick_filter", "support_handler")  # Deterministic route
    graph.add_edge("quick_filter", "ai_classifier")  # Fallback to AI
    graph.add_edge("ai_classifier", "sales_handler")
    graph.add_edge("ai_classifier", "support_handler")
    graph.add_edge("ai_classifier", "billing_handler")
    graph.add_edge("ai_classifier", "general_handler")

    graph.set_entry_point("process_input")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())

    test_cases = [
        {
            "name": "Clear keyword match",
            "message": "I need a refund immediately",
            "expected": "Deterministic → billing",
        },
        {
            "name": "Nuanced question",
            "message": "Can you help me compare the features of different plans?",
            "expected": "AI → sales",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        print(f"User: \"{test_case['message']}\"")
        print(f"Expected: {test_case['expected']}")

        context = ExecutionContext(
            graph_id="hybrid",
            session_id=test_case["name"],
            chat_history=[],
            variables={},
            state={},
        )

        async for event in executor.execute({"message": test_case["message"]}, context):
            if event.type == "node_complete" and event.node_id == "quick_filter":
                fulfilled = event.output.get("fulfilled", [])
                if "ai_classifier" in [cond.split("_")[0] for cond in fulfilled]:
                    print("✓ No deterministic match → falling back to AI")
                else:
                    print(f"✓ Deterministic match: {fulfilled}")

            elif event.type == "node_complete" and event.node_id == "ai_classifier":
                classification = event.output.get("classification")
                print(f"✓ AI classified as: {classification}")

            elif event.type == "node_complete" and event.node_id.endswith("_handler"):
                print(f"✓ Final handler: {event.output.get('department')}")


# ============================================================================
# Run Examples
# ============================================================================


async def main():
    """Run all AI routing examples."""
    print("\n" + "=" * 70)
    print("AI CONDITION NODE ROUTING EXAMPLES")
    print("Demonstrating LLM-driven conditional branching")
    print("=" * 70)

    await example_ai_intent_router()
    await example_hybrid_routing()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
