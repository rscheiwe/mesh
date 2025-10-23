"""Conditional Routing with ConditionNode

This example demonstrates how to use ConditionNode for branching logic based on
conditions. The ConditionNode evaluates deterministic rules (no AI) to route
workflow execution down different paths.

This matches Flowise's Condition Node behavior - pure logic-based branching.
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import Condition, SimpleCondition


# ============================================================================
# Example 1: Simple True/False Routing
# ============================================================================
# This demonstrates basic conditional routing based on a single condition


async def check_age(input: dict) -> dict:
    """Tool that extracts age from input."""
    # Handle nested input from START node
    if "input" in input:
        actual_input = input["input"]
    else:
        actual_input = input
    return {"age": actual_input.get("age", 0)}


async def adult_handler(input: dict) -> dict:
    """Handler for adult path."""
    # Extract age from condition node output
    if "input" in input and isinstance(input["input"], dict):
        age = input["input"].get("age", "unknown")
    else:
        age = input.get("age", "unknown")
    return {"message": f"Adult path - age {age} is >= 18"}


async def minor_handler(input: dict) -> dict:
    """Handler for minor path."""
    # Extract age from condition node output
    if "input" in input and isinstance(input["input"], dict):
        age = input["input"].get("age", "unknown")
    else:
        age = input.get("age", "unknown")
    return {"message": f"Minor path - age {age} is < 18"}


async def example_1_true_false_routing():
    """Example 1: Simple true/false routing based on age check."""
    print("\n" + "=" * 70)
    print("Example 1: True/False Routing - Age Check")
    print("=" * 70)

    graph = StateGraph()

    # Add check node (extracts age)
    graph.add_node("check_age", check_age, node_type="tool")

    # Add condition node with true/false branches
    # TRUE branch: age >= 18 → adult_path
    # FALSE branch: age < 18 → minor_path
    graph.add_node(
        "age_condition",
        [
            SimpleCondition.greater_than("age", 17.99, "adult_path"),  # True condition
            SimpleCondition.less_than("age", 18, "minor_path"),  # False condition
        ],
        node_type="condition",
        condition_routing="deterministic",  # Explicit mode
    )

    # Add handlers
    graph.add_node("adult_path", adult_handler, node_type="tool")
    graph.add_node("minor_path", minor_handler, node_type="tool")

    # Connect: START → check_age → age_condition → [adult_path | minor_path]
    graph.add_edge("START", "check_age")
    graph.add_edge("check_age", "age_condition")
    graph.add_edge("age_condition", "adult_path")
    graph.add_edge("age_condition", "minor_path")

    graph.set_entry_point("check_age")

    # Execute with adult age
    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())

    print("\n--- Test 1: Adult (age=25) ---")
    context = ExecutionContext(
        graph_id="example1",
        session_id="test1",
        chat_history=[],
        variables={},
        state={},
    )

    async for event in executor.execute({"age": 25}, context):
        if event.type == "node_complete":
            print(f"✓ {event.node_id}: {event.output}")

    print("\n--- Test 2: Minor (age=15) ---")
    context = ExecutionContext(
        graph_id="example1",
        session_id="test2",
        chat_history=[],
        variables={},
        state={},
    )

    async for event in executor.execute({"age": 15}, context):
        if event.type == "node_complete":
            print(f"✓ {event.node_id}: {event.output}")


# ============================================================================
# Example 2: Multiple Conditions with All SimpleCondition Helpers
# ============================================================================


async def process_request(input: dict) -> dict:
    """Tool that processes incoming request."""
    # Handle nested input from START node
    if "input" in input:
        actual_input = input["input"]
    else:
        actual_input = input
    return {
        "status": actual_input.get("status", "unknown"),
        "message": actual_input.get("message", ""),
        "score": actual_input.get("score", 0.0),
        "errors": actual_input.get("errors", []),
    }


async def success_path(input: dict) -> dict:
    return {"result": "SUCCESS: Request processed successfully"}


async def error_path(input: dict) -> dict:
    return {"result": "ERROR: Request contains errors"}


async def low_score_path(input: dict) -> dict:
    score = input.get("score", 0)
    return {"result": f"LOW SCORE: Score {score} is below threshold"}


async def empty_errors_path(input: dict) -> dict:
    return {"result": "NO ERRORS: Clean request with no errors"}


async def example_2_multiple_conditions():
    """Example 2: Multiple conditions using all SimpleCondition helpers."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Conditions - All SimpleCondition Operations")
    print("=" * 70)

    graph = StateGraph()

    # Process node
    graph.add_node("process", process_request, node_type="tool")

    # Multi-condition routing demonstrating all helpers
    graph.add_node(
        "router",
        [
            # equals: Check if status is "success"
            SimpleCondition.equals("status", "success", "success_path"),
            # contains: Check if message contains "error"
            SimpleCondition.contains("message", "error", "error_path"),
            # greater_than: Check if score > 0.7
            SimpleCondition.greater_than("score", 0.7, "high_score_path"),
            # less_than: Check if score < 0.3
            SimpleCondition.less_than("score", 0.3, "low_score_path"),
            # is_empty: Check if errors list is empty
            SimpleCondition.is_empty("errors", "empty_errors_path"),
            # not_equal: Check if status is NOT "pending"
            SimpleCondition.not_equal("status", "pending", "not_pending_path"),
            # not_contains: Check if message does NOT contain "skip"
            SimpleCondition.not_contains("message", "skip", "process_path"),
        ],
        node_type="condition",
        condition_routing="deterministic",
    )

    # Handler nodes
    graph.add_node("success_path", success_path, node_type="tool")
    graph.add_node("error_path", error_path, node_type="tool")
    graph.add_node("low_score_path", low_score_path, node_type="tool")
    graph.add_node("empty_errors_path", empty_errors_path, node_type="tool")

    # Note: In real usage, you'd have handler nodes for all paths
    # For this example, we're showing the condition evaluation

    graph.add_edge("START", "process")
    graph.add_edge("process", "router")
    graph.add_edge("router", "success_path")
    graph.add_edge("router", "error_path")
    graph.add_edge("router", "low_score_path")
    graph.add_edge("router", "empty_errors_path")

    graph.set_entry_point("process")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())

    # Test different scenarios
    test_cases = [
        {
            "name": "Success case",
            "input": {"status": "success", "message": "All good", "score": 0.9, "errors": []},
        },
        {
            "name": "Error case",
            "input": {
                "status": "failed",
                "message": "An error occurred",
                "score": 0.5,
                "errors": ["timeout"],
            },
        },
        {
            "name": "Low score case",
            "input": {"status": "complete", "message": "Done", "score": 0.2, "errors": []},
        },
    ]

    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        print(f"Input: {test_case['input']}")
        context = ExecutionContext(
            graph_id="example2",
            session_id=test_case["name"],
            chat_history=[],
            variables={},
            state={},
        )

        async for event in executor.execute(test_case["input"], context):
            if event.type == "node_complete" and event.node_id == "router":
                print(f"Fulfilled conditions: {event.output.get('fulfilled', [])}")
            elif event.type == "node_complete" and event.node_id in [
                "success_path",
                "error_path",
                "low_score_path",
                "empty_errors_path",
            ]:
                print(f"✓ Routed to: {event.node_id}")
                print(f"  Result: {event.output.get('result')}")


# ============================================================================
# Example 3: Custom Predicate Functions for Complex Logic
# ============================================================================


async def analyze_data(input: dict) -> dict:
    """Analyze input data."""
    # Handle nested input from START node
    if "input" in input:
        actual_input = input["input"]
    else:
        actual_input = input
    return {
        "temperature": actual_input.get("temperature", 0),
        "humidity": actual_input.get("humidity", 0),
        "pressure": actual_input.get("pressure", 0),
    }


async def optimal_conditions(input: dict) -> dict:
    return {"result": "OPTIMAL: All conditions are perfect"}


async def suboptimal_conditions(input: dict) -> dict:
    return {"result": "SUBOPTIMAL: Some conditions are off"}


async def example_3_custom_predicates():
    """Example 3: Custom predicate functions for complex logic."""
    print("\n" + "=" * 70)
    print("Example 3: Custom Predicates - Complex Logic")
    print("=" * 70)

    # Custom predicate: Check if temperature and humidity are in optimal range
    def is_optimal(data: dict) -> bool:
        temp = data.get("temperature", 0)
        humidity = data.get("humidity", 0)
        return 20 <= temp <= 25 and 40 <= humidity <= 60

    # Custom predicate: Check if any value is out of safe range
    def is_dangerous(data: dict) -> bool:
        temp = data.get("temperature", 0)
        pressure = data.get("pressure", 0)
        return temp > 30 or pressure > 1013

    graph = StateGraph()

    graph.add_node("analyze", analyze_data, node_type="tool")

    # Condition with custom predicates
    graph.add_node(
        "environment_check",
        [
            Condition(
                name="optimal_range", predicate=is_optimal, target_node="optimal_conditions"
            ),
            Condition(
                name="dangerous_range", predicate=is_dangerous, target_node="alert_path"
            ),
        ],
        node_type="condition",
        condition_routing="deterministic",
        default_target="suboptimal_conditions",  # Default if no conditions match
    )

    graph.add_node("optimal_conditions", optimal_conditions, node_type="tool")
    graph.add_node("suboptimal_conditions", suboptimal_conditions, node_type="tool")

    graph.add_edge("START", "analyze")
    graph.add_edge("analyze", "environment_check")
    graph.add_edge("environment_check", "optimal_conditions")
    graph.add_edge("environment_check", "suboptimal_conditions")

    graph.set_entry_point("analyze")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())

    test_cases = [
        {"name": "Optimal", "input": {"temperature": 22, "humidity": 50, "pressure": 1010}},
        {"name": "Too cold", "input": {"temperature": 15, "humidity": 50, "pressure": 1010}},
        {"name": "Too humid", "input": {"temperature": 22, "humidity": 70, "pressure": 1010}},
    ]

    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        print(f"Input: {test_case['input']}")
        context = ExecutionContext(
            graph_id="example3",
            session_id=test_case["name"],
            chat_history=[],
            variables={},
            state={},
        )

        async for event in executor.execute(test_case["input"], context):
            if event.type == "node_complete" and event.node_id in [
                "optimal_conditions",
                "suboptimal_conditions",
            ]:
                print(f"✓ {event.output.get('result')}")


# ============================================================================
# Run All Examples
# ============================================================================


async def main():
    """Run all condition node examples."""
    print("\n" + "=" * 70)
    print("CONDITION NODE ROUTING EXAMPLES")
    print("Demonstrating Flowise-compatible conditional branching")
    print("=" * 70)

    await example_1_true_false_routing()
    await example_2_multiple_conditions()
    await example_3_custom_predicates()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
