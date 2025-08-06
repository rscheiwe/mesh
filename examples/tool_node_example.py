"""Example demonstrating ToolNode for data gathering before AI processing."""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

from mesh import Edge, Graph
from mesh.compilation import GraphExecutor, StaticCompiler
from mesh.core import NodeConfig
from mesh.core.node import NodeStatus
from mesh.nodes import LLMNode, MultiToolNode, ToolNode
from mesh.nodes.llm import LLMConfig, LLMProvider
from mesh.nodes.tool import ToolNodeConfig
from mesh.state import GraphState


# Example tool functions
async def fetch_user_data(user_id: str) -> Dict[str, Any]:
    """Mock function to fetch user data from database."""
    # In reality, this would query a database
    users_db = {
        "123": {
            "name": "Alice Johnson",
            "age": 28,
            "interests": ["Python", "AI", "Music"],
            "last_login": "2024-01-15",
        },
        "456": {
            "name": "Bob Smith",
            "age": 35,
            "interests": ["Data Science", "Sports", "Cooking"],
            "last_login": "2024-01-14",
        },
    }

    user = users_db.get(
        user_id,
        {"name": "Unknown User", "age": 0, "interests": [], "last_login": "Never"},
    )

    return {
        "user_id": user_id,
        "profile": user,
        "fetched_at": datetime.now().isoformat(),
    }


async def fetch_product_recommendations(
    user_interests: List[str],
) -> List[Dict[str, Any]]:
    """Mock function to get product recommendations based on interests."""
    products = {
        "Python": [
            {"id": "p1", "name": "Python Crash Course", "type": "Book", "price": 39.99},
            {"id": "p2", "name": "PyCharm Pro", "type": "Software", "price": 199.00},
        ],
        "AI": [
            {
                "id": "p3",
                "name": "Deep Learning Specialization",
                "type": "Course",
                "price": 49.99,
            },
            {
                "id": "p4",
                "name": "GPT-4 API Credits",
                "type": "Service",
                "price": 20.00,
            },
        ],
        "Music": [
            {
                "id": "p5",
                "name": "Studio Headphones",
                "type": "Hardware",
                "price": 299.99,
            },
            {
                "id": "p6",
                "name": "Music Production Course",
                "type": "Course",
                "price": 89.99,
            },
        ],
        "Data Science": [
            {
                "id": "p7",
                "name": "Statistics for Data Science",
                "type": "Book",
                "price": 45.99,
            },
            {"id": "p8", "name": "Jupyter Pro", "type": "Software", "price": 79.99},
        ],
    }

    recommendations = []
    for interest in user_interests:
        if interest in products:
            recommendations.extend(products[interest])

    return recommendations[:5]  # Limit to 5 recommendations


def calculate_discount(user_age: int, last_login: str) -> Dict[str, Any]:
    """Calculate personalized discount based on user data."""
    base_discount = 10

    # Age-based discount
    if user_age < 25:
        base_discount += 5
    elif user_age > 60:
        base_discount += 10

    # Loyalty discount based on last login
    try:
        last_login_date = datetime.fromisoformat(last_login)
        days_since_login = (datetime.now() - last_login_date).days
        if days_since_login < 7:
            base_discount += 5
    except:
        pass

    return {
        "discount_percentage": min(base_discount, 25),  # Cap at 25%
        "reason": "Personalized discount based on profile",
    }


async def example_1_sequential_tools():
    """Example 1: Sequential tool execution providing context to LLM."""
    print("\n=== Example 1: Sequential Tool Execution ===")
    print("Fetch user data -> Get recommendations -> Generate personalized email")

    graph = Graph()

    # Tool node to fetch user data - this is now the start node
    fetch_user = ToolNode(
        tool_func=fetch_user_data,
        config=ToolNodeConfig(
            tool_name="fetch_user",
            tool_description="Fetch user profile data",
            output_key="user_data",
            store_in_state=True,  # Store in state
            state_key="user_profile",
        ),
        extract_args=lambda input_data: {"user_id": input_data["user_id"]},
    )

    # Tool node to get product recommendations
    get_recommendations = ToolNode(
        tool_func=fetch_product_recommendations,
        config=ToolNodeConfig(
            tool_name="get_recommendations",
            tool_description="Get product recommendations",
            output_key="recommendations",
            store_in_state=True,  # Store in state for later access
            state_key="product_recommendations",
        ),
        extract_args=lambda input_data: {
            "user_interests": input_data.get("user_data", {})
            .get("profile", {})
            .get("interests", [])
        },
    )

    # Custom node to prepare context for LLM
    from mesh.nodes import CustomFunctionNode

    async def prepare_email_context(
        data: Dict[str, Any], state: GraphState
    ) -> Dict[str, Any]:
        """Aggregate all context for the email writer."""
        # Get data from state
        user_data = await state.get("user_profile", {})
        recommendations = await state.get("product_recommendations", [])

        # Also get from current input
        current_recommendations = data.get("recommendations", [])

        # Build context
        context = f"""
User Profile:
- Name: {user_data.get("profile", {}).get("name", "Unknown")}
- Age: {user_data.get("profile", {}).get("age", "Unknown")}
- Interests: {", ".join(user_data.get("profile", {}).get("interests", []))}
- Last Login: {user_data.get("profile", {}).get("last_login", "Unknown")}

Product Recommendations ({len(recommendations or current_recommendations)} items):
"""
        for i, product in enumerate(
            (recommendations or current_recommendations)[:5], 1
        ):
            context += f"{i}. {product.get('name', 'Unknown')} - ${product.get('price', 0):.2f} ({product.get('type', 'Unknown')})\n"

        # Return prompt with context
        return {
            "prompt": f"{context}\n\nOriginal request: {data.get('prompt', 'Write a personalized marketing email')}"
        }

    context_preparer = CustomFunctionNode(
        prepare_email_context, config=NodeConfig(name="ContextPreparer")
    )

    # LLM node to generate personalized email
    email_writer = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            system_prompt="""You are a marketing email writer. 
        Create personalized emails based on the provided user data and product recommendations.
        Keep emails friendly and under 150 words.""",
        )
    )

    # Build graph: sequential execution
    graph.add_node(fetch_user)
    graph.add_node(get_recommendations)
    graph.add_node(context_preparer)
    graph.add_node(email_writer)

    # Create sequential chain - fetch_user is the start node (no incoming edges)
    graph.add_edge(Edge(fetch_user.id, get_recommendations.id))
    graph.add_edge(Edge(get_recommendations.id, context_preparer.id))
    graph.add_edge(Edge(context_preparer.id, email_writer.id))

    # Execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    print("\nExecution Plan:")
    for i, group in enumerate(compiled.execution_plan):
        print(f"  Group {i}: {len(group)} nodes")

    # Create state to share data between nodes
    state = GraphState()

    executor = GraphExecutor()
    result = await executor.execute(
        compiled,
        initial_input={
            "user_id": "123",
            "prompt": "Write a personalized marketing email for this user featuring their recommended products.",
        },
        state=state,
    )

    # Print results with error handling
    if fetch_user.id in result.outputs:
        output = result.outputs[fetch_user.id]
        if output.status == NodeStatus.COMPLETED and output.data:
            print(
                f"\nUser Data: {json.dumps(output.data.get('user_data', {}), indent=2)}"
            )
        else:
            print(f"\nUser Data: Failed - {output.error}")
    else:
        print("\nUser Data: Not in outputs")

    if (
        get_recommendations.id in result.outputs
        and result.outputs[get_recommendations.id].data
    ):
        recs = result.outputs[get_recommendations.id].data.get("recommendations", [])
        print(f"\nRecommendations: {len(recs)} products")
        for rec in recs[:3]:  # Show first 3
            print(f"  - {rec}")
    else:
        print("\nRecommendations: Not available")
        print(
            f"Debug - get_recommendations output: {result.outputs.get(get_recommendations.id)}"
        )

    if email_writer.id in result.outputs and result.outputs[email_writer.id].data:
        print(
            f"\nGenerated Email:\n{result.outputs[email_writer.id].data.get('response', 'No response')}"
        )
    else:
        print("\nGenerated Email: Not available")


async def example_2_multi_tool_node():
    """Example 2: MultiToolNode executing multiple tools in parallel."""
    print("\n=== Example 2: MultiToolNode (Parallel Execution) ===")
    print("Fetch multiple data sources in parallel -> Generate report")

    graph = Graph()

    # MultiToolNode to fetch data in parallel - this is the start node
    data_fetcher = MultiToolNode(
        tools=[
            ("user_data", fetch_user_data),
            ("recommendations", fetch_product_recommendations),
            ("discount", calculate_discount),
        ],
        parallel=True,  # Execute all tools in parallel
    )

    # LLM node to generate report
    report_writer = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
            system_prompt="You are a business analyst. Create concise reports from data.",
        )
    )

    # Build graph
    graph.add_node(data_fetcher)
    graph.add_node(report_writer)

    # data_fetcher is the start node (no incoming edges)
    graph.add_edge(Edge(data_fetcher.id, report_writer.id))

    # Execute
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    # Prepare input with proper structure for MultiToolNode
    input_data = {
        "tool_args": {
            "user_id": "456",
            "user_interests": ["Data Science", "Sports"],
            "user_age": 35,
            "last_login": "2024-01-14",
        },
        "prompt": "Generate a customer profile report including their data, recommendations, and applicable discounts.",
    }

    executor = GraphExecutor()
    result = await executor.execute(compiled, initial_input=input_data)

    # Print results with error handling
    if data_fetcher.id in result.outputs and result.outputs[data_fetcher.id].data:
        tool_results = result.outputs[data_fetcher.id].data.get("tool_results", {})
        tools_executed = result.outputs[data_fetcher.id].data.get("tools_executed", [])
        print(f"\nTools executed: {tools_executed}")
        for tool_name, tool_result in tool_results.items():
            print(f"  - {tool_name}: {type(tool_result).__name__}")
    else:
        print("\nData fetcher output not available")
        print(f"Debug - data_fetcher output: {result.outputs.get(data_fetcher.id)}")

    if report_writer.id in result.outputs and result.outputs[report_writer.id].data:
        print(
            f"\nGenerated Report:\n{result.outputs[report_writer.id].data.get('response', 'No response')}"
        )
    else:
        print("\nReport not available")


async def example_3_conditional_tool_usage():
    """Example 3: Conditional tool execution based on previous results."""
    print("\n=== Example 3: Conditional Tool Execution ===")
    print("Check user type -> Execute appropriate tools -> Generate response")

    from mesh.nodes import ConditionalNode

    graph = Graph()
    state = GraphState()

    # First tool to check user type - this is the start node
    check_user_type = ToolNode(
        tool_func=lambda user_id: {
            "user_id": user_id,
            "is_premium": user_id == "123",  # Mock check
            "account_type": "premium" if user_id == "123" else "basic",
        },
        config=ToolNodeConfig(tool_name="check_user_type", output_key="user_type_info"),
        extract_args=lambda data: {"user_id": data["user_id"]},
    )

    # Conditional node to route based on user type
    router = ConditionalNode(
        condition=lambda data, state: data.get("user_type_info", {}).get(
            "is_premium", False
        ),
        true_output={"route": "premium"},
        false_output={"route": "basic"},
    )

    # Premium user tools
    premium_tools = MultiToolNode(
        tools=[
            ("user_data", fetch_user_data),
            ("recommendations", fetch_product_recommendations),
            (
                "discount",
                lambda **kwargs: {
                    "discount_percentage": 20,
                    "reason": "Premium member",
                },
            ),
        ],
        parallel=True,
    )

    # Basic user tools
    basic_tools = ToolNode(
        tool_func=fetch_user_data,
        config=ToolNodeConfig(tool_name="basic_fetch", output_key="user_data"),
        extract_args=lambda data: {"user_id": data["user_id"]},
    )

    # LLM for response generation
    response_generator = LLMNode(
        config=LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            system_prompt="Generate appropriate responses based on user type and available data.",
        )
    )

    # Build graph with conditional routing
    graph.add_node(check_user_type)
    graph.add_node(router)
    graph.add_node(premium_tools)
    graph.add_node(basic_tools)
    graph.add_node(response_generator)

    # Connect nodes - check_user_type is the start node
    graph.add_edge(Edge(check_user_type.id, router.id))

    # Conditional routing
    from mesh.core.edge import EdgeType

    graph.add_edge(
        Edge(
            router.id,
            premium_tools.id,
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda data: data.get("route") == "premium",
        )
    )
    graph.add_edge(
        Edge(
            router.id,
            basic_tools.id,
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda data: data.get("route") == "basic",
        )
    )

    # Both routes lead to response generator
    graph.add_edge(Edge(premium_tools.id, response_generator.id))
    graph.add_edge(Edge(basic_tools.id, response_generator.id))

    # Execute with different user IDs
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    executor = GraphExecutor()

    # Test with premium user
    print("\n--- Premium User ---")
    result = await executor.execute(
        compiled,
        initial_input={
            "user_id": "123",
            "tool_args": {"user_id": "123", "user_interests": ["Python", "AI"]},
            "prompt": "Create a personalized message for this user.",
        },
        state=state,
    )

    print("Route taken: Premium")
    print(
        f"Tools executed: {len(result.outputs.get(premium_tools.id, {}).get('data', {}).get('tools_executed', []))} tools"
    )


async def main():
    """Run all examples."""
    print("ToolNode Examples")
    print("=" * 50)

    try:
        await example_1_sequential_tools()
        # await example_2_multi_tool_node()
        # await example_3_conditional_tool_usage()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
