"""Example demonstrating stateless conversation handling - Real-world pattern."""

import asyncio
from typing import Any, Dict, List

from mesh import Graph
from mesh.compilation import GraphExecutor, StaticCompiler
from mesh.nodes import AgentNode
from mesh.nodes.agent import AgentConfig
from mesh.nodes.llm import LLMProvider, Message


# Simulate frontend message storage
class FrontendSimulator:
    """Simulates a frontend that maintains conversation history."""

    def __init__(self):
        self.message_history: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.message_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Add an assistant response to history."""
        self.message_history.append({"role": "assistant", "content": content})

    def get_messages_for_request(self):
        """Get all messages formatted for API request."""
        return self.message_history.copy()

    def display_conversation(self):
        """Display the conversation history."""
        print("\n=== Conversation History (Frontend View) ===")
        for i, msg in enumerate(self.message_history):
            role = msg["role"].capitalize()
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
            print(f"{i + 1}. {role}: {content}")


async def main():
    """Example of stateless conversation handling."""

    print("=== Stateless Conversation Example ===\n")

    # Create simple graph
    graph = Graph()

    # Standard agent node - no custom logic needed!
    chatbot = AgentNode(
        config=AgentConfig(
            name="Chatbot",
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt="You are a helpful assistant. Answer questions based on the conversation history.",
        )
    )

    # Build graph - chatbot is the only node, so it's automatically the start node
    graph.add_node(chatbot)

    # Compile graph once
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    executor = GraphExecutor()

    # Simulate frontend
    frontend = FrontendSimulator()

    # Helper function to send request with message history
    async def send_chat_request(user_input: str):
        """Send a chat request with full message history."""
        # Add user message to frontend history
        frontend.add_user_message(user_input)

        # Get all messages for the request
        messages = frontend.get_messages_for_request()

        # Execute graph with messages
        result = await executor.execute(
            compiled,
            initial_input={
                "messages": messages  # Send full history
            },
        )

        # Extract response
        if result.success and chatbot.id in result.outputs:
            response_data = result.outputs[chatbot.id].data.get("response", {})
            if isinstance(response_data, dict) and "content" in response_data:
                response = response_data["content"]
            else:
                response = str(response_data)

            # Add assistant response to frontend history
            frontend.add_assistant_message(response)

            return response
        else:
            return "Error: No response"

    # Simulate conversation
    print("User: Tell me about Python in 50 words or less")
    response1 = await send_chat_request("Tell me about Python in 50 words or less")
    print(f"Bot: {response1}\n")

    print("User: What are its main features?")
    response2 = await send_chat_request("What are its main features?")
    print(f"Bot: {response2}\n")

    print("User: How does it compare to JavaScript?")
    response3 = await send_chat_request("How does it compare to JavaScript?")
    print(f"Bot: {response3}\n")

    # Show the conversation history as stored by the frontend
    frontend.display_conversation()


async def api_endpoint_example():
    """Example showing how this would work as an API endpoint."""

    print("\n\n=== API Endpoint Pattern ===\n")

    # Setup graph once at startup
    graph = Graph()
    chatbot = AgentNode(
        config=AgentConfig(
            name="APIBot",
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key="your-api-key",
            system_prompt="You are a helpful API assistant.",
        )
    )

    graph.add_node(chatbot)

    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)
    executor = GraphExecutor()

    # Simulate API requests
    async def handle_chat_request(request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a chat API request."""
        # Extract messages from request body
        messages = request_body.get("messages", [])

        # Execute graph
        result = await executor.execute(compiled, initial_input={"messages": messages})

        # Format response
        if result.success and chatbot.id in result.outputs:
            response_data = result.outputs[chatbot.id].data.get("response", {})
            if isinstance(response_data, dict) and "content" in response_data:
                content = response_data["content"]
            else:
                content = str(response_data)

            return {
                "success": True,
                "response": {"role": "assistant", "content": content},
            }
        else:
            return {"success": False, "error": "Failed to generate response"}

    # Example API requests
    print("Request 1:")
    request1 = {"messages": [{"role": "user", "content": "What is machine learning?"}]}
    response1 = await handle_chat_request(request1)
    print(f"Response: {response1}\n")

    print("Request 2 (with history):")
    request2 = {
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": response1["response"]["content"]},
            {"role": "user", "content": "Give me a simple example"},
        ]
    }
    response2 = await handle_chat_request(request2)
    print(f"Response: {response2['response']['content'][:200]}...")


if __name__ == "__main__":
    # Run stateless conversation example
    asyncio.run(main())

    # Uncomment to see API endpoint pattern
    # asyncio.run(api_endpoint_example())
