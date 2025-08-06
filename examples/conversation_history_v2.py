"""Example demonstrating conversation history management using GraphState - Fixed version."""

import asyncio
from typing import Any, Dict, List

from mesh import Edge, Graph
from mesh.compilation import GraphExecutor, StaticCompiler
from mesh.core import NodeConfig
from mesh.nodes import AgentNode, CustomFunctionNode
from mesh.nodes.agent import AgentConfig
from mesh.nodes.llm import LLMProvider, Message
from mesh.state import GraphState


class ConversationNode(AgentNode):
    """Custom agent node that maintains conversation history in state."""

    async def _execute_impl(
        self, input_data: Dict[str, Any], state: GraphState = None
    ) -> Dict[str, Any]:
        """Execute with conversation history management."""

        # Input data is passed directly without StartNode wrapping

        # Get message history from state
        message_history = []
        if state:
            stored_messages = await state.get("message_history", [])
            print(f"Debug - Retrieved {len(stored_messages)} messages from history")

            # Convert stored messages to Message objects
            for msg in stored_messages:
                message_history.append(
                    Message(role=msg["role"], content=msg["content"])
                )

        # Prepare input with full conversation history
        prompt = input_data.get("prompt", "")

        # Add current user prompt to messages if provided
        if prompt:
            message_history.append(Message(role="user", content=prompt))

        # Create input with messages
        modified_input = {"messages": message_history, "prompt": prompt}

        # Call parent implementation
        result = await super()._execute_impl(modified_input, state)

        # Update conversation history in state
        if state and "messages" in result:
            # Convert messages to storable format
            messages_to_store = []
            for msg in result["messages"]:
                if hasattr(msg, "role"):
                    messages_to_store.append({"role": msg.role, "content": msg.content})

            # Add the assistant's response
            if "response" in result:
                response = result["response"]
                if isinstance(response, dict) and "content" in response:
                    messages_to_store.append(
                        {"role": "assistant", "content": response["content"]}
                    )
                else:
                    messages_to_store.append(
                        {"role": "assistant", "content": str(response)}
                    )

            # Update state
            await state.set("message_history", messages_to_store)
            print(f"Debug - Stored {len(messages_to_store)} messages to history")

        return result


async def main():
    """Example of maintaining conversation history across interactions."""

    print("=== Conversation History Example (v2) ===\n")

    # Create graph
    graph = Graph()

    # Use our custom conversation node
    chatbot = ConversationNode(
        config=AgentConfig(
            name="Chatbot",
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            system_prompt="You are a helpful assistant. Use the conversation history to provide contextual responses.",
        )
    )

    # Build simple graph
    graph.add_node(chatbot)
    # Chatbot node has no incoming edges, so it becomes a starting node automatically

    # Compile graph
    compiler = StaticCompiler()
    compiled = await compiler.compile(graph)

    # Create persistent state
    state = GraphState()

    # Execute multiple conversations with the same state
    executor = GraphExecutor()

    # First interaction
    print("User: Tell me about Python in 50 words or less")
    result1 = await executor.execute(
        compiled,
        initial_input={"prompt": "Tell me about Python in 50 words or less"},
        state=state,
    )

    if result1.outputs.get(chatbot.id):
        response = result1.outputs[chatbot.id].data.get("response", {})
        if isinstance(response, dict) and "content" in response:
            print(f"Bot: {response['content']}\n")
        else:
            print(f"Bot: {response}\n")

    # Second interaction - bot should remember previous context
    print("User: What are its main features?")
    result2 = await executor.execute(
        compiled, initial_input={"prompt": "What are its main features?"}, state=state
    )

    if result2.outputs.get(chatbot.id):
        response = result2.outputs[chatbot.id].data.get("response", {})
        if isinstance(response, dict) and "content" in response:
            print(f"Bot: {response['content']}\n")
        else:
            print(f"Bot: {response}\n")

    # Third interaction - bot should have full context
    print("User: How does it compare to JavaScript?")
    result3 = await executor.execute(
        compiled,
        initial_input={"prompt": "How does it compare to JavaScript?"},
        state=state,
    )

    if result3.outputs.get(chatbot.id):
        response = result3.outputs[chatbot.id].data.get("response", {})
        if isinstance(response, dict) and "content" in response:
            print(f"Bot: {response['content']}\n")
        else:
            print(f"Bot: {response}\n")

    # Show full conversation history
    print("\n=== Full Conversation History ===")
    history = await state.get("message_history", [])
    for i, msg in enumerate(history):
        role = msg["role"].capitalize()
        content = (
            msg["content"][:100] + "..."
            if len(msg["content"]) > 100
            else msg["content"]
        )
        print(f"{i + 1}. {role}: {content}")


if __name__ == "__main__":
    asyncio.run(main())
