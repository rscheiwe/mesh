"""Advanced Event Types Demo

This example demonstrates the new event types added to Mesh:
- Reasoning events (o1/o3/Claude Extended Thinking)
- Response metadata (usage tracking)
- Source citations (Gemini grounding)
- File attachments (multi-modal)
- Custom data events
- Multi-step agent tracking

Note: This is a demonstration using mock events. For real usage,
you would use actual agents that emit these events (e.g., o1-preview for reasoning).
"""

import asyncio
from mesh import StateGraph, Executor, ExecutionContext, MemoryBackend
from mesh.nodes import AgentNode
from mesh.core.events import EventType


class MockAdvancedAgent:
    """Mock agent that emits all event types for demonstration."""

    def __init__(self, scenario="reasoning"):
        self.id = "advanced-agent"
        self.__module__ = "vel.agent"
        self.scenario = scenario

    async def run_stream(self, input, session_id):
        """Stream events based on scenario."""
        if self.scenario == "reasoning":
            # Simulate o1/o3 reasoning model
            yield {"type": "start"}
            yield {"type": "reasoning-start", "id": "r1"}
            yield {"type": "reasoning-delta", "id": "r1", "delta": "Let me analyze this problem step by step..."}
            yield {"type": "reasoning-delta", "id": "r1", "delta": " First, I'll consider the context..."}
            yield {"type": "reasoning-delta", "id": "r1", "delta": " Then, I'll evaluate possible solutions."}
            yield {"type": "reasoning-end", "id": "r1"}
            yield {"type": "text-start", "id": "t1"}
            yield {"type": "text-delta", "id": "t1", "delta": "Based on my analysis, "}
            yield {"type": "text-delta", "id": "t1", "delta": "the best approach is to "}
            yield {"type": "text-delta", "id": "t1", "delta": "proceed with option A."}
            yield {"type": "text-end", "id": "t1"}
            yield {
                "type": "response-metadata",
                "id": "resp-123",
                "modelId": "o1-preview",
                "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            }
            yield {"type": "finish-message", "finishReason": "stop"}

        elif self.scenario == "sources":
            # Simulate Gemini with grounding
            yield {"type": "start"}
            yield {"type": "text-start", "id": "t1"}
            yield {"type": "text-delta", "id": "t1", "delta": "According to recent research, "}
            yield {"type": "text-delta", "id": "t1", "delta": "AI models have improved significantly."}
            yield {"type": "text-end", "id": "t1"}
            yield {
                "type": "source",
                "sources": [
                    {"url": "https://arxiv.org/abs/2301.00001", "title": "Recent Advances in AI"},
                    {"url": "https://research.google/pubs/pub12345", "title": "Google AI Research"},
                ],
            }
            yield {
                "type": "response-metadata",
                "id": "resp-456",
                "modelId": "gemini-pro",
                "usage": {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70},
            }
            yield {"type": "finish-message", "finishReason": "stop"}

        elif self.scenario == "multimodal":
            # Simulate multi-modal response with file
            yield {"type": "start"}
            yield {"type": "text-start", "id": "t1"}
            yield {"type": "text-delta", "id": "t1", "delta": "Here's a diagram showing the architecture:"}
            yield {"type": "text-end", "id": "t1"}
            yield {
                "type": "file",
                "name": "architecture_diagram.png",
                "mimeType": "image/png",
                "content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            }
            yield {"type": "finish-message", "finishReason": "stop"}

        elif self.scenario == "custom":
            # Simulate custom progress events
            yield {"type": "start"}
            yield {"type": "data-progress", "data": {"percent": 0}, "transient": True}
            yield {"type": "text-start", "id": "t1"}
            yield {"type": "data-progress", "data": {"percent": 25}, "transient": True}
            yield {"type": "text-delta", "id": "t1", "delta": "Processing your request..."}
            yield {"type": "data-progress", "data": {"percent": 50}, "transient": True}
            yield {"type": "text-delta", "id": "t1", "delta": " Almost done..."}
            yield {"type": "data-progress", "data": {"percent": 75}, "transient": True}
            yield {"type": "text-delta", "id": "t1", "delta": " Complete!"}
            yield {"type": "data-progress", "data": {"percent": 100}, "transient": True}
            yield {"type": "text-end", "id": "t1"}
            yield {"type": "finish-message", "finishReason": "stop"}


async def demo_reasoning_events():
    """Demonstrate reasoning events from o1/o3 models."""
    print("\n" + "=" * 70)
    print("DEMO 1: Reasoning Events (o1/o3/Claude Extended Thinking)")
    print("=" * 70)

    graph = StateGraph()
    agent = MockAdvancedAgent(scenario="reasoning")
    graph.add_node("agent", agent, node_type="agent")
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="reasoning-demo",
        session_id="demo-session-1",
        chat_history=[],
        variables={},
        state={},
    )

    reasoning_content = ""
    response_content = ""
    usage_info = {}

    async for event in executor.execute("Analyze this problem", context):
        if event.type == EventType.REASONING_START:
            print("\n🤔 [Thinking...]")

        elif event.type == EventType.REASONING_TOKEN:
            reasoning_content += event.content
            print(f"   {event.content}", end="", flush=True)

        elif event.type == EventType.REASONING_END:
            print("\n✅ [Done thinking]\n")

        elif event.type == EventType.TOKEN:
            response_content += event.content
            print(event.content, end="", flush=True)

        elif event.type == EventType.RESPONSE_METADATA:
            usage_info = event.metadata.get("usage", {})
            model = event.metadata.get("model_id", "")

    print(f"\n\n📊 Summary:")
    print(f"   Model: {model}")
    print(f"   Reasoning: {len(reasoning_content)} chars")
    print(f"   Response: {len(response_content)} chars")
    print(f"   Tokens: {usage_info.get('total_tokens', 0)}")


async def demo_source_citations():
    """Demonstrate source citation events from Gemini grounding."""
    print("\n" + "=" * 70)
    print("DEMO 2: Source Citations (Gemini Grounding)")
    print("=" * 70)

    graph = StateGraph()
    agent = MockAdvancedAgent(scenario="sources")
    graph.add_node("agent", agent, node_type="agent")
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="sources-demo",
        session_id="demo-session-2",
        chat_history=[],
        variables={},
        state={},
    )

    citations = []
    usage_info = {}

    async for event in executor.execute("Tell me about AI advances", context):
        if event.type == EventType.TOKEN:
            print(event.content, end="", flush=True)

        elif event.type == EventType.SOURCE:
            sources = event.metadata.get("sources", [])
            for source in sources:
                citations.append(source)
                print(f"\n\n📚 Source: {source.get('title')}")
                print(f"   URL: {source.get('url')}")

        elif event.type == EventType.RESPONSE_METADATA:
            usage_info = event.metadata.get("usage", {})
            model = event.metadata.get("model_id", "")

    print(f"\n\n📊 Summary:")
    print(f"   Model: {model}")
    print(f"   Citations: {len(citations)}")
    print(f"   Tokens: {usage_info.get('total_tokens', 0)}")


async def demo_multimodal_files():
    """Demonstrate file attachment events."""
    print("\n" + "=" * 70)
    print("DEMO 3: Multi-Modal File Attachments")
    print("=" * 70)

    graph = StateGraph()
    agent = MockAdvancedAgent(scenario="multimodal")
    graph.add_node("agent", agent, node_type="agent")
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="multimodal-demo",
        session_id="demo-session-3",
        chat_history=[],
        variables={},
        state={},
    )

    files_received = []

    async for event in executor.execute("Show me the architecture", context):
        if event.type == EventType.TOKEN:
            print(event.content, end="", flush=True)

        elif event.type == EventType.FILE:
            filename = event.metadata.get("name")
            mime_type = event.metadata.get("mime_type")
            content = event.metadata.get("content")

            files_received.append(
                {"name": filename, "type": mime_type, "size": len(content) if content else 0}
            )

            print(f"\n\n📎 File Attached: {filename}")
            print(f"   Type: {mime_type}")
            print(f"   Size: {len(content) if content else 0} bytes")

    print(f"\n\n📊 Summary:")
    print(f"   Files received: {len(files_received)}")


async def demo_custom_progress():
    """Demonstrate custom data events for progress tracking."""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Progress Events")
    print("=" * 70)

    graph = StateGraph()
    agent = MockAdvancedAgent(scenario="custom")
    graph.add_node("agent", agent, node_type="agent")
    graph.add_edge("START", "agent")
    graph.set_entry_point("agent")

    compiled = graph.compile()
    executor = Executor(compiled, MemoryBackend())
    context = ExecutionContext(
        graph_id="custom-demo",
        session_id="demo-session-4",
        chat_history=[],
        variables={},
        state={},
    )

    progress_updates = []

    async for event in executor.execute("Process my request", context):
        if event.type == EventType.CUSTOM_DATA:
            data_type = event.metadata.get("data_type")
            data = event.content

            if data_type == "data-progress":
                percent = data.get("percent", 0)
                progress_updates.append(percent)
                print(f"\r⏳ Progress: {percent}%", end="", flush=True)

        elif event.type == EventType.TOKEN:
            print(f"\n{event.content}", end="", flush=True)

    print(f"\n\n📊 Summary:")
    print(f"   Progress updates: {len(progress_updates)}")
    print(f"   Final: {progress_updates[-1] if progress_updates else 0}%")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MESH ADVANCED EVENTS DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the new event types available in Mesh:")
    print("- Reasoning events (o1/o3/Claude)")
    print("- Source citations (Gemini grounding)")
    print("- File attachments (multi-modal)")
    print("- Custom data events (progress, RLM, etc.)")

    await demo_reasoning_events()
    await demo_source_citations()
    await demo_multimodal_files()
    await demo_custom_progress()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    print("\nThese event types enable:")
    print("✅ Transparent reasoning (o1/o3 thinking)")
    print("✅ Usage/cost tracking")
    print("✅ Citation display (Gemini)")
    print("✅ Multi-modal content")
    print("✅ Custom progress indicators")
    print("✅ Extensibility for new features")


if __name__ == "__main__":
    asyncio.run(main())
