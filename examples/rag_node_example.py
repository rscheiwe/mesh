"""Example: RAG Node with MosaicKnowledgeCenterService

This example demonstrates how to use RAGNode with dependency injection
for document retrieval in a graph workflow.

Pattern:
1. Parse React Flow JSON to create graph structure
2. Inject retriever instance into RAG nodes
3. Execute graph with RAG-enhanced context
"""

import asyncio
from typing import List, Dict, Any


# Mock retriever for demonstration (replace with actual MosaicKnowledgeCenterService)
class MockRetriever:
    """Mock retriever that simulates MosaicKnowledgeCenterService."""

    async def search_file(
        self,
        query: str,
        file_id: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Mock search_file method."""
        print(f"[MockRetriever] Searching file {file_id} for: {query}")
        return [
            {
                "file_title": "Climate Guide.pdf",
                "content": "Cats with short coats and larger ears tend to handle heat better. Breeds like Abyssinian, Siamese, and Bengal are well-suited for hot climates.",
                "page_number": 5,
                "heading": "Cat Breeds for Hot Weather",
                "similarity": 0.89,
            },
            {
                "file_title": "Climate Guide.pdf",
                "content": "Provide plenty of water and shade for cats in hot weather. Consider cooling mats and avoid exercising during peak heat hours.",
                "page_number": 7,
                "heading": "Heat Safety Tips",
                "similarity": 0.76,
            },
        ]

    async def search_folder(
        self,
        query: str,
        folder_uuid: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Mock search_folder method."""
        print(f"[MockRetriever] Searching folder {folder_uuid} for: {query}")
        return [
            {
                "file_title": "Pet Care Guide.pdf",
                "content": "Short-haired cats like Siamese and Devon Rex are ideal for hot climates due to their minimal coat.",
                "page_number": 12,
                "heading": "Breed Selection",
                "similarity": 0.85,
            }
        ]


async def example_direct_usage():
    """Example 1: Direct RAGNode usage without React Flow."""
    from mesh.nodes.rag import RAGNode
    from mesh.core.state import ExecutionContext

    print("\n=== Example 1: Direct RAGNode Usage ===\n")

    # Create RAG node
    rag_node = RAGNode(
        id="rag_0",
        query_template="{{$question}}",
        top_k=5,
        similarity_threshold=0.7,
        file_id="file-uuid-123",
    )

    # Inject retriever (dependency injection pattern)
    retriever = MockRetriever()
    rag_node.set_retriever(retriever)

    # Create execution context
    context = ExecutionContext(graph_id="example", session_id="test")
    context.variables["question"] = "What cats are good in hot weather?"

    # Simulate START node output (normally done by executor)
    context.executed_data.append({
        "node_id": "START",
        "output": "What cats are good in hot weather?"
    })

    # Execute node
    result = await rag_node.execute(
        input="What cats are good in hot weather?", context=context
    )

    print(f"Retrieved {result.output['num_results']} documents")
    print(f"\nFormatted context preview:")
    print(result.output["formatted"][:300] + "...")


async def example_react_flow_graph():
    """Example 2: RAGNode in React Flow graph with dependency injection."""
    from mesh.parsers.react_flow import ReactFlowParser
    from mesh.utils.registry import NodeRegistry
    from mesh.nodes.rag import RAGNode

    print("\n\n=== Example 2: RAGNode in React Flow Graph ===\n")

    # React Flow JSON from frontend
    react_flow_json = {
        "nodes": [
            {"id": "START", "type": "startAgentflow", "data": {}},
            {
                "id": "rag_0",
                "type": "ragAgentflow",
                "data": {
                    "inputs": {
                        "queryTemplate": "{{$question}}",
                        "topK": 5,
                        "similarityThreshold": 0.7,
                        "fileId": "file-uuid-456",
                        "retrieverType": "postgres",
                    }
                },
            },
        ],
        "edges": [
            {"source": "START", "target": "rag_0"},
        ],
    }

    # Parse graph
    registry = NodeRegistry()
    parser = ReactFlowParser(registry)
    graph = parser.parse(react_flow_json)

    print("Graph parsed successfully!")
    print(f"Nodes: {list(graph.nodes.keys())}")

    # Inject retriever into RAG nodes (dependency injection)
    retriever = MockRetriever()
    for node_id, node in graph.nodes.items():
        if isinstance(node, RAGNode):
            print(f"Injecting retriever into {node_id}")
            node.set_retriever(retriever)

    print("\nGraph ready for execution!")
    print("Example usage in API:")
    print("""
    # In your taboolabot-api:
    kc_service = MosaicKnowledgeCenterService()

    for node in graph.nodes.values():
        if isinstance(node, RAGNode):
            node.set_retriever(kc_service)

    result = await graph.run(input={"question": "What cats are good in hot weather?"})
    """)


async def example_formatted_output():
    """Example 3: Show formatted output structure."""
    from mesh.nodes.rag import RAGNode
    from mesh.core.state import ExecutionContext

    print("\n\n=== Example 3: Formatted Output Structure ===\n")

    rag_node = RAGNode(
        id="rag_0",
        query_template="{{$question}}",
        top_k=2,
        file_id="file-uuid-789",
    )

    retriever = MockRetriever()
    rag_node.set_retriever(retriever)

    context = ExecutionContext(graph_id="example", session_id="test")

    # Simulate START node output
    context.executed_data.append({
        "node_id": "START",
        "output": "cat breeds for heat"
    })

    result = await rag_node.execute(input="cat breeds for heat", context=context)

    print("Output structure:")
    print(f"  - formatted: {len(result.output['formatted'])} chars")
    print(f"  - documents: {len(result.output['documents'])} items")
    print(f"  - query: {result.output['query']}")
    print(f"  - num_results: {result.output['num_results']}")

    print("\nFormatted output (for LLM):")
    print(result.output["formatted"])

    print("\n\nRaw documents (for debugging):")
    for idx, doc in enumerate(result.output["documents"], 1):
        print(f"\n  Document {idx}:")
        print(f"    Title: {doc['file_title']}")
        print(f"    Similarity: {doc['similarity']:.2%}")
        print(f"    Content preview: {doc['content'][:80]}...")


async def main():
    """Run all examples."""
    await example_direct_usage()
    await example_react_flow_graph()
    await example_formatted_output()


if __name__ == "__main__":
    asyncio.run(main())
