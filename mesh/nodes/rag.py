"""RAG (Retrieval-Augmented Generation) node for document retrieval.

This node performs semantic search over vector stores to retrieve relevant
documents that can be used to enrich LLM context. It supports dependency
injection for retriever instances and formats results for downstream consumption.
"""

from typing import Any, Dict, Optional, List
import inspect

from mesh.nodes.base import BaseNode, NodeResult
from mesh.core.state import ExecutionContext
from mesh.core.events import ExecutionEvent, EventType
from mesh.utils.variables import VariableResolver


class RAGNode(BaseNode):
    """Retrieve documents from vector stores using semantic search.

    This node performs vector similarity search to find relevant documents
    and formats them for use in downstream LLM/Agent nodes. It supports
    dependency injection of retriever instances (like MosaicKnowledgeCenterService).

    The node outputs both:
    - `formatted`: Context block with <CONTEXT> tags for LLM consumption
    - `documents`: Raw array of document objects

    Example:
        >>> # In your API code:
        >>> kc_service = MosaicKnowledgeCenterService()
        >>>
        >>> rag_node = RAGNode(
        ...     id="rag_0",
        ...     query_template="{{$question}}",
        ...     top_k=5,
        ...     similarity_threshold=0.7,
        ...     file_id="uuid-123",
        ... )
        >>>
        >>> # Inject retriever after parsing
        >>> rag_node.set_retriever(kc_service)
        >>>
        >>> # Use in graph
        >>> graph.add_node("rag_0", rag_node, node_type="rag")
        >>> graph.add_edge("START", "rag_0")
        >>> graph.add_edge("rag_0", "llm_0")
    """

    def __init__(
        self,
        id: str,
        query_template: str = "{{$question}}",
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        file_id: Optional[str] = None,
        folder_uuid: Optional[str] = None,
        retriever_type: str = "postgres",
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize RAG node.

        Args:
            id: Node identifier
            query_template: Template for query (supports variable resolution, e.g. "{{$question}}")
            top_k: Number of documents to retrieve (default: 5)
            similarity_threshold: Minimum similarity score (default: 0.7)
            file_id: UUID of specific file to search (optional)
            folder_uuid: UUID of folder to search across (optional)
            retriever_type: Type of retriever ("postgres" or "chroma")
            event_mode: Event emission mode (default: "full")
                - "full": All events
                - "status_only": Only progress indicators
                - "silent": No events
            config: Additional configuration
        """
        super().__init__(id, config or {})
        self.query_template = query_template
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.file_id = file_id
        self.folder_uuid = folder_uuid
        self.retriever_type = retriever_type
        self.event_mode = event_mode

        # Retriever instance (injected later via set_retriever)
        self._retriever = None

    def set_retriever(self, retriever: Any) -> None:
        """Inject retriever instance for dependency injection pattern.

        Args:
            retriever: Retriever instance (e.g., MosaicKnowledgeCenterService)
        """
        self._retriever = retriever

    def _validate_retriever(self) -> None:
        """Validate that retriever has been injected and has required methods."""
        if self._retriever is None:
            raise RuntimeError(
                f"RAGNode '{self.id}' has no retriever set. "
                "Call set_retriever() before execution."
            )

        # Check for required methods based on search type
        if self.file_id:
            if not hasattr(self._retriever, 'search_file'):
                raise RuntimeError(
                    f"Retriever for RAGNode '{self.id}' must have 'search_file' method"
                )
        elif self.folder_uuid:
            if not hasattr(self._retriever, 'search_folder'):
                raise RuntimeError(
                    f"Retriever for RAGNode '{self.id}' must have 'search_folder' method"
                )
        else:
            raise RuntimeError(
                f"RAGNode '{self.id}' must have either file_id or folder_uuid configured"
            )

    async def _emit_event_if_enabled(
        self, context: ExecutionContext, event: ExecutionEvent
    ) -> None:
        """Emit event based on event_mode."""
        if self.event_mode == "silent":
            return

        if self.event_mode == "status_only":
            if event.type == EventType.CUSTOM_DATA:
                await context.emit_event(event)
            return

        await context.emit_event(event)

    async def _execute_impl(
        self,
        input: Any,
        context: ExecutionContext,
    ) -> NodeResult:
        """Execute RAG retrieval.

        Args:
            input: Input data (can contain query or be used in template)
            context: Execution context

        Returns:
            NodeResult with formatted context and raw documents
        """
        # Validate retriever
        self._validate_retriever()

        # Emit start event
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_START,
                node_id=self.id,
                metadata={
                    "node_type": "rag",
                    "retriever_type": self.retriever_type,
                    "top_k": self.top_k,
                },
            )
        )

        # Resolve query from template
        query = await self._resolve_query(input, context)

        # Emit retrieval start
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.CUSTOM_DATA,
                node_id=self.id,
                metadata={
                    "type": "rag_retrieval_start",
                    "query": query,
                    "top_k": self.top_k,
                },
            )
        )

        # Perform retrieval
        try:
            documents = await self._retrieve_documents(query)
        except Exception as e:
            # Emit error event
            await self._emit_event_if_enabled(
                context,
                ExecutionEvent(
                    type=EventType.NODE_ERROR,
                    node_id=self.id,
                    error=str(e),
                    metadata={
                        "node_type": "rag",
                        "query": query,
                    },
                )
            )
            raise RuntimeError(
                f"RAG retrieval failed for node '{self.id}': {str(e)}"
            ) from e

        # Format results
        formatted_context = self._format_documents(documents)

        # Emit retrieval complete
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.CUSTOM_DATA,
                node_id=self.id,
                metadata={
                    "type": "rag_retrieval_complete",
                    "num_documents": len(documents),
                    "query": query,
                },
            )
        )

        # Emit node complete
        await self._emit_event_if_enabled(
            context,
            ExecutionEvent(
                type=EventType.NODE_COMPLETE,
                node_id=self.id,
                output=f"Retrieved {len(documents)} documents",
                metadata={
                    "node_type": "rag",
                    "num_documents": len(documents),
                },
            )
        )

        return NodeResult(
            output={
                "formatted": formatted_context,
                "documents": documents,
                "query": query,
                "count": len(documents),
            },
            metadata={
                "retriever_type": self.retriever_type,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
            },
        )

    async def _resolve_query(self, input: Any, context: ExecutionContext) -> str:
        """Resolve query from template using variable resolution.

        Args:
            input: Input data
            context: Execution context

        Returns:
            Resolved query string
        """
        # Use VariableResolver to resolve template
        resolver = VariableResolver(context)
        query = await resolver.resolve(self.query_template)

        if not query:
            raise ValueError(
                f"RAGNode '{self.id}' could not resolve query from template: {self.query_template}"
            )

        return query

    async def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents using the injected retriever.

        Args:
            query: Search query

        Returns:
            List of document dictionaries
        """
        # Determine which search method to use
        if self.file_id:
            # Search within specific file
            if inspect.iscoroutinefunction(self._retriever.search_file):
                documents = await self._retriever.search_file(
                    query=query,
                    file_id=self.file_id,
                    similarity_threshold=self.similarity_threshold,
                    limit=self.top_k,
                )
            else:
                documents = self._retriever.search_file(
                    query=query,
                    file_id=self.file_id,
                    similarity_threshold=self.similarity_threshold,
                    limit=self.top_k,
                )
        elif self.folder_uuid:
            # Search across folder
            if inspect.iscoroutinefunction(self._retriever.search_folder):
                documents = await self._retriever.search_folder(
                    query=query,
                    folder_uuid=self.folder_uuid,
                    similarity_threshold=self.similarity_threshold,
                    limit=self.top_k,
                )
            else:
                documents = self._retriever.search_folder(
                    query=query,
                    folder_uuid=self.folder_uuid,
                    similarity_threshold=self.similarity_threshold,
                    limit=self.top_k,
                )
        else:
            raise RuntimeError(
                f"RAGNode '{self.id}' must have either file_id or folder_uuid configured"
            )

        return documents

    def _format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context block for LLM consumption.

        Follows the pattern from mock_router.py lines 719-728 and 747-751.

        Args:
            documents: List of document dictionaries

        Returns:
            Formatted context string with <CONTEXT> tags
        """
        if not documents:
            return ""

        context_parts = ["\n\n## Relevant Context from Knowledge Center\n\n"]

        for idx, result in enumerate(documents, 1):
            context_parts.append(f"### Source {idx}: {result.get('file_title', 'Unknown')}\n")

            if result.get('page_number'):
                context_parts.append(f"Page: {result['page_number']}\n")

            if result.get('heading'):
                context_parts.append(f"Section: {result['heading']}\n")

            if result.get('similarity') is not None:
                context_parts.append(f"Similarity: {result.get('similarity', 0):.2%}\n\n")

            context_parts.append(f"{result.get('content', '')}\n\n")
            context_parts.append("---\n\n")

        rag_context = "".join(context_parts)

        # Wrap in <CONTEXT> tags for LLM
        formatted = f"""<CONTEXT>
{rag_context}
</CONTEXT>"""

        return formatted

    def __repr__(self) -> str:
        search_target = f"file={self.file_id}" if self.file_id else f"folder={self.folder_uuid}"
        return f"RAGNode(id='{self.id}', {search_target}, top_k={self.top_k})"
