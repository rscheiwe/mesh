"""FastAPI server example with streaming support.

This example demonstrates how to use Mesh with FastAPI to create
an API server with both streaming and non-streaming endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid

from mesh import (
    StateGraph,
    Executor,
    ExecutionContext,
    NodeRegistry,
    MemoryBackend,
    SQLiteBackend,
    ReactFlowParser,
)
from mesh.streaming.sse import SSEAdapter
from mesh.utils import load_env

# Load environment variables from .env file
load_env()

# Initialize FastAPI app
app = FastAPI(
    title="Mesh Agent Graph API",
    description="Execute agent workflows as graphs with streaming support",
    version="0.1.0",
)

# Global state
graphs: Dict[str, Any] = {}
backend = SQLiteBackend("mesh_api_state.db")
registry = NodeRegistry()


# Request/Response models
class ExecuteRequest(BaseModel):
    """Request to execute a graph."""

    graph_id: str
    input: str
    session_id: Optional[str] = None
    variables: Dict[str, Any] = {}


class ExecuteResponse(BaseModel):
    """Response from graph execution."""

    session_id: str
    output: Any
    execution_id: str


class GraphCreateRequest(BaseModel):
    """Request to create graph from React Flow JSON."""

    flow_json: Dict[str, Any]
    graph_id: Optional[str] = None


class GraphCreateResponse(BaseModel):
    """Response from graph creation."""

    graph_id: str
    nodes: int
    edges: int


# Endpoints
@app.post("/execute", response_model=ExecuteResponse)
async def execute_graph(request: ExecuteRequest):
    """Execute graph and return final result (non-streaming).

    This endpoint executes a graph and waits for completion before
    returning the final result.
    """
    if request.graph_id not in graphs:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph = graphs[request.graph_id]
    executor = Executor(graph, backend)

    session_id = request.session_id or str(uuid.uuid4())
    context = ExecutionContext(
        graph_id=request.graph_id,
        session_id=session_id,
        chat_history=[],
        variables=request.variables,
        state=await backend.load(session_id) or {},
    )

    # Collect final result
    final_output = None
    async for event in executor.execute(request.input, context):
        if event.type == "execution_complete":
            final_output = event.output

    return ExecuteResponse(
        session_id=session_id,
        output=final_output,
        execution_id=context.trace_id,
    )


@app.post("/execute/stream")
async def execute_graph_stream(request: ExecuteRequest):
    """Execute graph with SSE streaming.

    This endpoint streams execution events in real-time using
    Server-Sent Events (SSE).
    """
    if request.graph_id not in graphs:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph = graphs[request.graph_id]
    executor = Executor(graph, backend)

    session_id = request.session_id or str(uuid.uuid4())
    context = ExecutionContext(
        graph_id=request.graph_id,
        session_id=session_id,
        chat_history=[],
        variables=request.variables,
        state=await backend.load(session_id) or {},
    )

    adapter = SSEAdapter()
    return adapter.to_streaming_response(executor.execute(request.input, context))


@app.post("/graphs", response_model=GraphCreateResponse)
async def create_graph(request: GraphCreateRequest):
    """Create graph from React Flow JSON.

    This endpoint parses React Flow JSON (Flowise-compatible)
    and creates an executable graph.
    """
    parser = ReactFlowParser(registry)

    try:
        graph = parser.parse(request.flow_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse graph: {str(e)}")

    graph_id = request.graph_id or str(uuid.uuid4())
    graphs[graph_id] = graph

    return GraphCreateResponse(
        graph_id=graph_id,
        nodes=len(graph.nodes),
        edges=len(graph.edges),
    )


@app.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    """Get graph information."""
    if graph_id not in graphs:
        raise HTTPException(status_code=404, detail="Graph not found")

    graph = graphs[graph_id]
    return {
        "graph_id": graph_id,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "starting_nodes": graph.starting_nodes,
        "ending_nodes": graph.ending_nodes,
    }


@app.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    """Delete a graph."""
    if graph_id not in graphs:
        raise HTTPException(status_code=404, detail="Graph not found")

    del graphs[graph_id]
    return {"message": "Graph deleted successfully"}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session state."""
    state = await backend.load(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "state": state}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session state."""
    await backend.delete(session_id)
    return {"message": "Session deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graphs": len(graphs),
        "backend": str(backend),
    }


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Mesh Agent Graph API",
        "version": "0.1.0",
        "endpoints": {
            "execute": "POST /execute - Execute graph (non-streaming)",
            "execute_stream": "POST /execute/stream - Execute graph (streaming)",
            "create_graph": "POST /graphs - Create graph from JSON",
            "get_graph": "GET /graphs/{graph_id} - Get graph info",
            "delete_graph": "DELETE /graphs/{graph_id} - Delete graph",
        },
    }


# Run server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
