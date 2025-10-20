"""FastAPI server with agent registry for React Flow integration.

This example demonstrates the recommended pattern for integrating Mesh with
a React frontend:

1. FastAPI backend owns agent instances and configuration
2. Agents registered in NodeRegistry at startup
3. Frontend references agents by ID in React Flow JSON
4. Parser resolves agent references at runtime
5. Endpoints for listing agents and executing graphs

Architecture:
    Frontend (React Flow) → FastAPI → Mesh (orchestration) → Agents

Usage:
    python examples/05_integrations/fastapi_with_registry.py

    # List available agents
    curl http://localhost:8000/api/agents

    # List available tools
    curl http://localhost:8000/api/tools

    # Execute graph with React Flow JSON
    curl -X POST http://localhost:8000/api/execute \\
      -H "Content-Type: application/json" \\
      -d @examples/05_integrations/flows/agent_flow.json
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mesh import (
    NodeRegistry,
    ReactFlowParser,
    Executor,
    ExecutionContext,
    MemoryBackend,
)
from mesh.utils import load_env

# Load environment variables
load_env()

# Initialize FastAPI app
app = FastAPI(
    title="Mesh Agent Execution API",
    description="Execute multi-agent workflows from React Flow configurations",
    version="0.1.0",
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry (initialized at startup)
registry: Optional[NodeRegistry] = None
backend: Optional[MemoryBackend] = None


# Request/Response models
class ExecuteRequest(BaseModel):
    """Request to execute a graph."""
    flow: Dict[str, Any]  # React Flow JSON
    input: str  # User input message
    session_id: Optional[str] = None


class AgentInfo(BaseModel):
    """Information about a registered agent."""
    id: str
    name: str
    type: str  # "vel" or "openai"
    description: Optional[str] = None


class ToolInfo(BaseModel):
    """Information about a registered tool."""
    id: str
    name: str
    description: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize agents and registry at startup."""
    global registry, backend

    print("🚀 Initializing Mesh Agent Registry...")

    # Create registry
    registry = NodeRegistry()
    backend = MemoryBackend()

    # Register agents
    # Try to import and register Vel agents
    try:
        from vel import Agent as VelAgent

        # Research agent with tools
        research_agent = VelAgent(
            id="research",
            model={
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.7,
            },
        )
        registry.register_agent("research_agent", research_agent)
        print("  ✓ Registered: research_agent (Vel)")

        # Coding agent
        coding_agent = VelAgent(
            id="coder",
            model={
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.2,
            },
        )
        registry.register_agent("coding_agent", coding_agent)
        print("  ✓ Registered: coding_agent (Vel)")

    except ImportError:
        print("  ⚠️  Vel not available, skipping Vel agents")

    # Try to import and register OpenAI Agents SDK
    try:
        from agents import Agent as OpenAIAgent

        qa_agent = OpenAIAgent(
            name="QA Agent",
            instructions="You are a helpful QA assistant that answers questions clearly and concisely.",
        )
        registry.register_agent("qa_agent", qa_agent)
        print("  ✓ Registered: qa_agent (OpenAI)")

        writer_agent = OpenAIAgent(
            name="Writer",
            instructions="You are a creative writer who crafts engaging content.",
        )
        registry.register_agent("writer_agent", writer_agent)
        print("  ✓ Registered: writer_agent (OpenAI)")

    except ImportError:
        print("  ⚠️  OpenAI Agents SDK not available, skipping OpenAI agents")

    # Register tool functions
    def calculate_stats(input_data):
        """Calculate statistics for a list of numbers."""
        if isinstance(input_data, dict):
            numbers = input_data.get("numbers", [])
        elif isinstance(input_data, str):
            numbers = [float(x.strip()) for x in input_data.split(",")]
        else:
            numbers = input_data

        if not numbers:
            return {"error": "No numbers provided"}

        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "mean": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
        }

    registry.register_tool("calculate_stats", calculate_stats)
    print("  ✓ Registered: calculate_stats (Tool)")

    def sentiment_analyzer(input_data):
        """Analyze sentiment of text (simple implementation)."""
        text = str(input_data).lower()

        positive_words = ["good", "great", "excellent", "happy", "love", "wonderful"]
        negative_words = ["bad", "terrible", "sad", "hate", "awful", "horrible"]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "text_length": len(text),
        }

    registry.register_tool("sentiment_analyzer", sentiment_analyzer)
    print("  ✓ Registered: sentiment_analyzer (Tool)")

    print(f"\n✅ Registry initialized: {len(registry.list_agents())} agents, {len(registry.list_tools())} tools\n")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "registry": {
            "agents": len(registry.list_agents()) if registry else 0,
            "tools": len(registry.list_tools()) if registry else 0,
        }
    }


@app.get("/api/agents")
async def list_agents():
    """List all registered agents.

    Frontend can use this to populate agent selection dropdowns.
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    agents = []
    for agent_id in registry.list_agents():
        agent = registry.get_agent(agent_id)

        # Detect agent type
        agent_module = agent.__class__.__module__.lower()
        if "vel" in agent_module:
            agent_type = "vel"
        elif "agents" in agent_module:
            agent_type = "openai"
        else:
            agent_type = "unknown"

        agents.append(
            AgentInfo(
                id=agent_id,
                name=agent_id.replace("_", " ").title(),
                type=agent_type,
                description=f"{agent_type.upper()} agent instance"
            )
        )

    return {"agents": agents}


@app.get("/api/tools")
async def list_tools():
    """List all registered tools.

    Frontend can use this to populate tool selection dropdowns.
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    tools = []
    for tool_id in registry.list_tools():
        tool_fn = registry.get_tool(tool_id)

        tools.append(
            ToolInfo(
                id=tool_id,
                name=tool_id.replace("_", " ").title(),
                description=tool_fn.__doc__ or "No description available"
            )
        )

    return {"tools": tools}


@app.post("/api/execute")
async def execute_graph(request: ExecuteRequest):
    """Execute a graph from React Flow JSON with streaming.

    Frontend sends:
    - React Flow JSON (nodes + edges)
    - User input message
    - Optional session ID

    Backend:
    - Parses React Flow JSON using registry
    - Resolves agent/tool references
    - Executes graph with streaming
    - Returns SSE stream of events
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    # Parse React Flow JSON
    try:
        parser = ReactFlowParser(registry)
        graph = parser.parse(request.flow)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse graph: {str(e)}"
        )

    # Create execution context
    session_id = request.session_id or f"session-{datetime.now().timestamp()}"
    context = ExecutionContext(
        graph_id=request.flow.get("id", "react-flow-graph"),
        session_id=session_id,
        chat_history=[],
        variables={},
        state={},
    )

    # Execute with streaming
    executor = Executor(graph, backend)

    async def event_stream():
        """Stream execution events as SSE."""
        try:
            async for event in executor.execute(request.input, context):
                # Convert event to SSE format
                event_data = {
                    "type": event.type,
                    "node_id": event.node_id,
                    "content": event.content,
                    "output": event.output,
                    "error": event.error,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "metadata": event.metadata,
                }

                # Filter out None values
                event_data = {k: v for k, v in event_data.items() if v is not None}

                yield f"data: {json.dumps(event_data)}\n\n"

        except Exception as e:
            error_event = {
                "type": "execution_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/api/execute-sync")
async def execute_graph_sync(request: ExecuteRequest):
    """Execute a graph and return complete result (non-streaming).

    Useful for simple use cases where streaming is not needed.
    """
    if not registry:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    # Parse and execute
    try:
        parser = ReactFlowParser(registry)
        graph = parser.parse(request.flow)

        session_id = request.session_id or f"session-{datetime.now().timestamp()}"
        context = ExecutionContext(
            graph_id=request.flow.get("id", "react-flow-graph"),
            session_id=session_id,
            chat_history=[],
            variables={},
            state={},
        )

        executor = Executor(graph, backend)

        # Collect all events
        events = []
        final_output = None

        async for event in executor.execute(request.input, context):
            events.append({
                "type": event.type,
                "node_id": event.node_id,
                "content": event.content,
                "metadata": event.metadata,
            })

            if event.type == "execution_complete":
                final_output = event.output

        return {
            "success": True,
            "output": final_output,
            "events": events,
            "session_id": session_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Execution failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*80)
    print("🚀 Starting Mesh Agent Execution API")
    print("="*80)
    print("\nEndpoints:")
    print("  • GET  /health              - Health check")
    print("  • GET  /api/agents          - List registered agents")
    print("  • GET  /api/tools           - List registered tools")
    print("  • POST /api/execute         - Execute graph (streaming)")
    print("  • POST /api/execute-sync    - Execute graph (sync)")
    print("\nDocs:")
    print("  • http://localhost:8000/docs")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
