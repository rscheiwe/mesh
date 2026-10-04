"""Vercel AI SDK UI message stream adapter.

Turns a mesh execution stream into the chunk sequence a ``useChat`` client
(``ai`` v5/v6) accepts: one ``start`` and one ``finish`` per run, only
protocol chunk types and keys, block ids namespaced per node, mesh lifecycle
as transient ``data-mesh-node`` parts, and errors reported once.

Example with FastAPI:
    >>> adapter = UIMessageStreamAdapter(graph=graph)
    >>> return adapter.to_streaming_response(executor.execute(payload, context))
"""

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from mesh.core.events import EventType, ExecutionEvent

logger = logging.getLogger(__name__)

STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "x-vercel-ai-ui-message-stream": "v1",
}

# Allowed keys per AI SDK UI message chunk type (ai v5.0.221+ / v6 schema)
_PROVIDER = {"providerMetadata"}
_TOOL_FLAGS = {"providerExecuted", "dynamic"}
CHUNK_KEYS: dict[str, set[str]] = {
    "abort": set(),
    "start-step": set(),
    "finish-step": set(),
    "text-start": {"id"} | _PROVIDER,
    "text-delta": {"id", "delta"} | _PROVIDER,
    "text-end": {"id"} | _PROVIDER,
    "reasoning-start": {"id"} | _PROVIDER,
    "reasoning-delta": {"id", "delta"} | _PROVIDER,
    "reasoning-end": {"id"} | _PROVIDER,
    "tool-input-start": {"toolCallId", "toolName"} | _TOOL_FLAGS,
    "tool-input-delta": {"toolCallId", "inputTextDelta"},
    "tool-input-available": {"toolCallId", "toolName", "input"} | _TOOL_FLAGS | _PROVIDER,
    "tool-input-error": {"toolCallId", "toolName", "input", "errorText"} | _TOOL_FLAGS,
    "tool-output-available": {"toolCallId", "output", "preliminary"} | _TOOL_FLAGS,
    "tool-output-error": {"toolCallId", "errorText"} | _TOOL_FLAGS,
    "source-url": {"sourceId", "url", "title"} | _PROVIDER,
    "source-document": {"sourceId", "mediaType", "title", "filename"} | _PROVIDER,
    "file": {"url", "mediaType"},
}
DATA_KEYS = {"id", "data", "transient"}
_BLOCK_TYPES = ("text", "reasoning")

# Mesh/vel chunks that never reach the client
_DROPPED = {
    "start",  # per node; the adapter sends one per run
    "finish",  # per node; the adapter sends one per run
    "finish-message",
    "response-metadata",  # not an AI SDK chunk; usage is on NodeResult.metadata
    "data-custom",  # generic wrapper with arbitrary metadata
    "data-mesh-node-start",  # replaced by data-mesh-node
    "data-mesh-node-complete",
}
_LIFECYCLE_STATUS = {
    EventType.NODE_START: "running",
    EventType.NODE_COMPLETE: "complete",
    EventType.NODE_ERROR: "error",
}


class UIMessageStreamAdapter:
    """Adapt mesh ExecutionEvents to the AI SDK UI message stream protocol.

    Args:
        message_id: Optional id for the assistant message (sent on ``start``).
        graph: The executing graph. When given, text from agent nodes whose
            agent has an ``output_type`` is hidden (vel streams structured
            output as raw JSON text; the ``data-object-*`` parts still pass).
        include_node_events: Emit transient ``data-mesh-node`` progress parts.
        include_outputs: Attach node outputs to ``data-mesh-node`` parts. Off by
            default so node outputs and state never reach the client.
        error_text: Maps a run failure to the ``errorText`` the client sees. The
            default is ``str(exc)``, which includes internal node names; pass a
            function returning a user-safe message in production.
    """

    def __init__(
        self,
        *,
        message_id: str | None = None,
        graph: Any = None,
        include_node_events: bool = True,
        include_outputs: bool = False,
        error_text: Callable[[BaseException], str] = str,
    ):
        self.message_id = message_id
        self.include_node_events = include_node_events
        self.include_outputs = include_outputs
        self.error_text = error_text
        self.hidden_text_nodes = _structured_output_nodes(graph)

    async def chunks(self, events: AsyncIterator[ExecutionEvent]) -> AsyncIterator[dict]:
        """Yield protocol chunks for one run, from ``start`` to ``finish``."""
        run = _RunState()
        start: dict[str, Any] = {"type": "start"}
        if self.message_id:
            start["messageId"] = self.message_id
        yield start
        try:
            async for event in events:
                for chunk in self._translate(event, run):
                    yield chunk
        except Exception as exc:  # node failures arrive as NodeExecutionError
            logger.debug("Run failed, closing stream: %s", exc)
            if not run.error_sent:
                run.error_sent = True
                yield {"type": "error", "errorText": self.error_text(exc)}
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()
        for chunk in run.close_open_blocks():
            yield chunk
        yield {"type": "finish"}

    async def sse(self, events: AsyncIterator[ExecutionEvent]) -> AsyncIterator[str]:
        """Yield SSE frames (``data: <json>``), ending with ``data: [DONE]``."""
        async for chunk in self.chunks(events):
            yield f"data: {json.dumps(chunk, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    def to_streaming_response(self, events: AsyncIterator[ExecutionEvent]) -> Any:
        """FastAPI/Starlette StreamingResponse with the AI SDK stream headers."""
        try:
            from fastapi.responses import StreamingResponse
        except ImportError as exc:
            raise ImportError("FastAPI not installed. Install with: uv add fastapi") from exc
        return StreamingResponse(
            self.sse(events), media_type="text/event-stream", headers=STREAM_HEADERS
        )

    def _translate(self, event: ExecutionEvent, run: "_RunState") -> list[dict]:
        if event.type in _LIFECYCLE_STATUS:
            return self._lifecycle_chunk(event, run)
        if event.type == EventType.EXECUTION_ERROR:
            return run.error(self.error_text(RuntimeError(event.error or "Execution failed")))
        if event.type in (EventType.INTERRUPT, EventType.EXECUTION_COMPLETE):
            return _pause_chunk(event)
        if event.type == EventType.EXECUTION_START:
            return []

        raw = event.to_dict()
        chunk_type = raw.get("type", "")
        if chunk_type in _DROPPED:
            return []
        if chunk_type == "error":
            text = raw.get("errorText") or event.error or "Error"
            return run.error(self.error_text(RuntimeError(text)))
        if chunk_type == "source":
            return _source_chunks(raw)
        if chunk_type.startswith("data-"):
            return [_pick(raw, {"type"} | DATA_KEYS)]
        if chunk_type not in CHUNK_KEYS:
            logger.debug("Dropping non-protocol chunk type: %s", chunk_type)
            return []
        if event.node_id in self.hidden_text_nodes and chunk_type.startswith("text-"):
            return []
        return run.track(_namespace(_pick(raw, {"type"} | CHUNK_KEYS[chunk_type]), event))

    def _lifecycle_chunk(self, event: ExecutionEvent, run: "_RunState") -> list[dict]:
        if not self.include_node_events or not event.node_id:
            return []
        status = _LIFECYCLE_STATUS[event.type]
        if run.node_status.get(event.node_id) == status:
            return []  # agent nodes report "running" twice
        run.node_status[event.node_id] = status
        data: dict[str, Any] = {
            "node_id": event.node_id,
            "node_type": (event.metadata or {}).get("node_type"),
            "status": status,
        }
        if status == "error":
            data["errorText"] = event.error
        if self.include_outputs and event.output is not None:
            data["output"] = event.output
        return [
            {
                "type": "data-mesh-node",
                "id": f"node-{event.node_id}",
                "data": data,
                "transient": True,
            }
        ]


class _RunState:
    """Open blocks and error bookkeeping for one run."""

    def __init__(self) -> None:
        self.open_blocks: dict[str, str] = {}  # block id -> "text" | "reasoning"
        self.step_open = False
        self.error_sent = False
        self.node_status: dict[str, str] = {}

    def track(self, chunk: dict) -> list[dict]:
        chunk_type = chunk["type"]
        if chunk_type == "start-step":
            prefix = [{"type": "finish-step"}] if self.step_open else []
            self.step_open = True
            return [*prefix, chunk]
        if chunk_type == "finish-step":
            if not self.step_open:
                return []
            self.step_open = False
            return [chunk]
        kind, _, phase = chunk_type.partition("-")
        if kind in _BLOCK_TYPES and phase == "start":
            self.open_blocks[chunk["id"]] = kind
        elif kind in _BLOCK_TYPES and phase == "end":
            self.open_blocks.pop(chunk["id"], None)
        return [chunk]

    def error(self, text: str) -> list[dict]:
        if self.error_sent:
            return []
        self.error_sent = True
        return [{"type": "error", "errorText": text}]

    def close_open_blocks(self) -> list[dict]:
        chunks = [
            {"type": f"{kind}-end", "id": block_id} for block_id, kind in self.open_blocks.items()
        ]
        self.open_blocks.clear()
        if self.step_open:
            chunks.append({"type": "finish-step"})
            self.step_open = False
        return chunks


def _structured_output_nodes(graph: Any) -> set[str]:
    nodes = getattr(graph, "nodes", None) or {}
    hidden = set()
    for node_id, node in nodes.items():
        agent = getattr(node, "agent", None)
        if agent is not None and getattr(agent, "output_type", None) is not None:
            hidden.add(node_id)
    return hidden


def _pick(raw: dict, keys: set[str]) -> dict:
    return {k: v for k, v in raw.items() if k in keys and v is not None}


def _namespace(chunk: dict, event: ExecutionEvent) -> dict:
    """Prefix text/reasoning block ids with the node id: vel reuses ids across
    agents, and parallel branches would otherwise interleave into one part."""
    if event.node_id and chunk["type"].split("-")[0] in _BLOCK_TYPES and "id" in chunk:
        chunk["id"] = f"{event.node_id}:{chunk['id']}"
    return chunk


def _source_chunks(raw: dict) -> list[dict]:
    """Vel's ``source`` event carries a list; the protocol wants one source-url each."""
    chunks = []
    for index, source in enumerate(raw.get("sources") or []):
        url = source.get("url")
        if not url:
            continue
        chunk = {
            "type": "source-url",
            "sourceId": source.get("id") or f"source-{index}",
            "url": url,
        }
        if source.get("title"):
            chunk["title"] = source["title"]
        chunks.append(chunk)
    return chunks


def _pause_chunk(event: ExecutionEvent) -> list[dict]:
    """Interrupt/approval pauses, as a data part without any execution state."""
    metadata = event.metadata or {}
    if event.type == EventType.INTERRUPT:
        data = {
            "status": "interrupted",
            "interrupt_id": metadata.get("interrupt_id"),
            "node_id": event.node_id,
            "position": metadata.get("position"),
        }
    elif metadata.get("status") in ("waiting_for_approval", "waiting_for_interrupt"):
        if metadata["status"] == "waiting_for_interrupt":
            return []  # already reported by the INTERRUPT event
        data = {
            "status": "waiting_for_approval",
            "approval_id": metadata.get("approval_id"),
        }
    else:
        return []
    return [{"type": "data-mesh-interrupt", "data": data}]
