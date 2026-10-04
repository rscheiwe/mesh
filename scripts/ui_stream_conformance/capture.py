"""Capture mesh UI-stream transcripts (.sse) for the example-DAG scenarios.

uv run python scripts/ui_stream_conformance/capture.py OUT_DIR
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.dags import brief_to_recommendation as dag
from examples.dags.scripted import json_turn, scripted_agent, text_turn
from mesh import ExecutionContext, Executor, MemoryBackend, StateGraph, UIMessageStreamAdapter
from tests.dags.test_dag_a import ACME_BRIEF, ACME_SLOTS

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


async def dag_a(name, extract_script, explain_script, payload, state=None, patch=None):
    if patch:
        setattr(dag, *patch)
    extract = scripted_agent("extract", extract_script, output_type=dag.Slots)
    explain = scripted_agent("explain", explain_script)
    graph = dag.build_graph(extract, explain)
    adapter = UIMessageStreamAdapter(message_id=f"msg-{name}", graph=graph)
    frames = [
        f
        async for f in adapter.sse(
            Executor(graph, MemoryBackend()).execute(payload, dag.new_context(state))
        )
    ]
    (OUT / f"{name}.sse").write_text("".join(frames))


async def interrupt():
    g = StateGraph()
    g.add_node("draft", lambda: {"ok": 1}, node_type="tool")
    g.add_node("publish", lambda: {"ok": 2}, node_type="tool")
    g.add_edge("draft", "publish")
    g.set_entry_point("draft")
    g.set_interrupt_before("publish")
    frames = [
        f
        async for f in UIMessageStreamAdapter().sse(
            Executor(g.compile(), MemoryBackend()).execute(
                "go", ExecutionContext(graph_id="g", session_id="s")
            )
        )
    ]
    (OUT / "interrupt.sse").write_text("".join(frames))


async def main():
    original_scale = dag.scale
    await dag_a(
        "dag_a_recommend",
        [json_turn(ACME_SLOTS)],
        [text_turn("Commerce Connect leads on CTR.")],
        {"text": ACME_BRIEF},
    )
    await dag_a(
        "dag_a_form",
        [json_turn({**ACME_SLOTS, "vertical": None})],
        [],
        {"text": "A US app, $25,000."},
    )
    await dag_a(
        "dag_a_form_submission",
        [],
        [text_turn("Done.")],
        {
            "text": "Vertical: Retail",
            "metadata": {"type": "tool_form_submission", "parameters": {"vertical": "Retail"}},
        },
        state={"slots": {"kpi": "ctr", "geo": "US", "budget": 30000}},
    )
    await dag_a(
        "dag_d_tool_error",
        [json_turn(ACME_SLOTS)],
        [text_turn("x")],
        {"text": ACME_BRIEF},
        patch=("scale", lambda state: 1 / 0),
    )
    dag.scale = original_scale
    await dag_a(
        "dag_d_model_error",
        [json_turn(ACME_SLOTS)],
        [[RuntimeError("model provider returned 503")]],
        {"text": ACME_BRIEF},
    )
    await interrupt()
    print(sorted(p.name for p in OUT.iterdir()))


asyncio.run(main())
