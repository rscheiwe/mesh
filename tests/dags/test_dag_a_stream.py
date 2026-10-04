"""DAG A and DAG D through UIMessageStreamAdapter: what a useChat client receives."""

import pytest

from examples.dags import brief_to_recommendation as dag
from examples.dags.scripted import json_turn, scripted_agent, text_turn
from mesh import Executor, MemoryBackend, UIMessageStreamAdapter
from tests.dags.conftest import assert_valid_ui_stream, ui_chunks
from tests.dags.test_dag_a import ACME_BRIEF, ACME_SLOTS

# State the pipeline keeps server-side; none of it may reach the client.
STATE_KEYS = ('"benchmarks": {', '"candidates": {', '"effective_capacity":', '"pending_form":')


async def _stream(extract_script, explain_script, payload, state=None, **adapter_kwargs):
    extract = scripted_agent("extract", extract_script, output_type=dag.Slots)
    explain = scripted_agent("explain", explain_script)
    graph = dag.build_graph(extract, explain)
    adapter = UIMessageStreamAdapter(message_id="msg-1", graph=graph, **adapter_kwargs)
    executor = Executor(graph, MemoryBackend())
    return await ui_chunks(adapter, executor.execute(payload, dag.new_context(state)))


async def test_recommendation_stream_is_protocol_clean():
    chunks = await _stream(
        [json_turn(ACME_SLOTS)], [text_turn("Commerce Connect leads on CTR.")], {"text": ACME_BRIEF}
    )

    assert_valid_ui_stream(chunks, forbidden=STATE_KEYS)
    assert chunks[0] == {"type": "start", "messageId": "msg-1"}
    texts = [c["delta"] for c in chunks if c["type"] == "text-delta"]
    assert texts == ["Commerce Connect leads on CTR."]  # extract's raw JSON is hidden
    objects = [c for c in chunks if c["type"] == "data-object-complete"]
    assert objects[0]["data"]["object"]["vertical"] == "Retail"


async def test_text_block_ids_are_namespaced_per_node():
    chunks = await _stream([json_turn(ACME_SLOTS)], [text_turn("Done.")], {"text": ACME_BRIEF})

    ids = {c["id"] for c in chunks if c["type"] == "text-start"}
    assert ids == {"explain:b"}


async def test_node_progress_is_transient_and_carries_no_outputs():
    chunks = await _stream([json_turn(ACME_SLOTS)], [text_turn("Done.")], {"text": ACME_BRIEF})

    progress = [c for c in chunks if c["type"] == "data-mesh-node"]
    assert progress and all(c["transient"] is True for c in progress)
    assert all("output" not in c["data"] for c in progress)
    statuses = [(c["data"]["node_id"], c["data"]["status"]) for c in progress]
    assert ("allocate", "complete") in statuses
    assert statuses.count(("explain", "running")) == 1


async def test_form_stream_carries_the_form_tool_part():
    chunks = await _stream(
        [json_turn({**ACME_SLOTS, "vertical": None})], [], {"text": "A US app, $25,000."}
    )

    assert_valid_ui_stream(chunks, forbidden=STATE_KEYS)
    outputs = [c for c in chunks if c["type"] == "tool-output-available"]
    assert outputs[-1]["output"]["status"] == "FORM_REQUIRED"
    assert not [c for c in chunks if c["type"] == "text-delta"]


async def test_include_outputs_opt_in_attaches_node_outputs():
    chunks = await _stream(
        [json_turn(ACME_SLOTS)], [text_turn("Done.")], {"text": ACME_BRIEF}, include_outputs=True
    )

    validate = [
        c
        for c in chunks
        if c["type"] == "data-mesh-node"
        and c["data"]["node_id"] == "validate"
        and c["data"]["status"] == "complete"
    ]
    assert validate[0]["data"]["output"] == {"missing": []}


@pytest.mark.parametrize("failure", ["tool", "model"])
async def test_failures_end_the_stream_with_one_error(monkeypatch, failure):
    explain_script = [text_turn("Done.")]
    if failure == "tool":
        monkeypatch.setattr(dag, "scale", lambda state: 1 / 0)
    else:
        explain_script = [[RuntimeError("model provider returned 503")]]

    chunks = await _stream([json_turn(ACME_SLOTS)], explain_script, {"text": ACME_BRIEF})

    assert_valid_ui_stream(chunks, forbidden=STATE_KEYS)
    errors = [c for c in chunks if c["type"] == "error"]
    assert len(errors) == 1
    assert ("division by zero" if failure == "tool" else "503") in errors[0]["errorText"]


async def test_sse_frames_end_with_done():
    extract = scripted_agent("extract", [json_turn(ACME_SLOTS)], output_type=dag.Slots)
    explain = scripted_agent("explain", [text_turn("Done.")])
    graph = dag.build_graph(extract, explain)
    executor = Executor(graph, MemoryBackend())

    frames = [
        f
        async for f in UIMessageStreamAdapter(graph=graph).sse(
            executor.execute({"text": ACME_BRIEF}, dag.new_context())
        )
    ]

    assert all(f.startswith("data: ") and f.endswith("\n\n") for f in frames)
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("failure", ["tool", "model"])
async def test_error_text_hook_controls_what_the_client_sees(monkeypatch, failure):
    explain_script = [text_turn("Done.")]
    if failure == "tool":
        monkeypatch.setattr(dag, "scale", lambda state: 1 / 0)
    else:
        explain_script = [[RuntimeError("upstream 503 from api.example internal")]]

    chunks = await _stream(
        [json_turn(ACME_SLOTS)],
        explain_script,
        {"text": ACME_BRIEF},
        error_text=lambda exc: "Something went wrong. Please try again.",
    )

    errors = [c for c in chunks if c["type"] == "error"]
    assert errors == [{"type": "error", "errorText": "Something went wrong. Please try again."}]
