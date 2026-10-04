"""Live tier: the example DAGs against real models through vel.

Deselected by default. Run with keys in the environment (or mesh's .env):

    OPENAI_API_KEY=... uv run pytest -m live tests/dags/test_live.py -s
    MESH_LIVE_PROVIDER=anthropic ANTHROPIC_API_KEY=... uv run pytest -m live ... -s

MESH_LIVE_MODEL overrides the model. Each test prints latency and token usage.
"""

import os
import time

import pytest
from vel import Agent

from examples.dags import brief_to_recommendation as dag
from examples.dags import followup_tools as dag_c
from mesh import Executor, MemoryBackend, UIMessageStreamAdapter
from mesh.core.events import EventType
from mesh.utils import load_env
from tests.dags.conftest import assert_valid_ui_stream, collect, completed_nodes

load_env()

PROVIDER = os.getenv("MESH_LIVE_PROVIDER", "openai")
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}
MODEL = os.getenv("MESH_LIVE_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))
KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv(KEY_VARS.get(PROVIDER, "")),
        reason=f"{KEY_VARS.get(PROVIDER, PROVIDER)} not set",
    ),
]

# The case-study briefs and the slots a correct extraction yields (None = must ask).
BRIEFS = [
    (
        "Acme Shoes is a Retail advertiser running in the US with a $30,000 budget. They "
        "care most about click-through rate and do not have a hard impression goal.",
        {"vertical": "Retail", "kpi": "ctr", "geo": "US", "budget": 30000, "impression_goal": None},
    ),
    (
        "Burger Bazaar is a QSR advertiser running in the US with a $15,000 budget. They "
        "care most about in-view rate and want at least 800,000 impressions.",
        {
            "vertical": "QSR",
            "kpi": "in_view_rate",
            "geo": "US",
            "budget": 15000,
            "impression_goal": 800000,
        },
    ),
    (
        "CineVerse is an Entertainment advertiser running in EMEA with a $40,000 budget. "
        "They care most about click-through rate and want a straightforward recommendation.",
        {"vertical": "Entertainment", "kpi": "ctr", "geo": "EMEA", "budget": 40000},
    ),
    (
        "Stellar Bank is a Finance advertiser running in the US with a $20,000 budget. They "
        "care most about click-through rate and want at least 800,000 impressions.",
        {
            "vertical": "Finance",
            "kpi": "ctr",
            "geo": "US",
            "budget": 20000,
            "impression_goal": 800000,
        },
    ),
    (
        "Wanderlust Air is a Travel advertiser running in APAC with an $18,000 budget. They "
        "care most about in-view rate and want at least 850,000 impressions.",
        {
            "vertical": "Travel",
            "kpi": "in_view_rate",
            "geo": "APAC",
            "budget": 18000,
            "impression_goal": 850000,
        },
    ),
    (
        "A new mobile app wants efficient awareness in the US with a $25,000 budget and "
        "strong in-view performance, but they have not provided their vertical.",
        {"vertical": None, "geo": "US", "budget": 25000},
    ),
    (
        "A Finance brand wants a recommendation for EMEA with a $25,000 budget, but they "
        "have not said whether they care more about click-through rate or in-view rate.",
        {"vertical": "Finance", "kpi": None, "geo": "EMEA", "budget": 25000},
    ),
]


def _model():
    return {"provider": PROVIDER, "model": MODEL}


def _extract_agent():
    return Agent(
        id="extract",
        model=_model(),
        system_prompt=dag.EXTRACT_PROMPT,
        output_type=dag.Slots,
        generation_config={"temperature": 0},
    )


def _explain_agent():
    return Agent(
        id="explain",
        model=_model(),
        system_prompt=dag.EXPLAIN_PROMPT,
        generation_config={"temperature": 0},
    )


def _report(label, started, events):
    usage = [
        meta.get("usage")
        for e in events
        if e.type == EventType.NODE_COMPLETE
        for meta in (e.metadata or {}).get("response_metadata", [])
    ]
    print(f"\n[{PROVIDER}:{MODEL}] {label}: {time.perf_counter() - started:.2f}s, usage={usage}")


@pytest.mark.parametrize("brief, expected", BRIEFS, ids=[f"brief{i + 1}" for i in range(7)])
async def test_extraction_and_routing_on_case_study_briefs(brief, expected):
    graph = dag.build_graph(_extract_agent(), _explain_agent())
    context = dag.new_context()
    started = time.perf_counter()

    events = await collect(Executor(graph, MemoryBackend()).execute({"text": brief}, context))

    _report("extract", started, events)
    slots = context.state["slots"]
    for key, value in expected.items():
        assert slots.get(key) == value, f"{key}: got {slots.get(key)!r}, want {value!r}"
    asks = None in expected.values()
    assert ("brief_form" in completed_nodes(events)) is asks


async def test_recommendation_stream_end_to_end():
    graph = dag.build_graph(_extract_agent(), _explain_agent())
    context = dag.new_context()
    started = time.perf_counter()
    events = await collect(
        Executor(graph, MemoryBackend()).execute({"text": BRIEFS[0][0]}, context)
    )
    _report("recommend", started, events)

    async def replay():
        for event in events:
            yield event

    chunks = [c async for c in UIMessageStreamAdapter(graph=graph).chunks(replay())]
    assert_valid_ui_stream(chunks)
    assert [line["product_id"] for line in context.state["allocation"]["lines"]] == [
        "P003",
        "P006",
    ]
    rationale = context.state["rationale"]
    print(f"rationale (guard_passed={rationale['guard_passed']}): {rationale['text']}")
    assert rationale["text"]


async def test_followup_model_picks_what_if_for_a_budget_question():
    state = dag_c.recommended_state()
    agent = Agent(
        id="followup",
        model=_model(),
        system_prompt=dag_c.FOLLOWUP_PROMPT,
        tools=dag_c.FOLLOWUP_TOOLS,
        tool_context={"state": state},
        generation_config={"temperature": 0},
    )
    graph = dag_c.build_graph(agent)
    started = time.perf_counter()
    events = await collect(
        Executor(graph, MemoryBackend()).execute(
            "What would the plan look like with a $40,000 budget?", dag.new_context(state)
        )
    )
    _report("followup", started, events)

    calls = [e.to_dict() for e in events if e.type == EventType.TOOL_INPUT_AVAILABLE]
    assert calls and calls[0]["toolName"] == "what_if"
    assert calls[0]["input"].get("budget") == 40000
