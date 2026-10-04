"""DAG A: media brief -> product recommendation.

Mirrors the per-turn pipeline of the Kargo media-recommendation case study:

    START -> route_input -+- form_submission -> merge_form ---+
                          +- free_text -------> extract (vel) -+-> validate
    validate -+- missing -> brief_form (FORM_REQUIRED tool part)
              +- ok ------> benchmarks -> scale -> viability -> allocate
                            -> explain (vel) -> guard

Deterministic steps are tool nodes; only ``extract`` (structured output) and
``explain`` (text) are LLM calls. ``validate`` joins two exclusive branches.

Run offline with scripted models:
    uv run python -m examples.dags.brief_to_recommendation
"""

import asyncio
import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel

from mesh import ExecutionContext, Executor, MemoryBackend, StateGraph
from mesh.core.events import EventType, ExecutionEvent
from mesh.core.graph import ExecutionGraph

# --------------------------------------------------------------------------
# Fixtures: a slice of the case-study data (2 verticals x 3 products, US/EMEA)
# --------------------------------------------------------------------------

CATALOG: dict[str, dict[str, Any]] = {
    "P001": {"name": "Display Plus", "cpm": 12.0},
    "P003": {"name": "Commerce Connect", "cpm": 15.0},
    "P006": {"name": "Attention Builder", "cpm": 14.0},
}

# (product, vertical) -> line items of (impressions, clicks, viewable_impressions)
HISTORY: dict[tuple, list[tuple]] = {
    ("P001", "Retail"): [(100_000, 600, 66_000), (50_000, 320, 33_500)],
    ("P003", "Retail"): [(100_000, 1_200, 69_000), (50_000, 630, 34_500)],
    ("P006", "Retail"): [(100_000, 1_000, 72_500), (50_000, 490, 36_500)],
    ("P001", "Finance"): [(100_000, 670, 68_000)],
    ("P003", "Finance"): [(100_000, 1_100, 70_000)],
    ("P006", "Finance"): [(100_000, 840, 76_000)],
}

# (product, vertical, geo) -> (available_imps, inventory_risk)
INVENTORY: dict[tuple, tuple] = {
    ("P001", "Retail", "US"): (1_466_000, 0.93),
    ("P003", "Retail", "US"): (1_720_000, 0.83),
    ("P006", "Retail", "US"): (1_593_000, 0.93),
    ("P001", "Finance", "US"): (1_276_000, 0.81),
    ("P003", "Finance", "US"): (1_650_000, 0.88),
    ("P006", "Finance", "US"): (1_386_000, 0.86),
}

REQUIRED_SLOTS = ("vertical", "kpi", "geo", "budget")
SLOT_OPTIONS = {
    "vertical": ["Retail", "Finance", "Travel", "QSR", "Entertainment"],
    "kpi": ["ctr", "in_view_rate"],
    "geo": ["US", "EMEA", "APAC"],
}
MAX_BUNDLE = 3


class Slots(BaseModel):
    """Structured output of the ``extract`` agent."""

    client_name: str | None = None
    vertical: Literal["Retail", "Finance", "Travel", "QSR", "Entertainment"] | None = None
    kpi: Literal["ctr", "in_view_rate"] | None = None
    geo: Literal["US", "EMEA", "APAC"] | None = None
    budget: float | None = None
    impression_goal: int | None = None


EXTRACT_PROMPT = (
    "Extract campaign details from the media brief. Use null for anything not "
    "stated explicitly; never guess the vertical or KPI."
)
EXPLAIN_PROMPT = (
    "Explain the recommendation to a media strategist in 2-3 sentences. Use only "
    "numbers that appear in the input."
)

# --------------------------------------------------------------------------
# Tool nodes (deterministic)
# --------------------------------------------------------------------------


def route_input(input: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Classify the turn: typed form answers vs a free-text brief."""
    request = input.get("input", input)
    metadata = request.get("metadata") or {}
    state.setdefault("slots", {})
    if metadata.get("type") == "tool_form_submission":
        state["form_parameters"] = metadata.get("parameters", {})
        return {"kind": "form_submission"}
    return {"kind": "free_text", "message": request.get("text", "")}


def merge_form(state: dict[str, Any]) -> dict[str, Any]:
    """Merge typed form answers into the stored slots (no LLM)."""
    state["slots"] = {**state["slots"], **state.pop("form_parameters", {})}
    return {"slots": state["slots"]}


def validate(input: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Merge extracted slots (if any) and list missing or invalid required slots."""
    extracted = _extracted_slots(input)
    merged = {**state["slots"], **{k: v for k, v in extracted.items() if v is not None}}
    state["slots"] = merged
    missing = [s for s in REQUIRED_SLOTS if not _valid(s, merged.get(s))]
    return {"missing": missing}


async def brief_form(context: ExecutionContext, state: dict[str, Any]) -> dict[str, Any]:
    """Emit a FORM_REQUIRED tool part asking for the missing slots."""
    missing = [s for s in REQUIRED_SLOTS if not _valid(s, state["slots"].get(s))]
    form = {
        "status": "FORM_REQUIRED",
        "form_config": {
            "title": "A few more details",
            "fields": [_form_field(s) for s in missing],
            "known": {k: v for k, v in state["slots"].items() if v is not None},
            "submit_label": "Get recommendation",
            "next_tool": "run_recommendation",
        },
    }
    call_id = f"form-{uuid.uuid4().hex[:8]}"
    await _emit_tool_part(context, call_id, "brief_details_form", form)
    state["pending_form"] = {"tool_call_id": call_id, **form}
    return form


def benchmarks(state: dict[str, Any]) -> dict[str, Any]:
    """Impression-weighted CTR and in-view rate per product for the vertical."""
    vertical = state["slots"]["vertical"]
    table = {}
    for (product_id, v), rows in HISTORY.items():
        if v != vertical:
            continue
        imps = sum(r[0] for r in rows)
        table[product_id] = {
            "ctr": sum(r[1] for r in rows) / imps,
            "in_view_rate": sum(r[2] for r in rows) / imps,
            "line_items": len(rows),
        }
    state["benchmarks"] = table
    return {"products": sorted(table)}


def scale(state: dict[str, Any]) -> dict[str, Any]:
    """Impressions the budget buys vs risk-adjusted inventory, per product."""
    slots = state["slots"]
    candidates = {}
    for product_id in state["benchmarks"]:
        available, risk = INVENTORY.get((product_id, slots["vertical"], slots["geo"]), (0, 0.0))
        cpm = CATALOG[product_id]["cpm"]
        candidates[product_id] = {
            "cpm": cpm,
            "affordable_imps": int(slots["budget"] / cpm * 1000),
            "effective_capacity": int(available * risk),
            "inventory_risk": risk,
        }
    state["candidates"] = candidates
    return {"candidates": len(candidates)}


def viability(state: dict[str, Any]) -> dict[str, Any]:
    """Reject products that cannot meet the impression goal; flag the rest."""
    goal = state["slots"].get("impression_goal") or 0
    rejected = {}
    for product_id, c in state["candidates"].items():
        if c["effective_capacity"] < goal or c["affordable_imps"] < goal:
            rejected[product_id] = "BELOW_IMPRESSION_GOAL"
        elif c["affordable_imps"] > c["effective_capacity"]:
            c["over_capacity"] = True
    state["rejected"] = rejected
    return {"rejected": rejected}


def allocate(state: dict[str, Any]) -> dict[str, Any]:
    """Greedy bundle: fill the best product to capacity, spill to the next."""
    kpi = state["slots"]["kpi"]
    remaining = float(state["slots"]["budget"])
    ranked = sorted(
        (p for p in state["candidates"] if p not in state["rejected"]),
        key=lambda p: state["benchmarks"][p][kpi],
        reverse=True,
    )
    lines = []
    for product_id in ranked[:MAX_BUNDLE]:
        if remaining <= 0:
            break
        c = state["candidates"][product_id]
        spend = min(remaining, c["effective_capacity"] * c["cpm"] / 1000)
        lines.append(
            {
                "product_id": product_id,
                "product_name": CATALOG[product_id]["name"],
                "spend": round(spend, 2),
                "impressions": int(spend / c["cpm"] * 1000),
            }
        )
        remaining -= spend
    allocation = {"lines": lines, "unspent": round(max(remaining, 0.0), 2)}
    state["allocation"] = allocation
    return {"message": _allocation_summary(allocation)}


def guard(input: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Pass the rationale only if every number in it appears in the allocation."""
    text = input.get("content", "")
    allowed = _numbers_in(_allocation_summary(state["allocation"]))
    passed = _numbers_in(text) <= allowed
    rationale = text if passed else _allocation_summary(state["allocation"])
    state["rationale"] = {"text": rationale, "guard_passed": passed}
    return state["rationale"]


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_graph(extract_agent: Any, explain_agent: Any) -> ExecutionGraph:
    """Wire DAG A. Agents are passed in so callers choose scripted or live models."""
    graph = StateGraph()
    graph.add_node("route_input", route_input, node_type="tool")
    graph.add_node("merge_form", merge_form, node_type="tool")
    graph.add_node("extract", extract_agent, node_type="agent")
    graph.add_node("validate", validate, node_type="tool")
    graph.add_node("brief_form", brief_form, node_type="tool")
    for name, fn in (
        ("benchmarks", benchmarks),
        ("scale", scale),
        ("viability", viability),
        ("allocate", allocate),
        ("guard", guard),
    ):
        graph.add_node(name, fn, node_type="tool")
    graph.add_node("explain", explain_agent, node_type="agent")

    graph.set_entry_point("route_input")
    graph.add_conditional_edges(
        "route_input",
        lambda out: out["kind"],
        {"form_submission": "merge_form", "free_text": "extract"},
    )
    graph.add_edge("merge_form", "validate")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda out: "missing" if out["missing"] else "ok",
        {"missing": "brief_form", "ok": "benchmarks"},
    )
    graph.add_edge("benchmarks", "scale")
    graph.add_edge("scale", "viability")
    graph.add_edge("viability", "allocate")
    graph.add_edge("allocate", "explain")
    graph.add_edge("explain", "guard")
    return graph.compile()


def new_context(state: dict[str, Any] | None = None) -> ExecutionContext:
    """Fresh execution context; pass ``state`` to continue a conversation."""
    return ExecutionContext(
        graph_id="brief_to_recommendation",
        session_id=f"session-{uuid.uuid4().hex[:8]}",
        chat_history=[],
        variables={},
        state=state if state is not None else {},
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extracted_slots(node_input: Any) -> dict[str, Any]:
    """Slots from the extract agent's structured output, if that branch ran."""
    if isinstance(node_input, dict) and set(node_input) & set(Slots.model_fields):
        return {k: node_input.get(k) for k in Slots.model_fields}
    if isinstance(node_input, dict) and isinstance(node_input.get("extract"), dict):
        return _extracted_slots(node_input["extract"])
    return {}


def _valid(slot: str, value: Any) -> bool:
    if slot == "budget":
        return isinstance(value, (int, float)) and value > 0
    return value in SLOT_OPTIONS[slot]


def _form_field(slot: str) -> dict[str, Any]:
    if slot == "budget":
        return {"name": "budget", "type": "number", "required": True}
    options = [{"value": o, "label": o} for o in SLOT_OPTIONS[slot]]
    return {"name": slot, "type": "select", "required": True, "options": options}


async def _emit_tool_part(
    context: ExecutionContext, call_id: str, tool_name: str, output: dict[str, Any]
) -> None:
    for raw in (
        {"type": "tool-input-start", "toolCallId": call_id, "toolName": tool_name},
        {"type": "tool-input-available", "toolCallId": call_id, "toolName": tool_name, "input": {}},
        {"type": "tool-output-available", "toolCallId": call_id, "output": output},
    ):
        await context.emit_event(ExecutionEvent(type=EventType(raw["type"]), raw_event=raw))


def _allocation_summary(allocation: dict[str, Any]) -> str:
    parts = [
        f"{line['product_name']} ({line['product_id']}): ${line['spend']:,.2f} "
        f"for {line['impressions']:,} impressions"
        for line in allocation["lines"]
    ]
    return "; ".join(parts) + f". Unspent: ${allocation['unspent']:,.2f}."


def _numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


async def _demo() -> None:
    from examples.dags.scripted import json_turn, scripted_agent, text_turn

    extract = scripted_agent(
        "extract",
        [json_turn({"client_name": "Acme Shoes", "vertical": "Retail", "kpi": "ctr",
                    "geo": "US", "budget": 30000})],
        output_type=Slots,
    )
    explain = scripted_agent("explain", [text_turn("Commerce Connect leads on CTR.")])
    executor = Executor(build_graph(extract, explain), MemoryBackend())
    context = new_context()
    brief = "Acme Shoes is a Retail advertiser in the US with a $30,000 budget; CTR matters most."
    async for event in executor.execute({"text": brief}, context):
        print(event.to_dict()["type"], event.node_id or "")
    print(context.state.get("allocation"))


if __name__ == "__main__":
    asyncio.run(_demo())
