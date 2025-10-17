# 04 - Event Translation

This folder contains examples demonstrating Mesh's event translation system and how it provides consistent events across different agent frameworks.

## Examples

### `event_translation_comparison.py`
**Difficulty:** ⭐⭐⭐ Advanced

Side-by-side comparison of:
1. **Native events** - Provider-specific event format
2. **Translated events** - Vel's standardized event format

**What you'll learn:**
- Difference between native and translated events
- Using `use_native_events=True` vs `False`
- Event format normalization
- When to use each approach

**Run:**
```bash
python 04_event_translation/event_translation_comparison.py
```

### `EVENT_TRANSLATION.md`

Comprehensive documentation on event translation including:
- Architecture overview
- Event translation flow
- Step-level tracking with `TranslatorOrchestrator`
- Gap filling for multi-step agents

## Key Concepts

### Event Translation Architecture

Mesh uses Vel's event translators to provide consistent events across providers:

```
OpenAI Agents SDK
    ↓ Runner.run_streamed() [actual SDK]
    ↓ Native OpenAI events
    ↓ [if use_native_events=False]
    ↓ Vel SDK Translator
    ↓ TranslatorOrchestrator (fills gaps)
    ↓ Mesh ExecutionEvent
```

### Native vs. Translated Events

**Native Events** (`use_native_events=True`):
- Provider-specific format
- Full control over event handling
- Requires provider-specific code
- Use when: You need provider-specific features

**Translated Events** (`use_native_events=False`, default):
- Consistent format across providers
- Standardized event types
- Provider-agnostic code
- Use when: You want portability

### Event Types

**Mesh Event Types:**
```python
# Execution lifecycle
EXECUTION_START, EXECUTION_COMPLETE, EXECUTION_ERROR

# Node lifecycle
NODE_START, NODE_COMPLETE, NODE_ERROR

# Step lifecycle (multi-step agents)
STEP_START, STEP_COMPLETE

# Content streaming
TOKEN, MESSAGE_START, MESSAGE_COMPLETE

# Tool execution
TOOL_CALL_START, TOOL_CALL_COMPLETE
```

### TranslatorOrchestrator

For multi-step agents (agents with tools), Mesh uses `TranslatorOrchestrator` to fill event gaps:

**What it does:**
1. Emits missing orchestration events (`start-step`, `finish-step`)
2. Tracks internal metadata (usage, finish_reason)
3. Detects step boundaries when tools are used
4. Accumulates total usage across steps

**Step Boundary Detection:**
```
Step 1: LLM call → "I'll check the weather"
    ↓ tool-input-available
    ↓ tool-output-available
Step 2: LLM call → "The weather is sunny"
    ↑ New step detected!
```

### Step-Level Tracking

With translated events, you get detailed step information:

```python
# Step events include metadata
{
    "type": "step_complete",
    "metadata": {
        "step_index": 0,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        "finish_reason": "tool_calls",
        "had_tool_calls": True
    }
}
```

**Benefits:**
- ✅ Per-step usage tracking (cost analysis)
- ✅ Step timing and duration
- ✅ Tool execution visibility
- ✅ Frontend progress indicators ("Step 1 of 3")

## Use Cases

### Provider-Agnostic Applications

```python
# Same code works with any agent type
graph.add_node("agent", agent, node_type="agent")

# Vel agent, OpenAI agent, etc. - all produce same events
```

### Cost Tracking

```python
total_tokens = 0
async for event in executor.execute(input, context):
    if event.type == "step_complete":
        usage = event.metadata.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)
```

### Multi-Step Monitoring

```python
steps = []
async for event in executor.execute(input, context):
    if event.type == "step_start":
        steps.append({"index": event.metadata["step_index"], "start": time.time()})
    elif event.type == "step_complete":
        steps[-1]["duration"] = time.time() - steps[-1]["start"]
```

## When Native Events are Better

Use `use_native_events=True` when:
- ✅ Using provider-specific features (e.g., OpenAI o1 reasoning)
- ✅ Need raw, unprocessed events
- ✅ Debugging provider integration
- ✅ Building provider-specific tooling

Use `use_native_events=False` (default) when:
- ✅ Building provider-agnostic apps
- ✅ Need consistent event format
- ✅ Want step-level tracking
- ✅ Need usage/cost tracking

## Related Files

- **`mesh/utils/translator_orchestrator.py`** - Gap filling logic
- **`mesh/nodes/agent.py`** - Agent node with translation
- **`mesh/core/events.py`** - Event type definitions

## Further Reading

- Vel Event Translators: `/Users/richard.s/vel/docs/Event Translators/`
- CLAUDE.md: Section 4 - TranslatorOrchestrator
