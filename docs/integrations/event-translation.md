---
layout: default
title: Event Translation
parent: Integrations
nav_order: 3
---

# Event Translation Guide

Understanding Mesh's event translation system.

## Overview

Mesh uses **Vel's event translators** by default to provide consistent, provider-agnostic events across all agent frameworks.

## Default: Vel-Translated Events

```python
# Vel translation enabled by default
graph.add_node("agent", openai_agent, node_type="agent")

async for event in executor.execute(input, context):
    # Consistent event types regardless of provider
    if event.type == "token":
        print(event.content, end="", flush=True)
```

**Benefits:**
- Consistent event structure across OpenAI, Anthropic, Google
- Switch providers without changing event handling code
- Single source of truth for event translation (in Vel)

## Opt-in: Native Events

```python
# Use provider's native event format
graph.add_node("agent", openai_agent, node_type="agent",
               use_native_events=True)

async for event in executor.execute(input, context):
    # Provider-specific events
    print(event)
```

**Use When:**
- You need provider-specific event fields
- You're already familiar with the provider's event structure
- You want to bypass Vel translation

## How It Works

### With Vel Translation (Default)

```
User's Agent (OpenAI Agents SDK)
    ↓ Runner.run_streamed() [USES ACTUAL AGENT]
    ↓ Native OpenAI events
    ↓ Vel's SDK Event Translator (translate only)
    ↓ Mesh ExecutionEvent (provider-agnostic)
```

### Without Translation (use_native_events=True)

```
User's Agent (OpenAI Agents SDK)
    ↓ Runner.run_streamed() [USES ACTUAL AGENT]
    ↓ Native OpenAI events
    ↓ Mesh ExecutionEvent (provider-specific)
```

**Important:** Mesh always uses your actual agent SDK. Event translation just converts the event format.

## Event Mapping

### OpenAI Agents SDK → Vel

| OpenAI Event | Vel Event | Mesh Event |
|--------------|-----------|------------|
| `raw_response_event` | `text-delta` | `token` |
| `run_item_stream_event` (completed) | `text-end` | `message_complete` |
| `run_item_stream_event` (tool) | `tool-input-start` | `tool_call_start` |

## Examples

See the [event_translation_comparison.py](https://github.com/rscheiwe/mesh/blob/main/examples/event_translation_comparison.py) example for a side-by-side comparison.

## See Also

- [Vel SDK Translators](https://rscheiwe.github.io/vel/event-translators)
- [Events Concept](../concepts/events)
