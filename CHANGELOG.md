# Changelog

## 0.2.x

Released by the publish workflow, which bumps the patch number: `pyproject.toml` says 0.2.0, so the first release is 0.2.1.

### Dependencies
- The dev dependency group pins vel 0.5.0 (`6d1315c`). It fixes the vel issues found while building these changes: run-scoped system messages accumulating in sessions, structured-output JSON streamed as text, error exits leaving the step open, tool results sent as a Python repr, and `from_function` tools not receiving `ctx`. Mesh's workarounds (`use_session=False`, the stream adapter's text hiding and step closing) remain and are harmless with the new vel.

### Fixed
- **Conditional edges never routed.** `StateGraph.add_conditional_edges` built predicates as `lambda x, k=key`, which `ConditionNode` called as `predicate(input, context)`, so no branch was ever taken. `ConditionNode` now counts only required positional parameters.
- **Joins after exclusive branches hung.** A node whose parents sit on branches a condition chose between waited for the branch that never ran. Skipped branches are now tracked per run and count as satisfied.
- A default target that is also a mapped branch was ignored when the mapped condition failed.
- Agent nodes after a condition received `str()` of the condition's wrapper dict as their message.
- vel `data-*` parts (e.g. `data-object-complete`) and mesh helper events were serialized as `data-custom`. They now keep their own type, with top-level `id` / `transient`.
- Agent nodes dropped vel's `tool-output-error`, `tool-input-error` and `abort`, leaving failed tool calls open on the client.
- Fan-in aggregators (`add_fan_in_edge(..., aggregator=...)`) were never applied.
- Concurrent runs on one `Executor` received each other's events. Each run now has a scoped emitter, and executor-level listeners still see every run.
- A disconnected consumer left the running node, such as an in-flight model call, running. A failing node left its event listener attached.
- Condition nodes emitted `data-node-start` twice.
- `SSEAdapter` wrote `event: EventType.X` instead of the wire type.

### Added
- `UIMessageStreamAdapter`: an opt-in Vercel AI SDK (`useChat`, ai v5.0.221+/v6) stream. It sends one `start` / `finish` per run, protocol chunks and keys only, node-namespaced block ids, transient `data-mesh-node` progress, interrupts without state, a single `error`, an `error_text` hook, `[DONE]`, and the `x-vercel-ai-ui-message-stream: v1` header.
- `ConditionNode(on_error="raise" | "unfulfilled")`. `StateGraph` conditional edges raise; parser-built conditions keep `"unfulfilled"`.
- Flow JSON conditions with `field` / `operation` / `value` (`equal`, `notEqual`, `contains`, `isEmpty`, `notEmpty`).
- `AgentNode(input_mode="messages", use_session=..., auto_parse_input=..., input_parser_model=...)`.
- Agent nodes report vel `response-metadata` (usage) on `NodeResult.metadata["response_metadata"]`.
- Example DAGs in `examples/dags/` (brief → recommendation, as a builder graph and as flow JSON; parallel card; follow-up tools) with offline tests that run real vel agents against a scripted provider, plus a `live` test tier.
- `scripts/ui_stream_conformance/` replays captured streams through a real `ai` client.
- CI: a test workflow (3.11 and 3.12), required before publishing.

### Behaviour changes to note when upgrading
- `ExecutionEvent.to_dict()` now emits vel and mesh `data-*` events under their own type instead of `data-custom`.
- A raising `add_conditional_edges` condition function now fails the run (`NodeExecutionError`) instead of silently taking no branch.
- `response-metadata` is no longer emitted as a stream event. It was already commented out, but two tests expected it.

### Known gaps
- `add_parallel_edges` branches run sequentially; the executor does not run them concurrently (`tests/dags/test_dag_b.py`, xfail).
