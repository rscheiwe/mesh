/**
 * Check mesh UI-stream transcripts against a real AI SDK client.
 *
 *   uv run python scripts/ui_stream_conformance/capture.py /tmp/mesh-transcripts
 *   AI_DIST=/path/to/node_modules/ai/dist/index.mjs \
 *   SSE_REFEREE=/path/to/harness-agent/scripts/sse-assertions.ts \
 *     npx tsx scripts/ui_stream_conformance/check.mts /tmp/mesh-transcripts
 *
 *  1. DefaultChatTransport.processResponseStream: per-chunk uiMessageChunkSchema validation (what useChat runs)
 *  2. readUIMessageStream: the accumulator that builds the assistant UIMessage
 *  3. harness-agent's createSseAssertionTracker: the neutral protocol referee used by harness-parity
 */
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
// Paths to an installed `ai` package (v5.0.221+/v6) and, optionally, harness-agent's
// protocol referee. Nothing in those checkouts is modified.
const AI_DIST = process.env.AI_DIST ?? "../harness-agent-ui/node_modules/ai/dist/index.mjs";
const REFEREE = process.env.SSE_REFEREE; // e.g. ~/dev/harness-agent/scripts/sse-assertions.ts
const { DefaultChatTransport, readUIMessageStream } = await import(AI_DIST);
const createSseAssertionTracker = REFEREE ? (await import(REFEREE)).createSseAssertionTracker : null;

const HOUSE_RULES = [/^finish part missing finishReason$/];
const dir = process.argv[2];
let failed = 0;
for (const file of (await readdir(dir)).filter((f) => f.endsWith(".sse")).sort()) {
  const text = await readFile(join(dir, file), "utf8");
  const problems: string[] = [];
  let message: any;
  try {
    const body = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(text)); c.close(); } });
    const chunks = (new DefaultChatTransport() as any).processResponseStream(body);
    for await (const m of readUIMessageStream({ stream: chunks, onError: (e: unknown) => problems.push(`accumulator: ${e}`) })) message = m;
  } catch (e) {
    problems.push(`ai@6: ${(e as Error).message?.slice(0, 300)}`);
  }
  if (createSseAssertionTracker) {
    const tracker = createSseAssertionTracker();
    let done = false;
    for (const line of text.split(/\r?\n/)) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") { done = true; continue; }
      tracker.observe(JSON.parse(data));
    }
    const summary = tracker.summarize(true, done);
    const protocol = summary.ok ? [] : summary.lines.filter((l) => !HOUSE_RULES.some((r) => r.test(l)));
    problems.push(...protocol.map((l) => `referee: ${l}`));
  }
  // An error chunk is meant to reach the client's onError (useChat status "error")
  const hasErrorChunk = /"type": ?"error"/.test(text);
  const surfaced = problems.filter((p) => p.startsWith("accumulator:"));
  if (hasErrorChunk && surfaced.length === 1) problems.splice(problems.indexOf(surfaced[0]), 1);
  const note = hasErrorChunk ? " [error surfaced to onError]" : "";
  const parts = (message?.parts ?? []).map((p: any) => p.type + (p.state ? `(${p.state})` : "")).join(", ");
  console.log(`${problems.length ? "FAIL" : "ok  "} ${file.padEnd(28)} parts: ${parts}${note}`);
  for (const p of problems) console.log(`       ${p}`);
  if (problems.length) failed++;
}
process.exit(failed ? 1 : 0);
