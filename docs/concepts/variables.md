# Variable Resolution

Mesh provides a powerful variable resolution system that allows you to reference data across nodes using template syntax. Variables use the `{{variable}}` syntax and can reference user input, node outputs, global variables, and more.

## Overview

The `VariableResolver` processes template strings at runtime, replacing `{{variable}}` references with actual values from the execution context. This enables dynamic prompt construction and data flow between nodes.

## Supported Variable Types

### User Input Variables

Reference the initial input to your graph:

```python
# Basic input reference
system_prompt="Analyze this topic: {{$input}}"

# Nested field access (requires structured input)
system_prompt="Write about {{$input.topic1}} and {{$input.topic2}}"

# Alternative syntax (same as $input)
system_prompt="User asked: {{$question}}"
```

**Important**: Use `{{$input.fieldname}}` syntax, not `{{$fieldname}}`. The `$input.` prefix is required for field extraction.

### Node Output Variables

Reference the output of other nodes in your graph:

```python
# Reference entire node output
system_prompt="Previous result: {{writer_node}}"

# Access nested fields in node output
system_prompt="The content was: {{writer_node.content}}"

# Deep nesting supported
system_prompt="User name: {{parser_node.data.user.name}}"
```

Node IDs must match the `id` parameter used when creating the node.

### Global Variables

Access global variables from the execution context:

```python
# Reference global variable
system_prompt="Hello {{$vars.user_name}}!"

# Nested global variables
system_prompt="Theme: {{$vars.settings.theme}}"
```

Global variables are passed via `ExecutionContext.variables`.

### Chat History

Access the full conversation history:

```python
system_prompt="Given this history: {{$chat_history}}, respond appropriately."
```

Formats chat history as:
```
user: Hello
assistant: Hi there!
user: How are you?
```

### Iteration Variables

When using `LoopNode`, access the current iteration context:

```python
# Current iteration value
system_prompt="Process item: {{$iteration}}"

# Nested field in iteration value
system_prompt="Name: {{$iteration.name}}, Age: {{$iteration.age}}"

# Iteration metadata
system_prompt="Processing item {{$iteration_index}} of {{$iteration_total}}"
```

## Automatic Input Parsing

Mesh can automatically parse natural language input into structured data when your system prompt references multiple input fields.

### How It Works

When your system prompt contains multiple `{{$input.X}}` variables, Mesh automatically:

1. **Detects** the field names you're referencing
2. **Parses** the natural language input using an LLM
3. **Structures** the data for variable resolution
4. **Resolves** your prompt with the extracted values

### Example

**System Prompt:**
```python
system_prompt="Write 3 posts about {{$input.topic1}}, and 2 about {{$input.topic2}}"
```

**Natural Language Input:**
```
"1. dogs, 2. cats"
```

**What Happens:**
1. Mesh detects `topic1` and `topic2` fields
2. Calls an LLM to parse: `"1. dogs, 2. cats"` → `{"topic1": "dogs", "topic2": "cats"}`
3. Resolves prompt: `"Write 3 posts about dogs, and 2 about cats"`

### Performance

- **Zero overhead** when not needed (detection is pattern-based)
- **Only triggers** when 2+ `{{$input.X}}` variables are detected
- **Uses gpt-4o-mini** for fast, cost-efficient parsing
- **Skips parsing** if input is already structured (dict/JSON)

### Syntax Requirements

For automatic parsing to work:

✅ **Correct:**
```python
{{$input.topic1}}
{{$input.topic2}}
{{$question.field_name}}
```

❌ **Incorrect:**
```python
{{$topic1}}          # Missing $input prefix
{{topic1}}           # Missing $ and prefix
{{$input_topic1}}    # Use dot notation, not underscore
```

### Structured Input (No Parsing)

If you provide structured input directly, parsing is skipped:

```python
# Input as JSON/dict
input_data = {
    "topic1": "dogs",
    "topic2": "cats"
}

# Variables resolve directly
system_prompt="Write about {{$input.topic1}} and {{$input.topic2}}"
# Result: "Write about dogs and cats"
```

## Variable Resolution API

### In Python Code

```python
from mesh.utils.variables import VariableResolver

# Create resolver with execution context
resolver = VariableResolver(context)

# Resolve a template string
template = "User asked: {{$question}}, agent said: {{agent_node.content}}"
resolved = await resolver.resolve(template)

# Resolve all strings in a dictionary
data = {
    "prompt": "Analyze {{$input}}",
    "title": "Response to {{$question}}"
}
resolved_data = resolver.resolve_dict(data)
```

### In Agent Nodes

Agent nodes automatically resolve `system_prompt` variables:

```python
from mesh.nodes.agent import AgentNode

agent_node = AgentNode(
    id="writer",
    agent=vel_agent,
    system_prompt="Write about {{$input}} in the style of {{$vars.style}}"
)
```

### In ReactFlow JSON

Variables work in Flowise/ReactFlow configurations:

```json
{
  "nodes": [
    {
      "id": "agent_1",
      "type": "agentAgentflow",
      "data": {
        "inputs": {
          "agent": "writer",
          "systemPrompt": "Analyze {{$input}} and compare to {{previous_node}}"
        }
      }
    }
  ]
}
```

## Best Practices

### 1. Use Descriptive Field Names

```python
# Good - clear intent
{{$input.user_query}}
{{$input.target_language}}

# Bad - unclear
{{$input.x}}
{{$input.data1}}
```

### 2. Provide Fallbacks

When a variable is missing, it's replaced with an empty string. For critical fields, validate inputs:

```python
# Check for required fields in your node logic
if not input.get("topic"):
    raise ValueError("Missing required field: topic")
```

### 3. Structure Your Data

For complex data, use structured input instead of relying on parsing:

```python
# Structured input (preferred for APIs)
{
    "query": "search term",
    "filters": {"category": "tech", "date": "2025"},
    "limit": 10
}

# Natural language (good for chat UIs)
"Find tech articles from 2025, limit 10 results"
```

### 4. Test Parsing

When using automatic parsing, test with various input formats:

```python
# Different ways users might express the same thing:
"1. dogs, 2. cats"
"First: dogs, Second: cats"
"dogs and cats"
"Write about dogs, then cats"
```

## Troubleshooting

### Variable Not Resolved

**Symptom:** `{{MISSING: variable_name}}` appears in output

**Causes:**
- Variable name typo
- Node hasn't executed yet (wrong execution order)
- Field doesn't exist in the data structure

**Solution:**
```python
# Check node execution order
graph.validate()  # Ensures nodes are reachable

# Verify field exists
print(context.get_node_output("node_id"))

# Check variable syntax
{{$input.field}}  # ✅ Correct
{{field}}         # ❌ Missing $ and prefix
```

### Parsing Not Triggered

**Symptom:** Natural language input not being parsed

**Causes:**
- Only one `{{$input.X}}` variable (parsing requires 2+)
- Input is already structured (dict)
- Wrong syntax (e.g., `{{$topic}}` instead of `{{$input.topic}}`)

**Solution:**
```python
# Requires at least 2 fields for auto-parsing
{{$input.field1}} and {{$input.field2}}  # ✅ Triggers parsing
{{$input.field1}} only                    # ❌ No parsing
```

### Parsing Extracts Wrong Data

**Symptom:** Parsed fields don't match user input

**Causes:**
- Ambiguous natural language input
- Field names don't match the content

**Solution:**
```python
# Use clear field names that match expected content
system_prompt="Topic: {{$input.topic}}, Tone: {{$input.tone}}"

# Good input: "topic is AI, tone is professional"
# Bad input: "write about AI in a good way"  # "tone" field unclear
```

## Examples

### Multi-Step Pipeline

```python
from mesh.nodes.agent import AgentNode
from mesh.core.graph import ExecutionGraph

# Step 1: Extract entities
extractor = AgentNode(
    id="extractor",
    agent=extractor_agent,
    system_prompt="Extract entities from: {{$input}}"
)

# Step 2: Process each entity (references previous node)
processor = AgentNode(
    id="processor",
    agent=processor_agent,
    system_prompt="Process entities: {{extractor.entities}}"
)

# Step 3: Summarize (references multiple nodes)
summarizer = AgentNode(
    id="summarizer",
    agent=summarizer_agent,
    system_prompt="Original: {{$input}}, Extracted: {{extractor}}, Processed: {{processor}}"
)

graph = ExecutionGraph.from_nodes_and_edges(
    nodes={"extractor": extractor, "processor": processor, "summarizer": summarizer},
    edges=[
        Edge("START", "extractor"),
        Edge("extractor", "processor"),
        Edge("processor", "summarizer"),
        Edge("summarizer", "END")
    ]
)
```

### Personalized Agent

```python
# Set global variables
context.variables = {
    "user_name": "Alice",
    "expertise_level": "beginner",
    "preferred_style": "casual"
}

# Agent uses variables in prompt
agent_node = AgentNode(
    id="assistant",
    agent=agent,
    system_prompt="""
    You are helping {{$vars.user_name}}, who is a {{$vars.expertise_level}}.
    Use a {{$vars.preferred_style}} tone.
    User question: {{$input}}
    """
)
```

### Natural Language Multi-Input

```python
# System prompt with multiple fields
agent_node = AgentNode(
    id="content_writer",
    agent=writer_agent,
    system_prompt="""
    Write {{$input.num_posts}} social media posts about {{$input.topic}}.
    Target audience: {{$input.audience}}.
    Tone: {{$input.tone}}.
    """
)

# User provides natural language input
input_text = "Write 5 posts about AI for developers in a technical tone"

# Mesh automatically parses to:
# {
#   "num_posts": "5",
#   "topic": "AI",
#   "audience": "developers",
#   "tone": "technical"
# }
```

## Related Documentation

- [Nodes](nodes.md) - Understanding node types and configuration
- [Execution](execution.md) - How graphs execute and manage state
- [Events](events.md) - Event system for monitoring execution
- [Vel Integration](../integrations/vel.md) - Using Vel agents with Mesh
