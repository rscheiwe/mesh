# Data Handler Node

A specialized ToolNode for executing SQL queries against various database sources with support for fixed and AI-interpolated parameters.

## Overview

The DataHandlerNode is a **specialized subclass of ToolNode** designed for database operations. It:

1. Connects to specified database source (MySQL, Postgres, Vertica, etc.)
2. Executes SQL queries with named parameters
3. Supports both **fixed parameters** (user-defined) and **interpolated parameters** (AI-generated)
4. Can be stored in database as records (like other tool nodes)

## Architecture

```
DataHandlerNode (extends ToolNode)
    ↓
Internally creates query executor function
    ↓
ToolNode handles execution, retries, events
    ↓
Query executed against injected DB session
```

## Usage Patterns

### Pattern 1: Fixed Parameters (User-Defined)

User provides all query parameters at configuration time:

```python
from mesh.nodes import DataHandlerNode

node = DataHandlerNode(
    id="data_handler_0",
    db_source="postgres",
    query="SELECT * FROM users WHERE status = :status AND age > :min_age",
    params={
        "status": "active",
        "min_age": 18
    }
)

# Inject DB session getter
node.set_db_session_getter(lambda source: get_db_session(source))

graph.add_node("data_handler_0", node, node_type="data_handler")
```

### Pattern 2: Interpolated Parameters (AI-Generated)

AI agent provides parameters dynamically:

```python
# LLM generates query parameters
graph.add_node("parameter_generator", None, node_type="llm",
               model="gpt-4",
               system_prompt="""
               Extract user info:
               - status: active/inactive
               - min_age: number

               Return JSON: {"status": "...", "min_age": ...}
               """)

# Data handler uses AI-generated params
node = DataHandlerNode(
    id="data_handler_0",
    db_source="postgres",
    query="SELECT * FROM users WHERE status = :status AND age > :min_age",
    # No fixed params - will use output from parameter_generator
)

graph.add_edge("START", "parameter_generator")
graph.add_edge("parameter_generator", "data_handler_0")
```

### Pattern 3: Hybrid (Fixed + Interpolated)

Mix of user-defined and AI-generated parameters:

```python
node = DataHandlerNode(
    id="data_handler_0",
    db_source="postgres",
    query="SELECT * FROM logs WHERE app = :app AND user_id = :user_id",
    params={
        "app": "production"  # Fixed
        # user_id will be interpolated from AI output
    }
)
```

## Database Storage Schema

Store DataHandlerNodes in your database like other tool nodes:

```sql
CREATE TABLE mosaic_agent_tool_nodes (
    id SERIAL PRIMARY KEY,
    node_uuid UUID UNIQUE,
    user_id INTEGER,

    -- Data Handler specific
    label VARCHAR(255),
    name VARCHAR(255),
    description TEXT,
    type VARCHAR(50) DEFAULT 'DataHandler',

    -- Configuration (JSON)
    inputs JSONB,  -- DB source, query, params config
    outputs JSONB,

    -- Metadata
    category VARCHAR(100) DEFAULT 'Data Operations',
    icon VARCHAR(100),
    is_public BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    active BOOLEAN DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Example Insert:

```sql
INSERT INTO mosaic_agent_tool_nodes (
    node_uuid, user_id, label, name, type, description, category, inputs
) VALUES (
    'e3d06619-dc5f-41ca-9afa-e7c197cbf7f9',
    129617,
    'User Query Handler',
    'get_users',
    'DataHandler',
    'Query users table with filters',
    'Data Operations',
    '[
        {
            "name": "db_source",
            "type": "options",
            "label": "Database Source",
            "options": [
                {"name": "postgres", "label": "PostgreSQL"},
                {"name": "mysql", "label": "MySQL"},
                {"name": "vertica", "label": "Vertica"}
            ],
            "default": "postgres",
            "optional": false
        },
        {
            "name": "query",
            "type": "code",
            "label": "SQL Query",
            "rows": 8,
            "default": "SELECT * FROM users WHERE status = :status",
            "optional": false,
            "placeholder": "SELECT * FROM table WHERE column = :param"
        },
        {
            "name": "params",
            "type": "code",
            "label": "Query Parameters (JSON)",
            "rows": 4,
            "default": "{\"status\": \"active\"}",
            "optional": true,
            "placeholder": "{\"param_name\": \"value\"}"
        }
    ]'::jsonb
);
```

## React Flow Integration

Add to UI registry:

```typescript
// In mesh-app/mesh-ui/src/registry/index.ts

{
  type: "dataHandlerAgentflow",
  name: "data_handler",
  label: "Data Handler",
  description: "Execute SQL queries against database sources",
  icon: "Database",
  category: "Data Operations",
  color: "#10b981", // green
  inputs: [
    {
      name: "id",
      type: "string",
      label: "Node ID (auto-generated)",
      placeholder: "data_handler_0",
      showInNode: true,
    },
    {
      name: "dbSource",
      type: "options",
      label: "Database Source",
      default: "postgres",
      showInNode: true,
      options: [
        { name: "postgres", label: "PostgreSQL" },
        { name: "mysql", label: "MySQL" },
        { name: "vertica", label: "Vertica" },
      ],
    },
    {
      name: "query",
      type: "code",
      label: "SQL Query",
      rows: 8,
      placeholder: "SELECT * FROM table WHERE column = :param",
      description: "Use :param_name for named parameters",
    },
    {
      name: "params",
      type: "code",
      label: "Fixed Parameters (JSON)",
      rows: 4,
      optional: true,
      placeholder: '{"param_name": "value"}',
      description: "Leave empty for AI-interpolated params",
    },
  ],
  outputs: ["rows", "count"],
}
```

## Parser Integration

Add to React Flow parser:

```python
# In mesh/parsers/react_flow.py

from mesh.nodes.data_handler import DataHandlerNode

NODE_TYPE_MAP = {
    # ... existing mappings
    "dataHandlerAgentflow": "data_handler",
}

def _create_data_handler_node(self, node_id: str, config: Dict[str, Any]) -> DataHandlerNode:
    """Create DataHandlerNode from config."""
    return DataHandlerNode(
        id=node_id,
        db_source=config.get("dbSource") or config.get("db_source", "postgres"),
        query=config.get("query", ""),
        params=config.get("params", {}),
        event_mode=config.get("eventMode", "full"),
        config=config,
    )

# In _create_node method:
elif node_type == "data_handler":
    return self._create_data_handler_node(node_id, config)
```

## Dependency Injection

Inject DB session getter after parsing (like RAGNode):

```python
# In backend/routers/execution.py

from mesh.nodes import DataHandlerNode

# After parsing
graph = parser.parse(request.flow)

# Inject DB session getter into DataHandler nodes
def get_db_session(source: str):
    """Get database session for given source."""
    from server.resources.database import DBSession

    source_map = {
        "postgres": DBSession.postgres,
        "mysql": DBSession.master,
        "vertica": DBSession.vrt,
    }

    if source not in source_map:
        raise ValueError(f"Unknown database source: {source}")

    return source_map[source].value()

for node in graph.nodes.values():
    if isinstance(node, DataHandlerNode):
        node.set_db_session_getter(get_db_session)
```

## Output Structure

DataHandlerNode returns:

```python
{
    "rows": [
        {"id": 1, "name": "Alice", "status": "active"},
        {"id": 2, "name": "Bob", "status": "active"},
    ],
    "count": 2,
    "query": "SELECT * FROM users WHERE status = :status",
    "params": {"status": "active"}
}
```

Access in downstream nodes:
- `{{data_handler_0.output.rows}}` - Array of result rows
- `{{data_handler_0.output.count}}` - Number of rows returned
- `{{data_handler_0.output.query}}` - The executed query

## Complete Example Flow

```python
from mesh import StateGraph
from mesh.nodes import DataHandlerNode

# 1. Create graph
graph = StateGraph()

# 2. LLM extracts query parameters from user question
graph.add_node("extractor", None, node_type="llm",
               model="gpt-4",
               system_prompt="""
               Extract database query params from: {{$question}}
               Return JSON with status and min_age.
               """)

# 3. Data handler executes query with AI-extracted params
data_handler = DataHandlerNode(
    id="query_users",
    db_source="postgres",
    query="SELECT * FROM users WHERE status = :status AND age > :min_age",
    # No fixed params - uses extractor output
)

# 4. LLM formats results for user
graph.add_node("formatter", None, node_type="llm",
               model="gpt-4",
               system_prompt="""
               Format these query results for the user:
               {{query_users.output.rows}}

               Found {{query_users.output.count}} users.
               """)

# 5. Connect
graph.add_edge("START", "extractor")
graph.add_edge("extractor", "query_users")
graph.add_edge("query_users", "formatter")

# 6. Inject DB session getter
data_handler.set_db_session_getter(get_db_session)

# 7. Execute
result = await graph.run(input="Show me active users over 25")
```

## Advantages Over Generic ToolNode

1. **Type Safety**: Explicit DB source and query validation
2. **UI Integration**: Specialized UI fields (DB dropdown, SQL editor)
3. **Parameter Resolution**: Built-in logic for fixed vs interpolated params
4. **Session Management**: Automatic connection handling and cleanup
5. **Database Storage**: Clean schema for persisting data handlers
6. **Error Handling**: Query-specific error messages with context

## When to Use

**✅ Use DataHandlerNode when:**
- Executing SQL queries as part of agent workflow
- Need to mix fixed and AI-generated query parameters
- Want specialized UI for database operations
- Storing node configurations in database
- Building data-driven agents

**❌ Use regular ToolNode when:**
- Non-database operations
- Custom Python logic beyond SQL
- Don't need parameter interpolation
- One-off scripts without reusability

## Security Considerations

1. **SQL Injection**: Always use named parameters (`:param_name`), never string interpolation
2. **Connection Pooling**: Ensure DB session getter uses connection pools
3. **Query Validation**: Consider whitelist of allowed tables/operations
4. **Parameter Sanitization**: Validate AI-generated params before execution
5. **Access Control**: Check user permissions before query execution

## Migration from ToolNode

Existing ToolNodes doing database operations can be migrated:

**Before (ToolNode):**
```python
def query_users(input: dict) -> dict:
    session = get_postgres_session()
    result = session.execute("SELECT * WHERE id = :id", {"id": input["user_id"]})
    return {"rows": result.fetchall()}

graph.add_node("tool", query_users, node_type="tool")
```

**After (DataHandlerNode):**
```python
node = DataHandlerNode(
    id="query_users",
    db_source="postgres",
    query="SELECT * WHERE id = :user_id",
)
node.set_db_session_getter(get_db_session)
graph.add_node("query_users", node, node_type="data_handler")
```

Benefits:
- Less code
- Configurable via UI
- Stored in database
- Better error handling
- Consistent parameter resolution
