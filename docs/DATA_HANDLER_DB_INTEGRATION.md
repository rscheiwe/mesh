# DataHandlerNode - Database Integration

How to integrate DataHandlerNode with your existing `mosaic_agent_tool_nodes` table.

## Existing Table Structure

You already have:

```sql
mosaic_agent_tool_nodes (
    id, node_uuid, user_id,
    code, imports, label, name, version,
    type, icon, category, description,
    base_classes, inputs, outputs,
    credential, file_path,
    active, is_used, is_public, is_verified,
    language, args,
    created_at, updated_at
)
```

## Storing DataHandlerNodes

Simply insert with `type='DataHandler'`:

```sql
INSERT INTO mosaic_agent_tool_nodes (
    node_uuid,
    user_id,
    code,           -- Empty for DataHandler (no custom Python needed)
    imports,        -- '[]'
    label,          -- 'Query Active Users'
    name,           -- 'query_active_users'
    version,        -- 1
    type,           -- 'DataHandler' (KEY FIELD)
    icon,           -- 'Database'
    category,       -- 'Data Operations'
    description,    -- 'Queries users table with filters'
    base_classes,   -- '["DataHandler"]'
    inputs,         -- JSON array with db_source, query, params
    outputs,        -- '["rows", "count"]'
    args,           -- JSON array (same as inputs for backwards compat)
    active,         -- true
    is_public,      -- false
    language,       -- 'sql'
    is_verified     -- false
) VALUES (
    'e3d06619-dc5f-41ca-9afa-e7c197cbf7f9',
    129617,
    '',  -- No Python code needed
    '[]',
    'Query Active Users',
    'query_active_users',
    1,
    'DataHandler',  -- Type distinguishes from Tool
    'Database',
    'Data Operations',
    'Retrieve active users from database',
    '["DataHandler"]',
    '[
        {
            "name": "db_source",
            "type": "options",
            "label": "Database Source",
            "options": [
                {"name": "postgres", "label": "PostgreSQL"},
                {"name": "mysql", "label": "MySQL"}
            ],
            "default": "postgres",
            "optional": false
        },
        {
            "name": "query",
            "type": "code",
            "label": "SQL Query",
            "default": "SELECT id, name, email FROM users WHERE status = :status",
            "optional": false
        },
        {
            "name": "params",
            "type": "code",
            "label": "Query Parameters (JSON)",
            "default": "{\"status\": \"active\"}",
            "optional": true
        }
    ]'::jsonb,
    '["rows", "count"]'::jsonb,
    '[
        {
            "name": "db_source",
            "type": "options",
            "default": "postgres"
        },
        {
            "name": "query",
            "type": "code",
            "default": "SELECT id, name FROM users WHERE status = :status"
        },
        {
            "name": "params",
            "type": "code",
            "default": "{\"status\": \"active\"}"
        }
    ]'::jsonb,
    true,
    false,
    'sql',  -- Language field = 'sql' for DataHandlers
    false
);
```

## Loading from Database

Create a loader that reads from your table and instantiates the correct node type:

```python
# backend/loaders/node_loader.py

from typing import Dict, Any, List
import json
from mesh.nodes import DataHandlerNode, ToolNode

def load_node_from_db(record: Dict[str, Any]):
    """Load node from mosaic_agent_tool_nodes record.

    Args:
        record: Database record dict

    Returns:
        DataHandlerNode or ToolNode instance
    """
    node_type = record.get('type', 'Tool')
    node_uuid = record['node_uuid']

    if node_type == 'DataHandler':
        return _load_data_handler(record)
    elif node_type == 'Tool':
        return _load_tool(record)
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def _load_data_handler(record: Dict[str, Any]) -> DataHandlerNode:
    """Load DataHandlerNode from DB record."""
    inputs = record.get('inputs', [])
    if isinstance(inputs, str):
        inputs = json.loads(inputs)

    # Extract config from inputs
    db_source = None
    query = None
    params = {}

    for inp in inputs:
        name = inp.get('name')
        default = inp.get('default')

        if name == 'db_source':
            db_source = default
        elif name == 'query':
            query = default
        elif name == 'params':
            if isinstance(default, str):
                params = json.loads(default) if default else {}
            else:
                params = default or {}

    if not db_source or not query:
        raise ValueError(
            f"DataHandler '{record['name']}' missing db_source or query"
        )

    return DataHandlerNode(
        id=record['node_uuid'],
        db_source=db_source,
        query=query,
        params=params,
    )


def _load_tool(record: Dict[str, Any]) -> ToolNode:
    """Load regular ToolNode from DB record."""
    # Execute user's Python code
    code = record.get('code', '')
    imports = record.get('imports', '[]')

    if isinstance(imports, str):
        imports = json.loads(imports)

    # Import dependencies
    for imp in imports:
        exec(imp)

    # Execute code to get function
    namespace = {}
    exec(code, namespace)

    # Find the function
    func_name = record['name']
    if func_name not in namespace:
        raise ValueError(f"Function '{func_name}' not found in code")

    tool_fn = namespace[func_name]

    return ToolNode(
        id=record['node_uuid'],
        tool_fn=tool_fn,
    )
```

## Backend Registry Integration

Update your backend to load nodes from database:

```python
# backend/registry.py

from backend.loaders.node_loader import load_node_from_db
from backend.database import get_db_connection

def load_nodes_from_database(user_id: int):
    """Load user's custom nodes from database."""
    conn = get_db_connection()

    # Query all active nodes for user
    records = conn.execute("""
        SELECT * FROM mosaic_agent_tool_nodes
        WHERE user_id = :user_id
        AND active = true
        ORDER BY created_at DESC
    """, {"user_id": user_id}).fetchall()

    nodes = {}
    for record in records:
        record_dict = dict(record)
        try:
            node = load_node_from_db(record_dict)
            nodes[record_dict['node_uuid']] = node
        except Exception as e:
            print(f"Warning: Failed to load node {record_dict['name']}: {e}")

    return nodes


# In main.py or registry setup
def create_registry(user_id: int):
    """Create registry with user's custom nodes."""
    registry = NodeRegistry()

    # Load from database
    custom_nodes = load_nodes_from_database(user_id)

    for node_uuid, node in custom_nodes.items():
        if isinstance(node, DataHandlerNode):
            registry.register_tool(node_uuid, node)
        elif isinstance(node, ToolNode):
            registry.register_tool(node_uuid, node.tool_fn)

    return registry
```

## Execution with Dependency Injection

In your execution endpoint:

```python
# backend/routers/execution.py

from mesh.nodes import DataHandlerNode, RAGNode

@router.post("/execute")
async def execute_graph(request: ExecuteRequest, req: Request):
    # Parse graph
    graph = parser.parse(request.flow)

    # Inject database session getter into DataHandlerNodes
    from server.resources.database import DBSession

    def get_db_session(source: str):
        source_map = {
            "postgres": DBSession.postgres,
            "mysql": DBSession.master,
            "vertica": DBSession.vrt,
        }
        return source_map.get(source).value()

    for node in graph.nodes.values():
        if isinstance(node, DataHandlerNode):
            node.set_db_session_getter(get_db_session)
        elif isinstance(node, RAGNode):
            node.set_retriever(rag_retriever)

    # Execute
    executor = Executor(graph, MemoryBackend())
    # ...
```

## React Flow UI Integration

The UI loads available DataHandlerNodes from the backend:

```typescript
// Frontend fetches user's custom nodes
const { data: customNodes } = await fetch('/api/nodes/custom');

// Filter DataHandlers for dropdown
const dataHandlers = customNodes.filter(n => n.type === 'DataHandler');

// In node palette
{
  category: "Data Operations",
  nodes: dataHandlers.map(dh => ({
    type: "dataHandlerAgentflow",
    name: dh.node_uuid,
    label: dh.label,
    description: dh.description,
    // ... config from dh.inputs
  }))
}
```

## Example: Creating via API

Endpoint to create DataHandler nodes:

```python
# backend/routers/nodes.py

@router.post("/nodes/data-handler")
async def create_data_handler(
    label: str,
    db_source: str,
    query: str,
    params: Dict[str, Any] = None,
    user_id: int = Depends(get_current_user_id)
):
    """Create a new DataHandler node."""
    node_uuid = str(uuid.uuid4())

    inputs = [
        {
            "name": "db_source",
            "type": "options",
            "default": db_source,
        },
        {
            "name": "query",
            "type": "code",
            "default": query,
        },
        {
            "name": "params",
            "type": "code",
            "default": json.dumps(params or {}),
        }
    ]

    conn.execute("""
        INSERT INTO mosaic_agent_tool_nodes (
            node_uuid, user_id, label, name, type,
            inputs, args, active, is_public
        ) VALUES (
            :node_uuid, :user_id, :label, :name, 'DataHandler',
            :inputs, :args, true, false
        )
    """, {
        "node_uuid": node_uuid,
        "user_id": user_id,
        "label": label,
        "name": label.lower().replace(" ", "_"),
        "inputs": json.dumps(inputs),
        "args": json.dumps(inputs),
    })

    return {"node_uuid": node_uuid, "label": label}
```

## Query Examples

**1. Fixed Parameters (User Config):**
```sql
-- User creates this via UI/API
INSERT INTO mosaic_agent_tool_nodes (..., inputs) VALUES (...,
    '[
        {"name": "db_source", "default": "postgres"},
        {"name": "query", "default": "SELECT * FROM products WHERE category = :category"},
        {"name": "params", "default": "{\"category\": \"electronics\"}"}
    ]'
);
```

**2. AI-Interpolated Parameters:**
```sql
-- No fixed params - AI will provide them
INSERT INTO mosaic_agent_tool_nodes (..., inputs) VALUES (...,
    '[
        {"name": "db_source", "default": "postgres"},
        {"name": "query", "default": "SELECT * FROM users WHERE id = :user_id"},
        {"name": "params", "default": "{}"}
    ]'
);

-- In graph: LLM extracts user_id → DataHandler uses it
```

## Migration Path

**Existing Tool Nodes:**
```sql
-- These stay as-is
SELECT * FROM mosaic_agent_tool_nodes WHERE type = 'Tool';
```

**New DataHandler Nodes:**
```sql
-- Add these with type = 'DataHandler'
INSERT INTO mosaic_agent_tool_nodes (type, ...) VALUES ('DataHandler', ...);
```

**Query Both:**
```sql
-- Backend loads both types
SELECT * FROM mosaic_agent_tool_nodes
WHERE user_id = :user_id
AND type IN ('Tool', 'DataHandler')
AND active = true;
```

## Benefits

✅ **No schema changes** - Uses existing table
✅ **Type discrimination** - `type` field distinguishes DataHandler from Tool
✅ **Backwards compatible** - Existing Tool nodes still work
✅ **Clean separation** - DataHandlers don't need Python `code` field
✅ **Database-driven** - All configs stored in DB, loadable at runtime
✅ **Per-user** - Each user can create their own DataHandlers
