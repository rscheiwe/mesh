"""Data Handler Node for database query execution.

This is a specialized ToolNode that executes SQL queries against various
database sources (MySQL, Postgres, etc.) with support for both fixed and
interpolated (AI-generated) parameters.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from mesh.nodes.tool import ToolNode
from mesh.nodes.base import NodeResult
from mesh.core.state import ExecutionContext


class DBSource(str, Enum):
    """Supported database sources."""
    MYSQL = "mysql"
    POSTGRES = "postgres"
    VERTICA = "vertica"
    SQLITE = "sqlite"


class DataHandlerNode(ToolNode):
    """Execute SQL queries against various database sources.

    This is a specialized ToolNode that:
    1. Connects to specified database source
    2. Executes SQL query with fixed or interpolated parameters
    3. Returns query results

    The node can operate in two modes:
    - Fixed parameters: User provides all parameter values in config
    - Interpolated parameters: AI generates values, node executes query

    Example (Fixed Parameters):
        >>> node = DataHandlerNode(
        ...     id="data_handler_0",
        ...     db_source="postgres",
        ...     query="SELECT * FROM users WHERE id = :user_id",
        ...     params={"user_id": 123}
        ... )

    Example (Interpolated - AI provides params):
        >>> node = DataHandlerNode(
        ...     id="data_handler_0",
        ...     db_source="postgres",
        ...     query="SELECT * FROM users WHERE id = :user_id",
        ...     # AI agent will provide user_id in its output
        ... )
    """

    def __init__(
        self,
        id: str,
        db_source: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        db_session_getter: Optional[Any] = None,
        event_mode: str = "full",
        config: Dict[str, Any] = None,
    ):
        """Initialize DataHandlerNode.

        Args:
            id: Node identifier
            db_source: Database source ("mysql", "postgres", "vertica", etc.)
            query: SQL query with named parameters (e.g., "SELECT * WHERE id = :id")
            params: Fixed parameter values (optional - can be interpolated from previous nodes)
            db_session_getter: Function to get database session for given source
            event_mode: Event emission mode
            config: Additional configuration
        """
        # Store data handler specific config
        self.db_source = db_source
        self.query = query
        self.fixed_params = params or {}
        self.db_session_getter = db_session_getter

        # Create the tool function that will execute the query
        tool_fn = self._create_query_executor()

        # Initialize as ToolNode
        super().__init__(
            id=id,
            tool_fn=tool_fn,
            event_mode=event_mode,
            config=config or {},
        )

    def _create_query_executor(self):
        """Create the async function that executes the query.

        This function will be called by ToolNode's execute logic.
        """
        async def execute_query(input: Any, context: ExecutionContext) -> Dict[str, Any]:
            """Execute the SQL query with resolved parameters.

            Args:
                input: Input data (may contain interpolated params)
                context: Execution context

            Returns:
                Query results
            """
            # Resolve parameters (fixed + interpolated)
            resolved_params = self._resolve_params(input, context)

            # Get database session
            if not self.db_session_getter:
                raise RuntimeError(
                    f"DataHandlerNode '{self.id}' has no db_session_getter. "
                    "Call set_db_session_getter() before execution."
                )

            session = self.db_session_getter(self.db_source)

            # Execute query
            try:
                if hasattr(session, 'execute'):
                    # SQLAlchemy-style session
                    from sqlalchemy import text
                    from uuid import UUID
                    from datetime import date, datetime
                    from decimal import Decimal

                    result = session.execute(text(self.query), resolved_params)
                    rows = result.fetchall()

                    # Convert to list of dicts with JSON-serializable values
                    if rows:
                        columns = result.keys()
                        serialized_rows = []
                        for row in rows:
                            row_dict = {}
                            for col, val in zip(columns, row):
                                # Convert non-JSON-serializable types
                                if isinstance(val, UUID):
                                    row_dict[col] = str(val)
                                elif isinstance(val, (date, datetime)):
                                    row_dict[col] = val.isoformat()
                                elif isinstance(val, Decimal):
                                    row_dict[col] = float(val)
                                elif isinstance(val, bytes):
                                    row_dict[col] = val.decode('utf-8', errors='replace')
                                else:
                                    row_dict[col] = val
                            serialized_rows.append(row_dict)

                        return {
                            "rows": serialized_rows,
                            "count": len(rows),
                            "query": self.query,
                            "params": resolved_params,
                        }
                    else:
                        return {
                            "rows": [],
                            "count": 0,
                            "query": self.query,
                            "params": resolved_params,
                        }
                else:
                    raise RuntimeError(f"Unsupported session type: {type(session)}")

            except Exception as e:
                raise RuntimeError(
                    f"Query execution failed in DataHandlerNode '{self.id}': {str(e)}\n"
                    f"Query: {self.query}\n"
                    f"Params: {resolved_params}"
                ) from e
            finally:
                # Close session if needed
                if hasattr(session, 'close'):
                    session.close()

        return execute_query

    def _resolve_params(self, input: Any, context: ExecutionContext) -> Dict[str, Any]:
        """Resolve query parameters from fixed values and interpolated data.

        Priority:
        1. Fixed params (configured at node creation) - HIGHEST PRIORITY
        2. Input data (from previous node - AI-generated) - only if no fixed params
        3. Context variables

        Args:
            input: Input data
            context: Execution context

        Returns:
            Resolved parameter dictionary
        """
        from mesh.utils.variables import VariableResolver

        resolved = {}

        # If we have fixed params, use them and ignore input
        if self.fixed_params:
            resolved.update(self.fixed_params)
        # Otherwise, try to extract params from input
        elif isinstance(input, dict):
            # Check for explicit 'params' key
            if "params" in input:
                resolved.update(input["params"])
            # Otherwise use input dict directly (for AI-interpolated params)
            else:
                resolved.update(input)

        # Resolve any variable templates in param values
        resolver = VariableResolver(context)
        for key, value in list(resolved.items()):
            if isinstance(value, str) and "{{" in value:
                # Resolve template synchronously using the pattern and internal method
                import re
                def replacer(match):
                    variable = match.group(1).strip()
                    try:
                        val = resolver._resolve_single(variable)
                        return str(val) if val is not None else ""
                    except Exception:
                        return ""
                resolved[key] = re.sub(r"\{\{(.*?)\}\}", replacer, value)

        return resolved

    def set_db_session_getter(self, getter: Any) -> None:
        """Inject database session getter for dependency injection.

        Args:
            getter: Function that takes db_source and returns a session
                    e.g., lambda source: DBSession[source].value()
        """
        self.db_session_getter = getter

    def __repr__(self) -> str:
        return f"DataHandlerNode(id='{self.id}', db_source='{self.db_source}')"


# Factory function for creating from database record
def create_data_handler_from_db(
    node_uuid: str,
    code: str,
    inputs: List[Dict[str, Any]],
    label: str,
    **kwargs
) -> DataHandlerNode:
    """Create DataHandlerNode from database record.

    This mirrors the pattern from your SQL example where nodes are stored
    as records in mosaic_agent_tool_nodes table.

    Args:
        node_uuid: UUID from database
        code: Python code (not used for DataHandlerNode, but kept for compatibility)
        inputs: Input configuration from database
        label: Node label
        **kwargs: Additional fields from database

    Returns:
        Configured DataHandlerNode
    """
    # Extract configuration from inputs
    db_source = None
    query = None
    params = {}

    for inp in inputs:
        if inp.get("name") == "db_source":
            db_source = inp.get("default") or inp.get("value")
        elif inp.get("name") == "query":
            query = inp.get("default") or inp.get("value")
        elif inp.get("name") == "params":
            params = inp.get("default") or inp.get("value") or {}

    if not db_source or not query:
        raise ValueError(
            f"DataHandlerNode requires 'db_source' and 'query' in inputs. "
            f"Got inputs: {inputs}"
        )

    return DataHandlerNode(
        id=node_uuid,
        db_source=db_source,
        query=query,
        params=params,
    )
