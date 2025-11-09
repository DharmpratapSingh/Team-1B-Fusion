#!/usr/bin/env python3
"""
ClimateGPT MCP Server (TRUE MCP Protocol)
Communicates via stdio using MCP protocol
"""
import asyncio
import calendar
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptMessage,
    PromptArgument,
)

import duckdb
from functools import lru_cache
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("climategpt")

# Initialize MCP server
app = Server("climategpt")

# Load manifest (same logic as mcp_server.py)
manifest_env = os.getenv("MCP_MANIFEST_PATH")
manifest_path = Path(manifest_env) if manifest_env else Path("data/curated-2/manifest_mcp_duckdb.json")

if not manifest_path.exists():
    raise FileNotFoundError(f"Manifest not found at {manifest_path}")

with open(manifest_path, "r") as f:
    MANIFEST = json.load(f)

# Database path resolution (from mcp_server.py)
def _resolve_db_path(db_path: str) -> str:
    """Resolve database path (relative or absolute)"""
    if Path(db_path).is_absolute():
        return db_path
    # Get project root (parent of this file)
    project_root = Path(__file__).parent
    resolved = project_root / db_path
    return str(resolved)

# Get DB path from manifest
first_file = MANIFEST["files"][0] if MANIFEST.get("files") else None
if first_file and first_file.get("path", "").startswith("duckdb://"):
    db_uri = first_file["path"]
    db_path_raw = db_uri[len("duckdb://"):].split("#")[0]
    DB_PATH = _resolve_db_path(db_path_raw)
else:
    DB_PATH = _resolve_db_path("data/warehouse/climategpt.duckdb")

# Connection pooling for DuckDB
from queue import Queue, Empty, Full
from contextlib import contextmanager

class DuckDBConnectionPool:
    """
    Thread-safe connection pool for DuckDB connections.

    Features:
    - Configurable pool size
    - Connection reuse
    - Health checking
    - Automatic connection cleanup
    - Thread-safe operations
    """

    def __init__(self, db_path: str, pool_size: int = 10, max_overflow: int = 5):
        """
        Initialize connection pool.

        Args:
            db_path: Path to DuckDB database
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum connections above pool_size (total = pool_size + max_overflow)
        """
        self.db_path = _resolve_db_path(db_path)
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.max_connections = pool_size + max_overflow

        # Connection pool (FIFO queue)
        self._pool = Queue(maxsize=self.max_connections)
        self._lock = threading.Lock()
        self._connection_count = 0
        self._connections_created = 0
        self._connections_reused = 0

        # Pre-populate pool with initial connections
        logger.info(f"Initializing DuckDB connection pool: size={pool_size}, max_overflow={max_overflow}")
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection."""
        with self._lock:
            if self._connection_count >= self.max_connections:
                raise RuntimeError(f"Maximum connections ({self.max_connections}) reached")

            conn = duckdb.connect(self.db_path, read_only=True)
            self._connection_count += 1
            self._connections_created += 1
            logger.debug(f"Created new connection (total: {self._connection_count})")
            return conn

    def _is_connection_healthy(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Check if connection is still healthy."""
        try:
            # Simple health check query
            conn.execute("SELECT 1").fetchone()
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager).

        Usage:
            with pool.get_connection() as conn:
                result = conn.execute("SELECT * FROM table").fetchall()
        """
        conn = None
        created_new = False

        try:
            # Try to get connection from pool (non-blocking)
            try:
                conn = self._pool.get_nowait()
                self._connections_reused += 1

                # Health check
                if not self._is_connection_healthy(conn):
                    logger.warning("Unhealthy connection detected, creating new one")
                    conn.close()
                    with self._lock:
                        self._connection_count -= 1
                    conn = self._create_connection()
                    created_new = True

            except Empty:
                # Pool is empty, create new connection if allowed
                logger.debug("Pool empty, creating new connection")
                conn = self._create_connection()
                created_new = True

            yield conn

        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    # Try to return to pool
                    self._pool.put_nowait(conn)
                except Full:
                    # Pool is full (overflow connection), close it
                    logger.debug("Pool full, closing overflow connection")
                    conn.close()
                    with self._lock:
                        self._connection_count -= 1

    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "max_connections": self.max_connections,
            "current_connections": self._connection_count,
            "available_connections": self._pool.qsize(),
            "connections_created": self._connections_created,
            "connections_reused": self._connections_reused,
            "reuse_ratio": self._connections_reused / max(1, self._connections_created + self._connections_reused)
        }

    def close_all(self):
        """Close all connections in pool."""
        logger.info("Closing all connections in pool")
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
                with self._lock:
                    self._connection_count -= 1
            except Empty:
                break
        logger.info(f"Connection pool closed (remaining connections: {self._connection_count})")


# Initialize global connection pool
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
POOL_MAX_OVERFLOW = int(os.getenv("DB_POOL_MAX_OVERFLOW", "5"))

_connection_pool = DuckDBConnectionPool(DB_PATH, pool_size=POOL_SIZE, max_overflow=POOL_MAX_OVERFLOW)

def _get_db_connection():
    """
    Get a database connection from the pool.

    Returns a context manager that yields a connection.
    Usage:
        with _get_db_connection() as conn:
            result = conn.execute("SELECT * FROM table").fetchall()
    """
    return _connection_pool.get_connection()

# ========================================
# HELPER FUNCTIONS (from mcp_server.py)
# ========================================

def _validate_file_id(file_id: str) -> tuple[bool, Optional[str]]:
    """Validate file_id format"""
    if not file_id or len(file_id) > 200:
        return False, "file_id must be 1-200 characters"
    if "/" in file_id or ".." in file_id:
        return False, "file_id cannot contain '/' or '..'"
    return True, None

def _find_file_meta(file_id: str):
    """Find file metadata in manifest"""
    for f in MANIFEST.get("files", []):
        if f.get("file_id") == file_id:
            return f
    return None

def _get_table_name(file_meta: dict) -> Optional[str]:
    """Extract table name from file metadata"""
    path = file_meta.get("path", "")
    if path.startswith("duckdb://"):
        return path.split("#")[1] if "#" in path else None
    return None

def _build_where_sql(where: dict[str, Any]) -> tuple[str, list]:
    """Build WHERE clause SQL with parameters"""
    if not where:
        return "", []

    conditions = []
    params = []

    for key, value in where.items():
        if isinstance(value, list):
            placeholders = ",".join(["?"] * len(value))
            conditions.append(f"{key} IN ({placeholders})")
            params.extend(value)
        elif isinstance(value, dict):
            # Support operators like {"$gt": 1000}
            for op, val in value.items():
                if op == "$gt":
                    conditions.append(f"{key} > ?")
                    params.append(val)
                elif op == "$lt":
                    conditions.append(f"{key} < ?")
                    params.append(val)
                elif op == "$gte":
                    conditions.append(f"{key} >= ?")
                    params.append(val)
                elif op == "$lte":
                    conditions.append(f"{key} <= ?")
                    params.append(val)
                elif op == "$ne":
                    conditions.append(f"{key} != ?")
                    params.append(val)
        else:
            conditions.append(f"{key} = ?")
            params.append(value)

    sql = " WHERE " + " AND ".join(conditions) if conditions else ""
    return sql, params

def _validate_column_name(column: str, file_meta: dict) -> tuple[bool, Optional[str]]:
    """Validate column name exists in dataset schema (prevents SQL injection)"""
    if not column:
        return False, "column name required"

    valid_columns = [col["name"] for col in file_meta.get("columns", [])]

    if column not in valid_columns:
        return False, f"Invalid column '{column}'. Valid columns: {', '.join(valid_columns[:10])}"

    return True, None

# ========================================
# TOOLS - Functions LLM can call
# ========================================

@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools"""
    return [
        Tool(
            name="list_emissions_datasets",
            description="List all available emissions datasets with sectors, resolutions, and temporal coverage",
        ),
        Tool(
            name="get_dataset_schema",
            description="Get the schema (columns and types) for a specific dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Dataset identifier (e.g., 'transport-country-year')"
                    }
                },
                "required": ["file_id"]
            }
        ),
        Tool(
            name="query_emissions",
            description="Query emissions data from ClimateGPT database with filters, aggregations, and sorting",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Dataset identifier (e.g., 'transport-country-year')"
                    },
                    "select": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to return (default: all)"
                    },
                    "where": {
                        "type": "object",
                        "description": "Filter conditions (e.g., {'year': 2020, 'country_name': 'United States of America'})"
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to group by"
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Column to sort by (e.g., 'MtCO2 DESC')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return (default: 20, max: 1000)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "aggregations": {
                        "type": "object",
                        "description": "Aggregations to apply (e.g., {'MtCO2': 'sum'})"
                    }
                },
                "required": ["file_id"]
            }
        ),
        Tool(
            name="calculate_yoy_change",
            description="Calculate year-over-year changes in emissions between two years",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                    "key_column": {"type": "string", "description": "Column to group by (e.g., 'country_name')"},
                    "value_column": {"type": "string", "default": "emissions_tonnes"},
                    "base_year": {"type": "integer", "default": 2019},
                    "compare_year": {"type": "integer", "default": 2020},
                    "top_n": {"type": "integer", "default": 10},
                    "direction": {"type": "string", "enum": ["rise", "drop"], "default": "drop"}
                },
                "required": ["file_id", "key_column"]
            }
        ),
        Tool(
            name="analyze_monthly_trends",
            description="Analyze monthly emissions trends for a specific entity (country/region) showing month-over-month changes, averages, and patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Monthly dataset identifier (must end with '-month', e.g., 'transport-country-month')"
                    },
                    "entity_column": {
                        "type": "string",
                        "description": "Column to filter by (e.g., 'country_name', 'admin1_name')"
                    },
                    "entity_value": {
                        "type": "string",
                        "description": "Entity value to analyze (e.g., 'United States of America')"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year to analyze (default: 2020)",
                        "default": 2020
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Column to measure (default: 'MtCO2')",
                        "default": "MtCO2"
                    }
                },
                "required": ["file_id", "entity_column", "entity_value"]
            }
        ),
        Tool(
            name="detect_seasonal_patterns",
            description="Detect seasonal patterns in emissions data by analyzing multi-year monthly averages and identifying peak/low months",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Monthly dataset identifier (e.g., 'transport-country-month')"
                    },
                    "entity_column": {
                        "type": "string",
                        "description": "Column to filter by (e.g., 'country_name')"
                    },
                    "entity_value": {
                        "type": "string",
                        "description": "Entity to analyze (e.g., 'Germany')"
                    },
                    "start_year": {
                        "type": "integer",
                        "description": "Start year for analysis (default: 2015)",
                        "default": 2015
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "End year for analysis (default: 2023)",
                        "default": 2023
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Column to measure (default: 'MtCO2')",
                        "default": "MtCO2"
                    }
                },
                "required": ["file_id", "entity_column", "entity_value"]
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    logger.info(f"Tool called: {name}")
    logger.debug(f"Arguments: {json.dumps(arguments, default=str)[:200]}")

    if name == "list_emissions_datasets":
        files = []
        for file in MANIFEST.get("files", []):
            file_id = file.get("file_id", "")
            files.append({
                "file_id": file_id,
                "name": file.get("name", ""),
                "description": file.get("description", ""),
                "sector": file_id.split("-")[0] if "-" in file_id else "",
                "resolution": file.get("resolution", ""),
                "temporal_coverage": file.get("temporal_coverage", "2000-2023"),
                "units": file.get("units", "tonnes CO₂")
            })
        
        return [TextContent(
            type="text",
            text=json.dumps({"datasets": files}, indent=2)
        )]
    
    elif name == "get_dataset_schema":
        file_id = arguments.get("file_id")
        if not file_id:
            return [TextContent(type="text", text=json.dumps({"error": "file_id required"}))]
        
        valid, error = _validate_file_id(file_id)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        
        file_meta = _find_file_meta(file_id)
        if not file_meta:
            available = [f.get("file_id") for f in MANIFEST.get("files", [])[:10]]
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "file_not_found",
                    "file_id": file_id,
                    "available": available
                })
            )]
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "file_id": file_id,
                "name": file_meta.get("name", ""),
                "description": file_meta.get("description", ""),
                "columns": file_meta.get("columns", []),
                "temporal_coverage": file_meta.get("temporal_coverage", ""),
                "resolution": file_meta.get("resolution", ""),
                "source": file_meta.get("source", "")
            }, indent=2)
        )]
    
    elif name == "query_emissions":
        file_id = arguments.get("file_id")
        if not file_id:
            return [TextContent(type="text", text=json.dumps({"error": "file_id required"}))]
        
        valid, error = _validate_file_id(file_id)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": error}))]
        
        file_meta = _find_file_meta(file_id)
        if not file_meta:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "file_not_found",
                    "file_id": file_id
                })
            )]
        
        # Get table name
        table = _get_table_name(file_meta)
        if not table:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "invalid_table_path", "path": file_meta.get("path")})
            )]
        
        # Build query
        select = arguments.get("select")
        where = arguments.get("where") or {}
        group_by = arguments.get("group_by")
        order_by = arguments.get("order_by")
        limit = min(arguments.get("limit", 20), 1000)
        aggregations = arguments.get("aggregations")
        
        try:
            # Build SELECT
            if select:
                select_sql = ", ".join(select) if isinstance(select, list) else str(select)
            else:
                select_sql = "*"
            
            # Handle aggregations
            if aggregations:
                agg_parts = []
                for col, func in aggregations.items():
                    agg_parts.append(f"{func.upper()}({col}) AS {func}_{col}")
                if group_by:
                    select_sql = ", ".join(group_by) + ", " + ", ".join(agg_parts)
                else:
                    select_sql = ", ".join(agg_parts)
            
            sql = f"SELECT {select_sql} FROM {table}"
            
            # WHERE clause
            where_sql, params = _build_where_sql(where)
            sql += where_sql
            
            # GROUP BY
            if group_by:
                sql += f" GROUP BY {', '.join(group_by)}"
            
            # ORDER BY
            if order_by:
                sql += f" ORDER BY {order_by}"
            
            # LIMIT
            sql += f" LIMIT {limit}"

            # Execute
            with _get_db_connection() as conn:
                result = conn.execute(sql, params).fetchall()
                columns = [desc[0] for desc in conn.description]

                # Convert to dict
                rows = [dict(zip(columns, row)) for row in result]
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "rows": rows,
                    "meta": {
                        "file_id": file_id,
                        "row_count": len(rows),
                        "limit": limit
                    }
                }, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            error_response = {
                "error": "query_failed",
                "detail": str(e)
            }
            # Only expose SQL in debug mode for security
            if os.getenv("DEBUG") == "true":
                error_response["sql"] = sql if 'sql' in locals() else None
            return [TextContent(
                type="text",
                text=json.dumps(error_response)
            )]
    
    elif name == "calculate_yoy_change":
        file_id = arguments.get("file_id")
        key_column = arguments.get("key_column")
        value_column = arguments.get("value_column", "emissions_tonnes")
        base_year = arguments.get("base_year", 2019)
        compare_year = arguments.get("compare_year", 2020)
        top_n = arguments.get("top_n", 10)
        direction = arguments.get("direction", "drop")

        file_meta = _find_file_meta(file_id)
        if not file_meta:
            return [TextContent(type="text", text=json.dumps({"error": "file_not_found"}))]

        table = _get_table_name(file_meta)
        if not table:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_table"}))]

        # Validate column names (security: prevent SQL injection)
        valid, error = _validate_column_name(key_column, file_meta)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_column", "detail": error}))]

        valid, error = _validate_column_name(value_column, file_meta)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_column", "detail": error}))]

        try:
            conn = _get_db_connection(DB_PATH)
            
            # Calculate YoY changes
            sql = f"""
            WITH base AS (
                SELECT {key_column}, {value_column} as base_value
                FROM {table}
                WHERE year = ?
            ),
            compare AS (
                SELECT {key_column}, {value_column} as compare_value
                FROM {table}
                WHERE year = ?
            )
            SELECT 
                b.{key_column} as entity,
                b.base_value,
                c.compare_value,
                (c.compare_value - b.base_value) as change,
                ((c.compare_value - b.base_value) / b.base_value * 100) as change_pct
            FROM base b
            JOIN compare c ON b.{key_column} = c.{key_column}
            WHERE b.base_value > 0
            ORDER BY change {'ASC' if direction == 'drop' else 'DESC'}
            LIMIT ?
            """
            
            result = conn.execute(sql, [base_year, compare_year, top_n]).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            rows = [dict(zip(columns, row)) for row in result]
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "yoy_changes": rows,
                    "meta": {
                        "base_year": base_year,
                        "compare_year": compare_year,
                        "direction": direction
                    }
                }, indent=2, default=str)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "yoy_calculation_failed", "detail": str(e)})
            )]

    elif name == "analyze_monthly_trends":
        file_id = arguments.get("file_id")
        entity_column = arguments.get("entity_column")
        entity_value = arguments.get("entity_value")
        year = arguments.get("year", 2020)
        value_column = arguments.get("value_column", "MtCO2")

        # Validate it's a monthly dataset
        if not file_id.endswith("-month"):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "invalid_dataset",
                    "detail": "file_id must be a monthly dataset (ending with '-month')",
                    "file_id": file_id
                })
            )]

        file_meta = _find_file_meta(file_id)
        if not file_meta:
            return [TextContent(type="text", text=json.dumps({"error": "file_not_found"}))]

        table = _get_table_name(file_meta)
        if not table:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_table"}))]

        # Validate column names (security: prevent SQL injection)
        valid, error = _validate_column_name(entity_column, file_meta)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_column", "detail": error}))]

        valid, error = _validate_column_name(value_column, file_meta)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_column", "detail": error}))]

        try:
            conn = _get_db_connection(DB_PATH)

            # Get monthly data
            sql = f"""
            SELECT month, {value_column}
            FROM {table}
            WHERE {entity_column} = ? AND year = ?
            ORDER BY month
            """

            result = conn.execute(sql, [entity_value, year]).fetchall()

            if not result:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "no_data",
                        "detail": f"No data found for {entity_value} in {year}"
                    })
                )]

            # Calculate statistics
            monthly_data = [{"month": row[0], value_column: row[1]} for row in result]
            values = [row[1] for row in result]

            # Calculate month-over-month changes
            mom_changes = []
            for i in range(1, len(values)):
                change = values[i] - values[i-1]
                change_pct = (change / values[i-1] * 100) if values[i-1] > 0 else 0
                mom_changes.append({
                    "from_month": result[i-1][0],
                    "to_month": result[i][0],
                    "change": float(change),
                    "change_pct": float(change_pct)
                })

            # Find extremes
            min_idx = values.index(min(values))
            max_idx = values.index(max(values))

            analysis = {
                "entity": entity_value,
                "year": year,
                "monthly_data": monthly_data,
                "statistics": {
                    "average": float(sum(values) / len(values)),
                    "total": float(sum(values)),
                    "min": {
                        "month": result[min_idx][0],
                        "value": float(values[min_idx])
                    },
                    "max": {
                        "month": result[max_idx][0],
                        "value": float(values[max_idx])
                    },
                    "range": float(max(values) - min(values)),
                    "std_dev": float(sum((x - sum(values)/len(values))**2 for x in values) / len(values)) ** 0.5
                },
                "month_over_month_changes": mom_changes,
                "insights": []
            }

            # Generate insights
            avg = analysis["statistics"]["average"]
            if values[0] > values[-1] * 1.2:
                analysis["insights"].append(f"Significant decline from January to December ({((values[-1]/values[0]-1)*100):.1f}%)")
            elif values[-1] > values[0] * 1.2:
                analysis["insights"].append(f"Significant increase from January to December ({((values[-1]/values[0]-1)*100):.1f}%)")

            # Check for dramatic drops (like COVID)
            for change in mom_changes:
                if change["change_pct"] < -30:
                    analysis["insights"].append(
                        f"Dramatic drop from month {change['from_month']} to {change['to_month']} ({change['change_pct']:.1f}%)"
                    )

            return [TextContent(
                type="text",
                text=json.dumps(analysis, indent=2, default=str)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "analysis_failed", "detail": str(e)})
            )]

    elif name == "detect_seasonal_patterns":
        file_id = arguments.get("file_id")
        entity_column = arguments.get("entity_column")
        entity_value = arguments.get("entity_value")
        start_year = arguments.get("start_year", 2015)
        end_year = arguments.get("end_year", 2023)
        value_column = arguments.get("value_column", "MtCO2")

        if not file_id.endswith("-month"):
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": "invalid_dataset",
                    "detail": "file_id must be a monthly dataset"
                })
            )]

        file_meta = _find_file_meta(file_id)
        if not file_meta:
            return [TextContent(type="text", text=json.dumps({"error": "file_not_found"}))]

        table = _get_table_name(file_meta)
        if not table:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_table"}))]

        # Validate column names (security: prevent SQL injection)
        valid, error = _validate_column_name(entity_column, file_meta)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_column", "detail": error}))]

        valid, error = _validate_column_name(value_column, file_meta)
        if not valid:
            return [TextContent(type="text", text=json.dumps({"error": "invalid_column", "detail": error}))]

        try:
            conn = _get_db_connection(DB_PATH)

            # Calculate average by month across years
            sql = f"""
            SELECT
                month,
                AVG({value_column}) as avg_value,
                MIN({value_column}) as min_value,
                MAX({value_column}) as max_value,
                COUNT(*) as year_count
            FROM {table}
            WHERE {entity_column} = ?
                AND year >= ?
                AND year <= ?
            GROUP BY month
            ORDER BY month
            """

            result = conn.execute(sql, [entity_value, start_year, end_year]).fetchall()

            if not result:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "no_data",
                        "detail": f"No data found for {entity_value} between {start_year}-{end_year}"
                    })
                )]

            # Build monthly patterns
            month_names = list(calendar.month_abbr)[1:]  # Skip empty string at index 0

            patterns = []
            for row in result:
                patterns.append({
                    "month": int(row[0]),
                    "month_name": month_names[int(row[0]) - 1],
                    "average": float(row[1]),
                    "min": float(row[2]),
                    "max": float(row[3]),
                    "years_included": int(row[4])
                })

            # Find peak and low seasons
            averages = [p["average"] for p in patterns]
            overall_avg = sum(averages) / len(averages)

            peak_months = [p for p in patterns if p["average"] > overall_avg * 1.1]
            low_months = [p for p in patterns if p["average"] < overall_avg * 0.9]

            # Calculate seasonality index (coefficient of variation)
            std_dev = (sum((x - overall_avg)**2 for x in averages) / len(averages)) ** 0.5
            seasonality_index = (std_dev / overall_avg) * 100 if overall_avg > 0 else 0

            analysis = {
                "entity": entity_value,
                "period": f"{start_year}-{end_year}",
                "monthly_patterns": patterns,
                "seasonality": {
                    "overall_average": float(overall_avg),
                    "seasonality_index": float(seasonality_index),
                    "interpretation": "High" if seasonality_index > 15 else "Moderate" if seasonality_index > 8 else "Low"
                },
                "peak_months": [{"month": p["month_name"], "average": p["average"]} for p in peak_months],
                "low_months": [{"month": p["month_name"], "average": p["average"]} for p in low_months],
                "insights": []
            }

            # Generate insights
            if peak_months:
                peak_names = ", ".join([p["month_name"] for p in peak_months])
                analysis["insights"].append(f"Peak emissions typically occur in: {peak_names}")

            if low_months:
                low_names = ", ".join([p["month_name"] for p in low_months])
                analysis["insights"].append(f"Lowest emissions typically occur in: {low_names}")

            if seasonality_index > 15:
                analysis["insights"].append("Strong seasonal pattern detected - emissions vary significantly by month")
            elif seasonality_index < 8:
                analysis["insights"].append("Weak seasonal pattern - emissions are relatively consistent year-round")

            return [TextContent(
                type="text",
                text=json.dumps(analysis, indent=2, default=str)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "seasonal_analysis_failed", "detail": str(e)})
            )]

    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "unknown_tool", "name": name})
        )]


# ========================================
# RESOURCES - Data LLM can access
# ========================================

@app.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="emissions://datasets",
            name="All Emissions Datasets",
            description="List of all available emissions datasets",
            mimeType="application/json"
        )
    ]


@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Handle resource reads"""
    if uri == "emissions://datasets":
        files = [
            {
                "file_id": f.get("file_id", ""),
                "name": f.get("name", ""),
                "description": f.get("description", "")
            }
            for f in MANIFEST.get("files", [])
        ]
        return json.dumps({"datasets": files}, indent=2)
    
    return json.dumps({"error": "resource_not_found", "uri": uri})


# ========================================
# PROMPTS - Pre-defined prompt templates
# ========================================

@app.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """List available prompts"""
    return [
        Prompt(
            name="analyze_emissions",
            description="Analyze emissions trends for a country and sector",
            arguments=[
                PromptArgument(name="country", description="Country name", required=True),
                PromptArgument(name="sector", description="Sector (default: transport)", required=False),
                PromptArgument(name="start_year", description="Start year (default: 2000)", required=False),
                PromptArgument(name="end_year", description="End year (default: 2023)", required=False)
            ]
        ),
        Prompt(
            name="compare_countries",
            description="Compare emissions across multiple countries",
            arguments=[
                PromptArgument(name="countries", description="List of country names", required=True),
                PromptArgument(name="sector", description="Sector (default: transport)", required=False),
                PromptArgument(name="year", description="Year for comparison (default: 2023)", required=False)
            ]
        ),
        Prompt(
            name="analyze_covid_impact",
            description="Analyze COVID-19 pandemic impact on emissions using monthly data",
            arguments=[
                PromptArgument(name="country", description="Country name (default: global analysis)", required=False),
                PromptArgument(name="sector", description="Sector (default: transport)", required=False),
                PromptArgument(name="year", description="Year to analyze (default: 2020)", required=False)
            ]
        )
    ]


@app.get_prompt()
async def handle_get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    """Handle prompt generation"""
    
    if name == "analyze_emissions":
        country = arguments.get("country", "Unknown")
        sector = arguments.get("sector", "transport")
        start_year = arguments.get("start_year", 2000)
        end_year = arguments.get("end_year", 2023)
        
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Analyze {sector} emissions trends for {country} from {start_year} to {end_year}.

Please:
1. Query the emissions data for this country and sector
2. Calculate the total change in emissions
3. Identify the year with highest and lowest emissions
4. Calculate the average annual growth rate
5. Compare to global or regional trends if possible

Provide insights on:
- Main trends (increasing, decreasing, or stable)
- Significant events or changes
- Comparison to climate targets if known
"""
                )
            )
        ]
    
    elif name == "compare_countries":
        countries = arguments.get("countries", [])
        if not isinstance(countries, list):
            countries = [countries]
        countries_str = ", ".join(countries)
        sector = arguments.get("sector", "transport")
        year = arguments.get("year", 2023)
        
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Compare {sector} emissions for {countries_str} in {year}.

Please:
1. Query emissions data for each country
2. Rank countries by total emissions
3. Calculate per-capita emissions if possible
4. Show emissions as percentage of global total
5. Identify trends over the past 5 years

Provide insights on:
- Which country is the largest emitter
- Per-capita comparisons (fairness perspective)
- Recent trends (increasing or decreasing)
- Policy implications
"""
                )
            )
        ]

    elif name == "analyze_covid_impact":
        country = arguments.get("country", "")
        sector = arguments.get("sector", "transport")
        year = arguments.get("year", 2020)

        analysis_scope = f"for {country}" if country else "globally"

        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Analyze the COVID-19 pandemic impact on {sector} emissions {analysis_scope} in {year}.

Please use the monthly trend analysis and seasonal pattern detection tools to:

1. **Monthly Analysis for {year}:**
   - Retrieve monthly emissions data for {year}
   - Identify the dramatic drop in March-April 2020 (lockdown period)
   - Calculate the magnitude of the drop from pre-COVID levels
   - Track the recovery pattern through the rest of the year

2. **Compare with Previous Years:**
   - Compare {year} monthly patterns to 2019 and earlier years
   - Identify deviations from normal seasonal patterns
   - Calculate total annual emissions reduction

3. **Key Metrics:**
   - Peak drop month and percentage decrease
   - Time to recovery (months)
   - Total annual reduction compared to 2019
   - Which months returned to normal vs. stayed depressed

4. **Insights:**
   - What does this tell us about the relationship between economic activity and emissions?
   - Were there any surprising patterns (e.g., sectors that didn't drop as expected)?
   - What lessons can we learn for emissions reduction policies?
   - Did emissions "rebound" above pre-pandemic levels after recovery?

5. **Regional Differences (if analyzing specific country):**
   - How did {country}'s experience compare to global trends?
   - Were there unique factors affecting {country}'s emissions during the pandemic?
   - What policy responses did {country} implement?

Use the analyze_monthly_trends tool with file_id='{sector}-country-month' for detailed month-by-month analysis, and detect_seasonal_patterns to compare {year} against historical norms.
"""
                )
            )
        ]

    return []


# ========================================
# MAIN - Run MCP server
# ========================================

async def main():
    """Run the MCP server via stdio"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())









