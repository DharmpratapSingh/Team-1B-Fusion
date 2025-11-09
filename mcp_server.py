import json
import os
import re
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib
import threading
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import hmac

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from dotenv import load_dotenv
import duckdb
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, validator

# ---------------------------------------------------------------------
# Robust .env loading (from the same folder as this file)
# ---------------------------------------------------------------------
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

# Not strictly required for the API to run, but warn if missing
_missing = [k for k, v in {
    "OPENAI_BASE_URL": OPENAI_BASE_URL,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "MODEL": MODEL
}.items() if not v]
if _missing:
    print(f"[mcp_server] Warning: missing env vars in {ENV_PATH.name}: {', '.join(_missing)}")

# ---------------------------------------------------------------------
# App and manifest
# ---------------------------------------------------------------------
app = FastAPI(
    title="ClimateGPT MCP Server",
    version="0.4.0",
    description="""
    ClimateGPT MCP Server provides comprehensive emissions data querying capabilities.
    
    ## Features
    - Query emissions data from multiple sectors (transport, power, agriculture, etc.)
    - Support for country, admin1, and city-level data
    - Annual and monthly temporal resolutions
    - Advanced query features: aggregations, HAVING clauses, computed columns
    - Query suggestions and fuzzy matching
    - Schema validation
    - Performance metrics and telemetry
    
    ## Authentication
    No authentication required for local development.
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS for local dev (frontend / notebooks)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:*",
        "http://127.0.0.1",
        "http://127.0.0.1:*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load manifest (env-configurable path)
manifest_env = os.getenv("MCP_MANIFEST_PATH")
manifest_path = Path(manifest_env) if manifest_env else Path("data/curated-2/manifest_mcp_duckdb.json")
if not manifest_path.exists():
    raise FileNotFoundError(f"Manifest not found at {manifest_path}")
with open(manifest_path, "r") as f:
    MANIFEST = json.load(f)
_MANIFEST_CHECKSUM = hashlib.sha1(json.dumps(MANIFEST, sort_keys=True).encode("utf-8")).hexdigest()[:12]

# ---------------------------------------------------------------------
# Config defaults (env-driven)
# ---------------------------------------------------------------------
ASSIST_DEFAULT = os.getenv("ASSIST_DEFAULT", "true").lower() == "true"
PROXY_DEFAULT = os.getenv("PROXY_DEFAULT", "false").lower() == "true"
PROXY_MAX_K_DEFAULT = int(os.getenv("PROXY_MAX_K", "3") or 3)
PROXY_RADIUS_KM_DEFAULT = int(os.getenv("PROXY_RADIUS_KM", "150") or 150)

# ---------------------------------------------------------------------
# File ID resolver (for future alias support)
# ---------------------------------------------------------------------
def _resolve_file_id(fid: str) -> str:
    """Resolve file_id, potentially applying aliases in the future."""
    # Aliases removed - they referenced non-existent .expanded datasets
    # Keep function for future alias support if needed
    return fid


# ---------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent

def _resolve_db_path(db_path: str) -> str:
    """
    Resolve DuckDB database path to absolute path.
    Converts relative paths to absolute based on project root.
    """
    path_obj = Path(db_path)
    if path_obj.is_absolute():
        return db_path
    # Resolve relative to project root
    resolved = _PROJECT_ROOT / path_obj
    return str(resolved.resolve())


# ---------------------------------------------------------------------
# Logging Infrastructure
# ---------------------------------------------------------------------
def _setup_logging():
    """Setup structured logging with JSON format for production."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json")  # "json" or "text"
    
    # Create logger
    logger = logging.getLogger("climategpt_mcp")
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if log_format == "json":
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                if hasattr(record, "request_id"):
                    log_entry["request_id"] = record.request_id
                if hasattr(record, "query_context"):
                    log_entry["query_context"] = record.query_context
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)
        
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

_logger = _setup_logging()


# ---------------------------------------------------------------------
# Security and Input Validation
# ---------------------------------------------------------------------
# Valid characters for identifiers (file_id, column names)
_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
_MAX_QUERY_COMPLEXITY = {
    "max_columns": 50,
    "max_filters": 20,
    "max_list_items": 100,
    "max_string_length": 500,
    "max_query_size": 10000,  # bytes
}

def _validate_file_id(file_id: str) -> Tuple[bool, Optional[str]]:
    """Validate file_id format and prevent path traversal."""
    if not file_id or not isinstance(file_id, str):
        return False, "file_id must be a non-empty string"
    
    if len(file_id) > 200:
        return False, "file_id too long (max 200 characters)"
    
    # Prevent path traversal
    if '..' in file_id or '/' in file_id or '\\' in file_id:
        return False, "file_id contains invalid characters"
    
    # Check for valid identifier pattern
    if not _IDENTIFIER_PATTERN.match(file_id):
        return False, "file_id contains invalid characters (only alphanumeric, _, -, . allowed)"
    
    return True, None


def _validate_column_name(col: str) -> Tuple[bool, Optional[str]]:
    """Validate column name to prevent SQL injection."""
    if not col or not isinstance(col, str):
        return False, "Column name must be a non-empty string"
    
    if len(col) > 100:
        return False, "Column name too long (max 100 characters)"
    
    # Whitelist approach - only allow safe characters
    if not _IDENTIFIER_PATTERN.match(col):
        return False, "Column name contains invalid characters"
    
    # Prevent SQL keywords (basic check)
    sql_keywords = {'select', 'from', 'where', 'insert', 'update', 'delete', 
                    'drop', 'create', 'alter', 'exec', 'execute', 'union'}
    if col.lower() in sql_keywords:
        return False, f"Column name cannot be SQL keyword: {col}"
    
    return True, None


def _validate_filter_value(value: Any, filter_type: str) -> Tuple[bool, Optional[str]]:
    """Validate filter values for safety and size limits."""
    if filter_type == "list" and isinstance(value, list):
        if len(value) > _MAX_QUERY_COMPLEXITY["max_list_items"]:
            return False, f"Filter list too large (max {_MAX_QUERY_COMPLEXITY['max_list_items']} items)"
        for item in value:
            if isinstance(item, str) and len(item) > _MAX_QUERY_COMPLEXITY["max_string_length"]:
                return False, f"Filter list item too long (max {_MAX_QUERY_COMPLEXITY['max_string_length']} chars)"
    
    if isinstance(value, str):
        if len(value) > _MAX_QUERY_COMPLEXITY["max_string_length"]:
            return False, f"Filter value too long (max {_MAX_QUERY_COMPLEXITY['max_string_length']} chars)"
        # Check for potential injection patterns
        if re.search(r'[;\'"\\]', value):
            return False, "Filter value contains potentially dangerous characters"
    
    return True, None


def _validate_query_complexity(req: "QueryRequest") -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Validate query complexity to prevent DoS."""
    issues = []
    
    # Check column count
    if len(req.select) > _MAX_QUERY_COMPLEXITY["max_columns"]:
        issues.append(f"Too many columns in select (max {_MAX_QUERY_COMPLEXITY['max_columns']})")
    
    # Check filter count
    if len(req.where) > _MAX_QUERY_COMPLEXITY["max_filters"]:
        issues.append(f"Too many filters (max {_MAX_QUERY_COMPLEXITY['max_filters']})")
    
    # Check group_by count
    if len(req.group_by) > _MAX_QUERY_COMPLEXITY["max_columns"]:
        issues.append(f"Too many group_by columns (max {_MAX_QUERY_COMPLEXITY['max_columns']})")
    
    # Validate all column names
    all_columns = set(req.select) | set(req.group_by)
    if req.order_by:
        order_col = req.order_by.split()[0]
        all_columns.add(order_col)
    
    for col in all_columns:
        valid, error = _validate_column_name(col)
        if not valid:
            issues.append(f"Invalid column name '{col}': {error}")
    
    # Validate filter values
    for key, value in req.where.items():
        if isinstance(value, dict):
            if "in" in value and isinstance(value["in"], list):
                valid, error = _validate_filter_value(value["in"], "list")
                if not valid:
                    issues.append(f"Invalid filter value for '{key}': {error}")
        else:
            valid, error = _validate_filter_value(value, "single")
            if not valid:
                issues.append(f"Invalid filter value for '{key}': {error}")
    
    if issues:
        return False, "; ".join(issues), {
            "max_columns": _MAX_QUERY_COMPLEXITY["max_columns"],
            "max_filters": _MAX_QUERY_COMPLEXITY["max_filters"],
        }
    
    return True, None, None


# ---------------------------------------------------------------------
# Error handling helpers
# ---------------------------------------------------------------------
def _parse_duckdb_column_error(error_str: str) -> Optional[Tuple[List[str], List[str]]]:
    """
    Parse DuckDB error to detect invalid column errors.
    Returns (bad_columns, candidate_columns) if it's a column error, None otherwise.
    
    Example: "Binder Error: Referenced column \"x\" not found...\nCandidate bindings: \"a\", \"b\""
    """
    error_lower = error_str.lower()
    
    # Check if it's a column not found error
    if "referenced column" not in error_lower or "not found" not in error_lower:
        return None
    
    # Extract column names from error message
    # Find quoted column names (the ones that don't exist)
    missing_pattern = r'Referenced column\s+"([^"]+)"'
    missing_match = re.search(missing_pattern, error_str, re.IGNORECASE)
    if not missing_match:
        return None
    
    bad_columns = [missing_match.group(1)]
    
    # Extract candidate bindings
    candidate_pattern = r'Candidate bindings:\s*"([^"]+)"(?:\s*,\s*"([^"]+)")*'
    candidate_match = re.search(candidate_pattern, error_str)
    candidate_columns = []
    if candidate_match:
        # Get all quoted values after "Candidate bindings:"
        all_matches = re.findall(r'"([^"]+)"', error_str[error_str.find("Candidate bindings:"):])
        candidate_columns = all_matches if all_matches else []
    
    return bad_columns, candidate_columns


def _error_response(code: str, detail: str, hint: Optional[str] = None, context: Optional[Dict[str, Any]] = None, suggestions: Optional[List[str]] = None, trigger_webhook: bool = True) -> Dict[str, Any]:
    """
    Create a standardized error response with enhanced context.
    
    Args:
        code: Error code (e.g., "file_not_found", "read_failed")
        detail: Detailed error message
        hint: Optional hint for resolving the error
        context: Additional context about the error
        suggestions: List of suggested actions
    
    Returns:
        Standardized error dict
    """
    response: Dict[str, Any] = {
        "error": code,
        "detail": detail,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if hint:
        response["hint"] = hint
    if context:
        response["context"] = context
    if suggestions:
        response["suggestions"] = suggestions
    
    # Trigger error webhook if enabled
    if trigger_webhook:
        _trigger_webhook_event("error", {
            "error_code": code,
            "detail": detail,
            "hint": hint,
            "context": context
        })
    
    return response


# ---------------------------------------------------------------------
# Query Validation and Intent Detection
# ---------------------------------------------------------------------
def _parse_temporal_coverage(coverage_str: str) -> Optional[Tuple[int, int]]:
    """Parse temporal coverage string like '2000-2023' into (start, end)."""
    if not coverage_str or '-' not in coverage_str:
        return None
    try:
        parts = coverage_str.split('-')
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        pass
    return None


def _validate_query_intent(req: "QueryRequest", file_meta: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[List[str]]]:
    """
    Validate query and detect potential issues before execution.
    Returns: (is_valid, warning_message, suggestions_dict, suggestions_list)
    """
    warnings = []
    suggestions_dict: Dict[str, Any] = {}
    suggestions_list: List[str] = []
    
    if not file_meta:
        return False, "File metadata not found", None, ["Check available files via /list_files endpoint"]
    
    # Check temporal coverage
    if "year" in req.where:
        year_val = req.where["year"]
        if isinstance(year_val, int):
            temporal = file_meta.get("temporal_coverage", "")
            coverage = _parse_temporal_coverage(temporal)
            if coverage:
                start, end = coverage
                if year_val < start or year_val > end:
                    warnings.append(f"Year {year_val} outside dataset coverage ({start}-{end})")
                    # Suggest nearest available year
                    nearest = max(start, min(end, year_val))
                    suggestions_dict["nearest_year"] = nearest
                    suggestions_list.append(f"Try year {nearest} (dataset covers {start}-{end})")
    
    # Check spatial coverage for city queries
    if "city" in req.file_id and "country_name" in req.where:
        country = req.where.get("country_name")
        if isinstance(country, str):
            coverage_info = _get_cities_data_coverage()
            available = coverage_info.get("available_countries", [])
            if country not in available:
                warnings.append(f"City data not available for '{country}'")
                suggestions_dict.update(_get_cities_suggestions(country))
                suggestions_list.extend([
                    f"City data available for: {', '.join(available[:5])}",
                    "Try querying at country or admin1 level instead"
                ])
    
    # Check for ambiguous filters
    if not req.where and req.assist:
        warnings.append("No filters specified - returning sample data")
        suggestions_list.append("Add filters like 'year' or 'country_name' to narrow results")
    
    # Check if select columns exist in manifest
    if file_meta.get("columns"):
        manifest_cols = {col.get("name") for col in file_meta.get("columns", []) if isinstance(col, dict)}
        missing_cols = [c for c in req.select if c not in manifest_cols]
        if missing_cols:
            warnings.append(f"Some requested columns may not exist: {missing_cols}")
            suggestions_list.append(f"Available columns: {', '.join(sorted(manifest_cols)[:10])}...")
    
    warning_msg = "; ".join(warnings) if warnings else None
    return True, warning_msg, suggestions_dict if suggestions_dict else None, suggestions_list if suggestions_list else None


def _detect_query_patterns(req: "QueryRequest") -> Dict[str, Any]:
    """Detect query patterns to provide better suggestions."""
    patterns = {
        "is_top_n": False,
        "is_comparison": False,
        "is_trend": False,
        "has_temporal_filter": "year" in req.where or "month" in req.where,
        "has_spatial_filter": any(k in req.where for k in ["country_name", "admin1_name", "city_name"]),
        "needs_aggregation": bool(req.group_by),
    }
    
    # Detect top N pattern
    if req.order_by and "DESC" in req.order_by.upper():
        if req.limit and req.limit <= 20:
            patterns["is_top_n"] = True
    
    # Detect comparison pattern
    if "year" in req.where and isinstance(req.where["year"], dict) and "in" in req.where["year"]:
        if len(req.where["year"]["in"]) == 2:
            patterns["is_comparison"] = True
    
    # Detect trend pattern
    if "year" in req.where or req.group_by and "year" in req.group_by:
        patterns["is_trend"] = True
    
    return patterns


# ---------------------------------------------------------------------
# Request Middleware for Logging
# ---------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Add request ID and logging to all requests."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Log request
    extra = {"request_id": request_id}
    _logger.info(
        f"Request: {request.method} {request.url.path}",
        extra=extra
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        _logger.info(
            f"Response: {response.status_code} | Time: {process_time:.3f}s",
            extra={**extra, "status_code": response.status_code, "process_time": process_time}
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        _logger.error(
            f"Request failed: {str(e)}",
            extra={**extra, "process_time": process_time},
            exc_info=True
        )
        raise


# ---------------------------------------------------------------------
# Connection pooling for DuckDB
# ---------------------------------------------------------------------
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
        _logger.info(f"Initializing DuckDB connection pool: size={pool_size}, max_overflow={max_overflow}")
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
            _logger.debug(f"Created new connection (total: {self._connection_count})")
            return conn

    def _is_connection_healthy(self, conn: duckdb.DuckDBPyConnection) -> bool:
        """Check if connection is still healthy."""
        try:
            # Simple health check query
            conn.execute("SELECT 1").fetchone()
            return True
        except Exception as e:
            _logger.warning(f"Connection health check failed: {e}")
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
                    _logger.warning("Unhealthy connection detected, creating new one")
                    conn.close()
                    with self._lock:
                        self._connection_count -= 1
                    conn = self._create_connection()
                    created_new = True

            except Empty:
                # Pool is empty, create new connection if allowed
                _logger.debug("Pool empty, creating new connection")
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
                    _logger.debug("Pool full, closing overflow connection")
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
        _logger.info(f"Connection pool closed (remaining connections: {self._connection_count})")


# Initialize global connection pool
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
POOL_MAX_OVERFLOW = int(os.getenv("DB_POOL_MAX_OVERFLOW", "5"))

# Get DB path from manifest
first_file = MANIFEST["files"][0] if MANIFEST.get("files") else None
if first_file and first_file.get("path", "").startswith("duckdb://"):
    db_uri = first_file["path"]
    db_path_raw = db_uri[len("duckdb://"):].split("#")[0]
    DB_PATH = _resolve_db_path(db_path_raw)
else:
    DB_PATH = _resolve_db_path("data/warehouse/climategpt.duckdb")

# Create global connection pool
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


# Columns that can be aggregated
AGG_COLUMNS = ["emissions_tonnes", "MtCO2"]


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class QueryRequest(BaseModel):
    file_id: str
    select: List[str] = []
    where: Dict[str, Any] = {}
    group_by: List[str] = []
    order_by: Optional[str] = None
    limit: Optional[int] = 20
    offset: Optional[int] = 0
    assist: Optional[bool] = ASSIST_DEFAULT
    proxy: Optional[bool] = PROXY_DEFAULT
    max_proxy_k: Optional[int] = PROXY_MAX_K_DEFAULT
    proxy_radius_km: Optional[int] = PROXY_RADIUS_KM_DEFAULT
    # Advanced query features
    aggregations: Optional[Dict[str, str]] = None  # {"column": "sum|avg|min|max|count|distinct"}
    having: Optional[Dict[str, Any]] = None  # Post-aggregation filters
    computed_columns: Optional[Dict[str, str]] = None  # {"alias": "expression"}


class DeltaRequest(BaseModel):
    file_id: str
    where: Dict[str, Any] = {}
    key_col: str = "admin1_name"
    value_col: str = "emissions_tonnes"
    base_year: int = 2019
    compare_year: int = 2020
    top_n: int = 10
    direction: str = "drop"  # "drop" or "rise"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@lru_cache(maxsize=16)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="skip",
    )

# ---------------------------------------------------------------------
# Table loader: supports DuckDB or CSV
# ---------------------------------------------------------------------
def _load_table(file_meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Load a table according to the manifest entry.
    Supported:
      - DuckDB URI: duckdb://<absolute_or_relative_db_path>#<table_or_view>
      - CSV path (default)
    """
    engine = file_meta.get("engine")
    if engine == "duckdb":
        uri = file_meta.get("path")
        if not uri:
            raise ValueError("Missing 'path' field in manifest for DuckDB entry")
        if not isinstance(uri, str) or not uri.startswith("duckdb://"):
            raise ValueError(f"Invalid DuckDB URI in manifest: {uri}")
        # parse duckdb://<db>#<table>
        db_path, sep, table = uri[len("duckdb://"):].partition("#")
        if not sep or not db_path or not table:
            raise ValueError(f"DuckDB URI must be duckdb://<db_path>#<table>, got: {uri}")
        # Security: Validate table name
        valid_table, table_error = _validate_column_name(table)
        if not valid_table:
            raise ValueError(f"Invalid table name in manifest: {table_error}")
        with _get_db_connection() as con:
            return con.execute(f"SELECT * FROM {table}").df()
    # default: CSV
    path = file_meta.get("path")
    if not path:
        raise ValueError("Missing 'path' field in manifest")
    return _load_csv(path)


def _validate_aggregation_function(func: str) -> Tuple[bool, Optional[str]]:
    """Validate aggregation function name."""
    valid_functions = {"sum", "avg", "mean", "min", "max", "count", "distinct", "std", "stddev", "variance"}
    if func.lower() not in valid_functions:
        return False, f"Invalid aggregation function: {func}. Allowed: {', '.join(sorted(valid_functions))}"
    return True, None


def _build_aggregation_sql(aggregations: Dict[str, str], select: List[str]) -> Tuple[str, List[str]]:
    """
    Build SQL for aggregations.
    Returns (SQL fragment, list of aggregated columns for SELECT).
    """
    if not aggregations:
        return "", select if select else []
    
    agg_parts = []
    agg_cols = []
    
    for col, func in aggregations.items():
        # Validate column name
        valid_col, col_error = _validate_column_name(col)
        if not valid_col:
            raise ValueError(f"Invalid column in aggregation: {col_error}")
        
        # Validate function
        valid_func, func_error = _validate_aggregation_function(func)
        if not valid_func:
            raise ValueError(func_error)
        
        func_upper = func.upper()
        # Quote column name for safety
        quoted_col = f'"{col}"'
        if func_upper == "DISTINCT":
            agg_parts.append(f"COUNT(DISTINCT {quoted_col}) AS \"{col}_distinct_count\"")
            agg_cols.append(f"{col}_distinct_count")
        elif func_upper == "COUNT":
            # When counting a column that might be in GROUP BY, COUNT(*) is more appropriate
            # But COUNT(col) is also valid - counts non-null values in the group
            agg_parts.append(f"COUNT({quoted_col}) AS \"{col}_count\"")
            agg_cols.append(f"{col}_count")
        elif func_upper in ("AVG", "MEAN"):
            agg_parts.append(f"AVG({quoted_col}) AS \"{col}_avg\"")
            agg_cols.append(f"{col}_avg")
        elif func_upper == "SUM":
            agg_parts.append(f"SUM({quoted_col}) AS \"{col}_sum\"")
            agg_cols.append(f"{col}_sum")
        elif func_upper == "MIN":
            agg_parts.append(f"MIN({quoted_col}) AS \"{col}_min\"")
            agg_cols.append(f"{col}_min")
        elif func_upper == "MAX":
            agg_parts.append(f"MAX({quoted_col}) AS \"{col}_max\"")
            agg_cols.append(f"{col}_max")
        elif func_upper in ("STD", "STDDEV"):
            agg_parts.append(f"STDDEV({quoted_col}) AS \"{col}_stddev\"")
            agg_cols.append(f"{col}_stddev")
        elif func_upper == "VARIANCE":
            agg_parts.append(f"VAR({quoted_col}) AS \"{col}_variance\"")
            agg_cols.append(f"{col}_variance")
    
    # Combine with regular select columns
    all_cols = (select if select else []) + agg_cols
    sql_fragment = ", ".join(agg_parts) if agg_parts else ""
    
    return sql_fragment, all_cols


def _build_having_sql(having: Dict[str, Any]) -> Tuple[str, list]:
    """
    Build SQL HAVING clause for post-aggregation filtering.
    Similar to WHERE but for aggregated columns.
    """
    if not having:
        return "", []
    
    clauses = []
    params = []
    
    for key, val in having.items():
        # Validate column name (can be aggregated column alias)
        valid, error = _validate_column_name(key)
        if not valid:
            raise ValueError(f"Invalid column in HAVING: {error}")
        
        if isinstance(val, dict):
            if "in" in val:
                placeholders = ", ".join(["?" for _ in val["in"]])
                clauses.append(f'"{key}" IN ({placeholders})')
                params.extend(val["in"])
            elif "between" in val:
                lo, hi = val["between"]
                clauses.append(f'"{key}" BETWEEN ? AND ?')
                params.extend([lo, hi])
            elif "gte" in val:
                clauses.append(f'"{key}" >= ?')
                params.append(val["gte"])
            elif "lte" in val:
                clauses.append(f'"{key}" <= ?')
                params.append(val["lte"])
            elif "gt" in val:
                clauses.append(f'"{key}" > ?')
                params.append(val["gt"])
            elif "lt" in val:
                clauses.append(f'"{key}" < ?')
                params.append(val["lt"])
            elif "contains" in val:
                clauses.append(f'CAST("{key}" AS VARCHAR) LIKE ?')
                params.append(f"%{val['contains']}%")
        else:
            # Equality
            clauses.append(f'"{key}" = ?')
            params.append(val)
    
    if not clauses:
        return "", []
    return " HAVING " + " AND ".join(clauses), params


def _validate_computed_expression(expression: str, available_columns: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate computed column expression for security.
    Returns (is_valid, error_message).

    Security checks:
    - Expression length limit
    - AST parsing to detect dangerous operations
    - Only allow safe operations and column references
    """
    import ast

    # Length limit to prevent abuse
    if len(expression) > 500:
        return False, "Expression too long (max 500 characters)"

    # Forbidden patterns (case-insensitive)
    forbidden = ['import', 'exec', 'eval', '__', 'compile', 'globals', 'locals', 'open', 'file']
    expr_lower = expression.lower()
    for pattern in forbidden:
        if pattern in expr_lower:
            return False, f"Forbidden pattern '{pattern}' in expression"

    # Try parsing as AST to validate structure
    try:
        # Prepare expression for AST parsing by replacing df['col'] patterns
        test_expr = expression
        for col in available_columns:
            # Replace df['col'] with a valid identifier for AST parsing
            test_expr = test_expr.replace(f"df['{col}']", f"_col_{col.replace('-', '_')}")
            test_expr = test_expr.replace(f'df["{col}"]', f"_col_{col.replace('-', '_')}")

        tree = ast.parse(test_expr, mode='eval')

        # Allowed node types (safe operations only)
        allowed_nodes = (
            ast.Expression, ast.Expr, ast.Load, ast.Store,
            ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.USub, ast.UAdd,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Num, ast.Constant,  # Literals
            ast.Name,  # Variables (column names)
            ast.Call,  # Function calls (we'll validate function names)
            ast.Subscript, ast.Index, ast.Slice,  # For df[] access
            ast.Attribute,  # For df.column or Series.method
        )

        # Allowed function names
        allowed_functions = {'abs', 'round', 'min', 'max', 'sum', 'float', 'int', 'str'}

        # Walk the AST and check for disallowed operations
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False, f"Disallowed operation: {type(node).__name__}"

            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed_functions:
                        return False, f"Disallowed function: {node.func.id}"

        return True, None

    except SyntaxError as e:
        return False, f"Invalid syntax: {str(e)}"
    except Exception as e:
        return False, f"Expression validation failed: {str(e)}"


def _build_computed_columns_sql(computed_columns: Dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed columns to dataframe.
    Expressions are evaluated as Python code with strict security validation.

    Security features:
    - AST parsing to validate expression structure
    - Whitelist of allowed operations and functions
    - Expression length limits
    - Forbidden pattern detection
    - Sandboxed eval with minimal allowed names
    """
    if not computed_columns:
        return df

    result_df = df.copy()

    for alias, expression in computed_columns.items():
        # Validate alias name
        valid, error = _validate_column_name(alias)
        if not valid:
            raise ValueError(f"Invalid alias in computed column: {error}")

        # Validate expression security
        expr_valid, expr_error = _validate_computed_expression(expression, list(df.columns))
        if not expr_valid:
            raise ValueError(f"Invalid expression for '{alias}': {expr_error}")

        # Simple expression evaluation (basic math operations only)
        # Security: Only allow basic operations on existing columns
        try:
            # Replace column references in expression (re is imported at module level)
            safe_expr = expression
            for col in df.columns:
                if col in safe_expr and f"df['{col}']" not in safe_expr and f'df["{col}"]' not in safe_expr:
                    # Replace simple column name with df reference using word boundaries
                    pattern = r'\b' + re.escape(col) + r'\b'
                    safe_expr = re.sub(pattern, f"df['{col}']", safe_expr)

            # Evaluate with very limited scope - only allow safe math operations
            # Security: __builtins__ is explicitly set to empty dict to disable all built-ins
            allowed_names = {
                "__builtins__": {},  # Critical: disable all built-in functions
                "df": result_df,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "float": float,
                "int": int,
            }
            result_df[alias] = eval(safe_expr, allowed_names, {})
        except Exception as e:
            raise ValueError(f"Error evaluating computed column '{alias}': {str(e)}")

    return result_df


def _build_where_sql(where: Dict[str, Any]) -> Tuple[str, list]:
    """
    Build WHERE clause SQL with parameterized queries.
    Validates column names to prevent SQL injection.
    """
    if not where:
        return "", []
    clauses = []
    params: list[Any] = []
    for col, val in where.items():
        # Security: Validate column name
        valid, error = _validate_column_name(col)
        if not valid:
            raise ValueError(f"Invalid column name '{col}': {error}")
        if isinstance(val, dict):
            if "in" in val and isinstance(val["in"], list):
                placeholders = ",".join(["?"] * len(val["in"]))
                clauses.append(f"{col} IN ({placeholders})")
                params.extend(val["in"])
            elif "between" in val and isinstance(val["between"], (list, tuple)) and len(val["between"]) == 2:
                clauses.append(f"{col} BETWEEN ? AND ?")
                params.extend(list(val["between"]))
            elif "gte" in val:
                clauses.append(f"{col} >= ?")
                params.append(val["gte"])
            elif "lte" in val:
                clauses.append(f"{col} <= ?")
                params.append(val["lte"])
            elif "contains" in val:
                clauses.append(f"CAST({col} AS VARCHAR) ILIKE ?")
                params.append(f"%{val['contains']}%")
        else:
            clauses.append(f"{col} = ?")
            params.append(val)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def _duckdb_pushdown(file_meta: Dict[str, Any], select: List[str], where: Dict[str, Any], group_by: List[str], order_by: Optional[str], limit: Optional[int], offset: Optional[int], aggregations: Optional[Dict[str, str]] = None, having: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    if file_meta.get("engine") != "duckdb":
        return None
    uri = file_meta.get("path")
    db_path, _, table = uri[len("duckdb://"):].partition("#")
    if not table:
        return None
    
    # Security: Validate table name
    valid_table, table_error = _validate_column_name(table)
    if not valid_table:
        raise ValueError(f"Invalid table name in pushdown: {table_error}")
    
    # Security: Validate all column names
    all_cols = set(select) | set(group_by)
    if order_by:
        order_col = order_by.split()[0]
        # For aggregated columns, the order_by might use the aggregated name (e.g., "MtCO2_sum")
        # So we validate the base column name or accept the aggregated name
        if aggregations:
            # Check if it's an aggregated column name
            is_agg_col = False
            for orig_col, func in aggregations.items():
                func_upper = func.upper()
                expected_names = {
                    "SUM": f"{orig_col}_sum",
                    "AVG": f"{orig_col}_avg", "MEAN": f"{orig_col}_avg",
                    "COUNT": f"{orig_col}_count",
                    "DISTINCT": f"{orig_col}_distinct_count",
                    "MIN": f"{orig_col}_min",
                    "MAX": f"{orig_col}_max"
                }
                if func_upper in expected_names and order_col == expected_names[func_upper]:
                    is_agg_col = True
                    break
                elif order_col == orig_col:
                    # Will be converted to aggregated name in SQL
                    is_agg_col = True
                    break
            if not is_agg_col:
                all_cols.add(order_col)
        else:
            all_cols.add(order_col)
    
    for col in all_cols:
        valid, error = _validate_column_name(col)
        if not valid:
            raise ValueError(f"Invalid column name in pushdown: {error}")
    
    # Build SELECT clause with aggregations
    if aggregations:
        agg_sql, agg_cols = _build_aggregation_sql(aggregations, select)
        if agg_sql:
            # Include group_by columns in SELECT when grouping
            group_select_cols = [f'"{col}"' for col in group_by] if group_by else []
            # Combine group_by, regular select, and aggregations
            select_cols = [f'"{col}"' for col in select] if select else []
            # Remove duplicates (group_by columns might also be in select)
            all_select_cols = list(dict.fromkeys(group_select_cols + select_cols))  # Preserves order, removes dupes
            if all_select_cols and agg_sql:
                cols = ", ".join(all_select_cols) + ", " + agg_sql
            elif agg_sql:
                cols = agg_sql
                # If no explicit select but we have group_by, include group_by columns
                if group_by and not select:
                    group_cols = ", ".join([f'"{col}"' for col in group_by])
                    cols = group_cols + ", " + agg_sql if cols else group_cols
            else:
                cols = ", ".join(all_select_cols) if all_select_cols else "*"
        else:
            cols = ", ".join([f'"{col}"' for col in select]) if select else "*"
    else:
        cols = ", ".join([f'"{col}"' for col in select]) if select else "*"
    
    where_sql, params = _build_where_sql(where)
    if group_by:
        quoted_cols = [f'"{col}"' for col in group_by]
        group_sql = f" GROUP BY {', '.join(quoted_cols)}"
    else:
        group_sql = ""
    
    # Add HAVING clause
    having_sql, having_params = _build_having_sql(having) if having else ("", [])
    params.extend(having_params)
    
    order_sql = ""
    if order_by:
        parts = order_by.split()
        col = parts[0]
        dirc = parts[1] if len(parts) > 1 else ""
        
        # If using aggregations, check if order_by column needs to be converted to aggregated name
        if aggregations:
            for orig_col, func in aggregations.items():
                func_upper = func.upper()
                expected_names = {
                    "SUM": f"{orig_col}_sum",
                    "AVG": f"{orig_col}_avg", "MEAN": f"{orig_col}_avg",
                    "COUNT": f"{orig_col}_count",
                    "DISTINCT": f"{orig_col}_distinct_count",
                    "MIN": f"{orig_col}_min",
                    "MAX": f"{orig_col}_max"
                }
                if func_upper in expected_names:
                    expected_name = expected_names[func_upper]
                    # If order_by uses original column name, use aggregated name
                    if col == orig_col:
                        col = expected_name
                        break
                    # If already using aggregated name, keep it
                    elif col == expected_name:
                        break
        
        order_sql = f' ORDER BY "{col}" {dirc}'
    
    limit_sql = ""
    if offset:
        limit_sql += f" OFFSET {int(offset)}"
    if limit:
        limit_sql = f" LIMIT {int(limit)}" + limit_sql
    
    sql = f"SELECT {cols} FROM {table}{where_sql}{group_sql}{having_sql}{order_sql}{limit_sql}"
    with _get_db_connection() as con:
        return con.execute(sql, params).df()


def _duckdb_yoy(
    file_meta: Dict[str, Any],
    key_col: str,
    value_col: str,
    base_year: int,
    compare_year: int,
    extra_where: Dict[str, Any],
    top_n: int,
    direction: str,
) -> Optional[pd.DataFrame]:
    if file_meta.get("engine") != "duckdb":
        return None
    uri = file_meta.get("path")
    db_path, _, table = uri[len("duckdb://"):].partition("#")
    if not table:
        return None
    
    # Security: Validate table name (prevent SQL injection)
    valid_table, table_error = _validate_column_name(table)  # Reuse same validation for table names
    if not valid_table:
        raise ValueError(f"Invalid table name: {table_error}")
    
    # Security: Validate column names
    for col in [key_col, value_col]:
        valid, error = _validate_column_name(col)
        if not valid:
            raise ValueError(f"Invalid column name in yoy: {error}")
    
    # Build WHERE with enforced years
    where = dict(extra_where or {})
    where["year"] = {"in": [base_year, compare_year]}
    where_sql, params = _build_where_sql(where)
    sql = f"""
        WITH t AS (
            SELECT {key_col} AS k, CAST(year AS INT) AS y, SUM({value_col}) AS v
            FROM {table}
            {where_sql}
            GROUP BY {key_col}, y
        )
        SELECT a.k AS key,
               a.v AS base,
               b.v AS compare,
               (a.v - b.v) AS delta,
               CASE WHEN a.v <> 0 THEN (a.v - b.v) / a.v * 100.0 ELSE NULL END AS pct
        FROM t a
        JOIN t b ON a.k = b.k AND a.y = ? AND b.y = ?
        ORDER BY delta { 'DESC' if direction == 'drop' else 'ASC' }
        LIMIT {int(top_n)}
    """
    # Note: join parameters for base/compare years at the end
    with _get_db_connection() as con:
        return con.execute(sql, params + [base_year, compare_year]).df()


def _get_file_meta(file_id: str) -> Optional[Dict[str, Any]]:
    return next((f for f in MANIFEST["files"] if f["file_id"] == file_id), None)


# ---------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------
def _validate_table_schema(file_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that a DuckDB table matches its manifest schema definition.
    
    Returns:
        Dict with validation results:
        - valid: bool
        - errors: List[str]
        - warnings: List[str]
        - details: Dict with column info, row count, etc.
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "details": {}
    }
    
    if not file_meta:
        result["valid"] = False
        result["errors"].append("File metadata not found")
        return result
    
    # Only validate DuckDB tables
    if file_meta.get("engine") != "duckdb":
        result["warnings"].append("Schema validation only supported for DuckDB tables")
        return result
    
    uri = file_meta.get("path")
    if not uri or not uri.startswith("duckdb://"):
        result["valid"] = False
        result["errors"].append(f"Invalid DuckDB URI: {uri}")
        return result
    
    try:
        # Parse URI
        db_path, _, table = uri[len("duckdb://"):].partition("#")
        if not table:
            result["valid"] = False
            result["errors"].append("No table name in URI")
            return result
        
        # Validate table name
        valid_table, table_error = _validate_column_name(table)
        if not valid_table:
            result["valid"] = False
            result["errors"].append(f"Invalid table name: {table_error}")
            return result
        
        resolved_path = _resolve_db_path(db_path)

        with _get_db_connection() as con:
            # Check if table exists
            try:
                test_query = f"SELECT COUNT(*) FROM {table} LIMIT 1"
                con.execute(test_query)
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Table '{table}' does not exist or is not accessible: {str(e)}")
                return result

            # Get actual table schema
            try:
                schema_query = f"DESCRIBE {table}"
                schema_df = con.execute(schema_query).df()
                actual_columns = {}
                for _, row in schema_df.iterrows():
                    col_name = row.iloc[0]
                    col_type = row.iloc[1]
                    actual_columns[col_name] = col_type.upper()
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Failed to get table schema: {str(e)}")
                return result

            # Get row count
            try:
                count_query = f"SELECT COUNT(*) as cnt FROM {table}"
                row_count = con.execute(count_query).fetchone()[0]
                result["details"]["row_count"] = row_count

                if row_count == 0:
                    result["warnings"].append("Table is empty")
            except Exception as e:
                result["warnings"].append(f"Could not get row count: {str(e)}")

            # Get manifest columns
            manifest_columns = {}
            if "columns" in file_meta:
                for col_def in file_meta["columns"]:
                    col_name = col_def.get("name")
                    col_type = col_def.get("type", "").upper()
                    if col_name:
                        manifest_columns[col_name] = col_type

            result["details"]["manifest_columns"] = list(manifest_columns.keys())
            result["details"]["actual_columns"] = list(actual_columns.keys())

            # Validate columns exist
            missing_columns = []
            for col_name in manifest_columns.keys():
                if col_name not in actual_columns:
                    missing_columns.append(col_name)
                    result["valid"] = False
                    result["errors"].append(f"Column '{col_name}' defined in manifest but not found in table")

            # Check for extra columns (warnings, not errors)
            extra_columns = []
            for col_name in actual_columns.keys():
                if col_name not in manifest_columns:
                    extra_columns.append(col_name)
                    result["warnings"].append(f"Column '{col_name}' exists in table but not in manifest")

            # Validate column types (if specified)
            type_mismatches = []
            type_mapping = {
                "VARCHAR": ["VARCHAR", "TEXT", "CHAR"],
                "INTEGER": ["INTEGER", "INT", "BIGINT"],
                "DOUBLE": ["DOUBLE", "FLOAT", "REAL", "NUMERIC"],
            }

            for col_name, expected_type in manifest_columns.items():
                if col_name in actual_columns:
                    actual_type = actual_columns[col_name]
                    # Check if types match (with some flexibility)
                    if expected_type in type_mapping:
                        if actual_type not in type_mapping[expected_type]:
                            type_mismatches.append({
                                "column": col_name,
                                "expected": expected_type,
                                "actual": actual_type
                            })
                            result["warnings"].append(
                                f"Column '{col_name}' type mismatch: expected {expected_type}, got {actual_type}"
                            )
                    elif expected_type != actual_type:
                        type_mismatches.append({
                            "column": col_name,
                            "expected": expected_type,
                            "actual": actual_type
                        })
                        result["warnings"].append(
                            f"Column '{col_name}' type mismatch: expected {expected_type}, got {actual_type}"
                        )

            result["details"]["type_mismatches"] = type_mismatches
            result["details"]["missing_columns"] = missing_columns
            result["details"]["extra_columns"] = extra_columns

            # Validate constraints if available
            if "temporal_coverage" in file_meta and "year" in actual_columns:
                try:
                    coverage = file_meta["temporal_coverage"]
                    if "-" in coverage:
                        start_year, end_year = coverage.split("-")
                        try:
                            start_year = int(start_year.strip())
                            end_year = int(end_year.strip())

                            # Check year range in data
                            year_query = f'SELECT MIN(CAST(year AS INTEGER)) as min_year, MAX(CAST(year AS INTEGER)) as max_year FROM {table} WHERE year IS NOT NULL'
                            year_range = con.execute(year_query).fetchone()
                            if year_range and year_range[0] is not None:
                                min_year = int(year_range[0])
                                max_year = int(year_range[1])

                                result["details"]["year_range"] = {"min": min_year, "max": max_year}
                                result["details"]["expected_year_range"] = {"min": start_year, "max": end_year}

                                if min_year < start_year or max_year > end_year:
                                    result["warnings"].append(
                                        f"Year range in data ({min_year}-{max_year}) extends beyond manifest coverage ({start_year}-{end_year})"
                                    )
                                elif min_year > start_year or max_year < end_year:
                                    result["warnings"].append(
                                        f"Year range in data ({min_year}-{max_year}) is narrower than manifest coverage ({start_year}-{end_year})"
                                    )
                        except (ValueError, TypeError):
                            pass
                except Exception as e:
                    result["warnings"].append(f"Could not validate temporal coverage: {str(e)}")
        
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Schema validation failed: {str(e)}")
        _logger.error(f"Schema validation error: {e}", exc_info=True)
    
    return result


# ---------------------------------------------------------------------
# Query Suggestions Engine
# ---------------------------------------------------------------------
def _get_distinct_values(file_meta: Dict[str, Any], column: str, limit: int = 100) -> List[str]:
    """
    Get distinct values for a column from a table.
    Returns list of distinct values (sorted, limited).
    """
    if not file_meta:
        return []
    
    # Validate column name
    valid, error = _validate_column_name(column)
    if not valid:
        return []
    
    try:
        if file_meta.get("engine") == "duckdb":
            uri = file_meta.get("path")
            db_path, _, table = uri[len("duckdb://"):].partition("#")
            if not table:
                return []
            
            # Validate table name
            valid_table, _ = _validate_column_name(table)
            if not valid_table:
                return []
            
            # Security: validate column again for SQL
            sql = f'SELECT DISTINCT "{column}" FROM {table} WHERE "{column}" IS NOT NULL ORDER BY "{column}" LIMIT {limit}'
            with _get_db_connection() as con:
                df = con.execute(sql).df()
                if len(df) > 0 and column in df.columns:
                    values = df[column].dropna().astype(str).unique().tolist()
                    return sorted(values)[:limit]
        else:
            # CSV fallback - load and get distinct values
            df = _load_table(file_meta)
            if column in df.columns:
                values = df[column].dropna().astype(str).unique().tolist()
                return sorted(values)[:limit]
    except Exception as e:
        _logger.warning(f"Error getting distinct values for {column}: {e}")
        return []
    
    return []


def _fuzzy_match(query: str, options: List[str], limit: int = 5) -> List[str]:
    """
    Find similar strings using simple string matching.
    Returns list of matching options sorted by similarity.
    """
    if not query or not options:
        return []
    
    query_lower = query.lower().strip()
    if not query_lower:
        return options[:limit]
    
    # Simple scoring: exact match > starts with > contains
    scores = []
    for opt in options:
        opt_lower = opt.lower()
        if opt_lower == query_lower:
            scores.append((0, opt))  # Exact match - highest priority
        elif opt_lower.startswith(query_lower):
            scores.append((1, opt))  # Starts with - high priority
        elif query_lower in opt_lower:
            scores.append((2, opt))  # Contains - medium priority
        elif query_lower[:3] in opt_lower or opt_lower[:3] in query_lower:
            scores.append((3, opt))  # Partial match - low priority
    
    # Sort by score, then alphabetically
    scores.sort(key=lambda x: (x[0], x[1].lower()))
    return [opt for _, opt in scores[:limit]]


def _get_suggestions_for_column(file_meta: Dict[str, Any], column: str, query: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Get suggestions for a column, optionally filtered by query string.
    Returns dict with suggestions and metadata.
    """
    if not file_meta:
        return {"suggestions": [], "column": column, "total_available": 0}
    
    # Get all distinct values
    all_values = _get_distinct_values(file_meta, column, limit=500)  # Get more for filtering
    total_available = len(all_values)
    
    if not all_values:
        return {"suggestions": [], "column": column, "total_available": 0}
    
    # If query provided, do fuzzy matching
    if query:
        suggestions = _fuzzy_match(query, all_values, limit=limit)
    else:
        # No query - return first N values
        suggestions = all_values[:limit]
    
    return {
        "suggestions": suggestions,
        "column": column,
        "total_available": total_available,
        "showing": len(suggestions)
    }


def _apply_filter(df: pd.DataFrame, col: str, val: Any) -> pd.DataFrame:
    if col not in df.columns:
        return df
    if isinstance(val, dict):
        if "in" in val:
            return df[df[col].isin(val["in"])]
        if "between" in val:
            lo, hi = val["between"]
            return df[df[col].between(lo, hi, inclusive="both")]
        if "gte" in val:
            return df[df[col] >= val["gte"]]
        if "lte" in val:
            return df[df[col] <= val["lte"]]
        if "contains" in val:
            return df[df[col].astype(str).str.contains(str(val["contains"]), case=False, na=False, regex=False)]
    # default equality
    return df[df[col] == val]


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns and not is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.8 or c in AGG_COLUMNS:
                df[c] = conv


def _get_cities_data_coverage() -> Dict[str, Any]:
    """Get cities dataset coverage information"""
    return {
        "available_countries": [
            "Azerbaijan", "India", "Kazakhstan", "Madagascar", 
            "People's Republic of China", "Samoa", "Somalia", "South Africa",
            "France", "Germany", "United States of America", "United Kingdom",
            "Italy", "Spain", "Japan", "Brazil", "Canada"
        ],
        "total_countries": 17,
        "total_cities": 116,
        "coverage_period": "2000-2023",
        "status": "comprehensive",
        "major_cities_included": [
            "Paris", "London", "New York", "Tokyo", "Berlin", "Rome", "Madrid",
            "São Paulo", "Toronto", "Mumbai", "Beijing"
        ]
    }

def _get_cities_suggestions(country_name: str) -> Dict[str, Any]:
    """Get smart suggestions for unavailable cities data"""
    available_countries = [
        "Azerbaijan", "India", "Kazakhstan", "Madagascar", 
        "People's Republic of China", "Samoa", "Somalia", "South Africa",
        "France", "Germany", "United States of America", "United Kingdom",
        "Italy", "Spain", "Japan", "Brazil", "Canada"
    ]
    
    return {
        "message": f"City data is not available for {country_name}",
        "available_alternatives": [
            "Which Indian city has the highest emissions?",
            "Which Chinese city has the highest emissions?",
            f"What are {country_name}'s total transport emissions by year?"
        ],
        "available_countries": available_countries,
        "suggestions": [
            f"Try asking about one of these countries: {', '.join(available_countries[:3])}",
            "Ask about country-level emissions instead of city-level",
            "Check if the country is available in the country dataset"
        ]
    }

def _response(df: pd.DataFrame, file_id: str, limit: Optional[int]) -> Dict[str, Any]:
    lim = 100 if limit is None else int(limit)
    lim = max(1, min(lim, 1000))
    out = df.head(lim).replace({np.nan: None})
    
    # lightweight ETag based on table + shape
    etag_src = f"{file_id}:{len(df)}:{','.join(df.columns)}".encode("utf-8")
    etag = hashlib.sha1(etag_src).hexdigest()[:16]

    src = _get_file_meta(file_id).get("semantics", {}).get("source")
    response = {
        "rows": out.to_dict(orient="records"),
        "row_count": int(len(df)),
        "meta": {
            "units": _get_file_meta(file_id).get("semantics", {}).get("units", ["tonnes CO2"]),
            "source": src if src else "EDGAR v2024",
            "table_id": file_id,
            "spatial_resolution": _get_file_meta(file_id).get("semantics", {}).get("spatial_resolution"),
            "temporal_resolution": _get_file_meta(file_id).get("semantics", {}).get("temporal_resolution"),
            "etag": etag,
        }
    }
    
    # Add data coverage information for cities dataset
    if "city" in file_id:
        response["data_coverage"] = _get_cities_data_coverage()
        
        # Check if no data returned and provide suggestions
        if len(out) == 0:
            response["suggestions"] = {
                "message": "No city data found for the specified criteria",
                "available_countries": _get_cities_data_coverage()["available_countries"],
                "alternative_queries": [
                    "Which Indian city has the highest emissions?",
                    "Which Chinese city has the highest emissions?",
                    "What are the top 5 cities by emissions globally?"
                ]
            }
    
    return response
# ---------------------------------------------------------------------
# Coverage and Resolver
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _coverage_index() -> Dict[str, List[str]]:
    idx = {"city": set(), "admin1": set(), "country": set()}
    for f in MANIFEST["files"]:
        try:
            df = _load_table(f)
        except Exception:
            continue
        cols = set(df.columns)
        if "city_name" in cols:
            idx["city"].update(df["city_name"].dropna().astype(str).unique().tolist())
        if "admin1_name" in cols:
            idx["admin1"].update(df["admin1_name"].dropna().astype(str).unique().tolist())
        if "country_name" in cols:
            idx["country"].update(df["country_name"].dropna().astype(str).unique().tolist())
    return {k: sorted(v) for k, v in idx.items()}


def _top_matches(name: str, pool: List[str], k: int = 5) -> List[str]:
    nm = (name or "").lower()
    scored = []
    for p in pool:
        pl = p.lower()
        score = 0 if pl == nm else (1 if nm in pl else 2)
        scored.append((p, score, len(p)))
    scored.sort(key=lambda x: (x[1], x[2]))
    return [p for p, _, _ in scored[:k]]


@app.get("/coverage")
def coverage():
    idx = _coverage_index()
    # quick stats per level from first matching manifest file with year
    stats: Dict[str, Dict[str, Any]] = {"city": {}, "admin1": {}, "country": {}}
    for level in ("city","admin1","country"):
        try:
            # find a representative file
            rep = next((f for f in MANIFEST["files"] if (level+"_name") in f.get("columns", [{}]) or True), None)
        except Exception:
            rep = None
        # fallback: skip heavy stats if uncertain
        stats[level] = {"year_min": None, "year_max": None}
    return {
        "levels": {k: {"count": len(v)} for k, v in idx.items()},
        "sample": {k: v[:100] for k, v in idx.items()},
        "stats": stats,
    }


@app.get("/resolve")
def resolve(name: str, level_hint: Optional[str] = None):
    idx = _coverage_index()
    levels = [level_hint] if level_hint in ("city", "admin1", "country") else ("city", "admin1", "country")
    out = []
    for lv in levels:
        out.append({"level": lv, "matches": _top_matches(name, idx.get(lv, []))})
    return {"name": name, "candidates": out}


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    """Comprehensive health check including database validation."""
    health_status = {
        "status": "ok",
        "version": "0.4.0",
        "manifest_path": str(manifest_path),
        "manifest_checksum": _MANIFEST_CHECKSUM,
        "components": {
            "manifest": True,
            "duckdb_file": False,
            "duckdb_read": False,
            "tables_accessible": False,
        },
        "details": {},
    }
    
    # Check manifest
    if not manifest_path.exists():
        health_status["status"] = "degraded"
        health_status["components"]["manifest"] = False
        health_status["details"]["manifest_error"] = "Manifest file not found"
        return health_status
    
    # Check DuckDB database
    duck_ok = True
    try:
        rep = next((f for f in MANIFEST["files"] if f.get("engine") == "duckdb"), None)
        if rep:
            uri = rep.get("path")
            db_path, _, table = uri[len("duckdb://"):].partition("#")
            resolved_path = _resolve_db_path(db_path)
            
            # Check if database file exists
            db_path_obj = Path(resolved_path)
            if not db_path_obj.exists():
                health_status["status"] = "degraded"
                health_status["components"]["duckdb_file"] = False
                health_status["details"]["db_file_error"] = f"Database file not found: {resolved_path}"
                return health_status
            
            # Check file size
            file_size = db_path_obj.stat().st_size
            if file_size == 0:
                health_status["status"] = "degraded"
                health_status["components"]["duckdb_file"] = False
                health_status["details"]["db_file_error"] = "Database file is empty"
                return health_status
            
            health_status["components"]["duckdb_file"] = True
            health_status["details"]["db_file_size"] = file_size
            
            # Test database connectivity and table access
            try:
                # Security: Validate table name
                valid_table, table_error = _validate_column_name(table)
                if not valid_table:
                    health_status["status"] = "degraded"
                    health_status["components"]["duckdb_read"] = False
                    health_status["details"]["db_read_error"] = f"Invalid table name: {table_error}"
                    return health_status
                with _get_db_connection() as con:
                    con.execute(f"SELECT 1 FROM {table} LIMIT 1")
                    health_status["components"]["duckdb_read"] = True
                    health_status["components"]["tables_accessible"] = True
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["components"]["duckdb_read"] = False
                health_status["components"]["tables_accessible"] = False
                health_status["details"]["db_read_error"] = str(e)
        else:
            # No DuckDB files in manifest, which is fine
            health_status["details"]["note"] = "No DuckDB datasets in manifest"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["duckdb_file"] = False
        health_status["details"]["db_error"] = str(e)
    
    # Overall status based on critical components
    if not health_status["components"]["manifest"]:
        health_status["status"] = "error"
    elif not health_status["components"]["duckdb_file"] or not health_status["components"]["duckdb_read"]:
        health_status["status"] = "degraded"
    
    return health_status
@app.get("/list_files")
def list_files():
    """Enumerate available tables with basic descriptors."""
    return MANIFEST["files"]


@app.get("/validate/schema/{file_id}")
def validate_schema(file_id: str, request: Request):
    """
    Validate that a DuckDB table matches its manifest schema definition.
    
    Returns:
        Dict with validation results including:
        - valid: bool
        - errors: List of error messages
        - warnings: List of warning messages
        - details: Dict with column info, row count, type mismatches, etc.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Validate file_id
    valid, error = _validate_file_id(file_id)
    if not valid:
        _logger.warning(f"Invalid file_id in validate_schema: {error}", extra={"request_id": request_id})
        return _error_response(
            "invalid_file_id",
            error,
            "Use /list_files to see available file_ids",
            {"file_id": file_id},
            ["Check available files via /list_files endpoint"]
        )
    
    # Get file metadata
    fid = _resolve_file_id(file_id)
    file_meta = _get_file_meta(fid)
    if not file_meta:
        _logger.warning(f"File not found: {fid}", extra={"request_id": request_id})
        return _error_response(
            "file_not_found",
            f"File '{fid}' not found in manifest",
            "Check available files via /list_files endpoint",
            {"requested_file_id": fid},
            ["Use /list_files to see available datasets"]
        )
    
    # Validate schema
    validation_result = _validate_table_schema(file_meta)
    
    _logger.info(
        f"Schema validation for {fid}: {'valid' if validation_result['valid'] else 'invalid'}",
        extra={
            "request_id": request_id,
            "file_id": fid,
            "errors": len(validation_result.get("errors", [])),
            "warnings": len(validation_result.get("warnings", []))
        }
    )
    
    return validation_result


@app.get("/validate/all")
def validate_all_schemas(request: Request):
    """
    Validate all DuckDB tables in the manifest.
    
    Returns:
        Dict with:
        - total: Total number of files
        - validated: Number of files validated
        - valid: Number of valid schemas
        - invalid: Number of invalid schemas
        - results: List of validation results per file
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    results = []
    valid_count = 0
    invalid_count = 0
    
    for file_meta in MANIFEST.get("files", []):
        file_id = file_meta.get("file_id")
        if not file_id:
            continue
        
        validation_result = _validate_table_schema(file_meta)
        validation_result["file_id"] = file_id
        
        if validation_result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
        
        results.append(validation_result)
    
    summary = {
        "total": len(results),
        "validated": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "results": results
    }
    
    _logger.info(
        f"Schema validation complete: {valid_count} valid, {invalid_count} invalid",
        extra={"request_id": request_id, "total": len(results)}
    )
    
    return summary


@app.get("/get_schema/{file_id}")
def get_schema(file_id: str, request: Request):
    """Return full schema/metadata for a file."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Validate file_id format
    valid, error = _validate_file_id(file_id)
    if not valid:
        _logger.warning(f"Invalid file_id in get_schema: {error}", extra={"request_id": request_id})
        return _error_response(
            "invalid_file_id",
            error,
            "Use /list_files to see available file_ids",
            {"file_id": file_id},
            ["Check available files via /list_files endpoint"]
        )
    
    meta = _get_file_meta(file_id)
    if meta:
        _logger.debug(f"Schema retrieved for {file_id}", extra={"request_id": request_id})
        return meta
    else:
        _logger.warning(f"File not found: {file_id}", extra={"request_id": request_id})
        return _error_response(
            "file_not_found",
            f"File '{file_id}' not found in manifest",
            "Check available files via /list_files endpoint",
            {"requested_file_id": file_id},
            ["Use /list_files to see available datasets"]
        )


@app.get("/suggestions/{file_id}")
def get_suggestions(file_id: str, column: str, request: Request, query: Optional[str] = None, limit: int = 10):
    """
    Get suggestions for values in a specific column.
    
    Args:
        file_id: The dataset file ID
        column: Column name to get suggestions for (e.g., "country_name", "year")
        query: Optional partial query for fuzzy matching (e.g., "Unit" matches "United States")
        limit: Maximum number of suggestions to return (default 10)
    """
    if request:
        request_id = getattr(request.state, "request_id", "unknown")
    else:
        request_id = "unknown"
    
    # Validate file_id
    valid, error = _validate_file_id(file_id)
    if not valid:
        _logger.warning(f"Invalid file_id in suggestions: {error}", extra={"request_id": request_id})
        return _error_response(
            "invalid_file_id",
            error,
            "Use /list_files to see available file_ids",
            {"file_id": file_id},
            ["Check available files via /list_files endpoint"]
        )
    
    # Validate column name
    valid_col, col_error = _validate_column_name(column)
    if not valid_col:
        _logger.warning(f"Invalid column name in suggestions: {col_error}", extra={"request_id": request_id})
        return _error_response(
            "invalid_column",
            col_error,
            "Check available columns via /get_schema endpoint",
            {"column": column},
            ["Use /get_schema/{file_id} to see available columns"]
        )
    
    # Get file metadata
    fid = _resolve_file_id(file_id)
    file_meta = _get_file_meta(fid)
    if not file_meta:
        _logger.warning(f"File not found: {fid}", extra={"request_id": request_id})
        return _error_response(
            "file_not_found",
            f"File '{fid}' not found in manifest",
            "Check available files via /list_files endpoint",
            {"requested_file_id": fid},
            ["Use /list_files to see available datasets"]
        )
    
    # Get suggestions
    try:
        suggestions_data = _get_suggestions_for_column(file_meta, column, query=query, limit=limit)
        _logger.debug(
            f"Suggestions retrieved for {fid}.{column}",
            extra={"request_id": request_id, "suggestion_count": suggestions_data.get("showing", 0)}
        )
        return suggestions_data
    except Exception as e:
        _logger.error(
            f"Error getting suggestions: {e}",
            extra={"request_id": request_id, "file_id": fid, "column": column},
            exc_info=True
        )
        return _error_response(
            "suggestions_failed",
            f"Failed to get suggestions: {str(e)}",
            "Check that the column exists in the dataset",
            {"file_id": fid, "column": column},
            ["Verify column name via /get_schema/{file_id}"]
        )

_RATE_BUCKET: Dict[str, Dict[str, Any]] = {}
_RATE_CAP = int(os.getenv("MCP_RATE_CAP", "60"))  # requests per 5 minutes per IP
_RATE_WINDOW_SEC = 300

# ---------------------------------------------------------------------
# Metrics and Telemetry
# ---------------------------------------------------------------------
_METRICS_STORE: Dict[str, Any] = {
    "queries": [],  # List of query execution records
    "endpoints": {},  # Endpoint usage counts
    "errors": {},  # Error counts by type
    "start_time": datetime.utcnow().isoformat() + "Z"
}

# ---------------------------------------------------------------------
# Webhook Storage and Management
# ---------------------------------------------------------------------
_WEBHOOKS_STORE: Dict[str, Dict[str, Any]] = {}  # webhook_id -> webhook config
_WEBHOOK_DELIVERY_HISTORY: Dict[str, List[Dict[str, Any]]] = {}  # webhook_id -> delivery history
_WEBHOOK_LOCK = threading.Lock()  # Thread-safe access to webhooks
_WEBHOOK_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="webhook")

def _record_query_metric(file_id: str, execution_time_ms: float, row_count: int, success: bool = True):
    """Record a query execution metric."""
    try:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_id": file_id,
            "execution_time_ms": execution_time_ms,
            "row_count": row_count,
            "success": success
        }
        _METRICS_STORE["queries"].append(record)
        
        # Keep only last 1000 queries for memory management
        if len(_METRICS_STORE["queries"]) > 1000:
            _METRICS_STORE["queries"] = _METRICS_STORE["queries"][-1000:]
    except Exception:
        pass  # Fail silently to not break queries


def _record_endpoint_usage(endpoint: str):
    """Record endpoint usage."""
    try:
        if endpoint not in _METRICS_STORE["endpoints"]:
            _METRICS_STORE["endpoints"][endpoint] = 0
        _METRICS_STORE["endpoints"][endpoint] += 1
    except Exception:
        pass


def _record_error(error_type: str):
    """Record an error occurrence."""
    try:
        if error_type not in _METRICS_STORE["errors"]:
            _METRICS_STORE["errors"][error_type] = 0
        _METRICS_STORE["errors"][error_type] += 1
    except Exception:
        pass


def _calculate_percentiles(values: List[float], percentiles: List[float]) -> Dict[float, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {p: 0.0 for p in percentiles}
    
    sorted_vals = sorted(values)
    result = {}
    for p in percentiles:
        idx = int((p / 100.0) * (len(sorted_vals) - 1))
        result[p] = sorted_vals[min(idx, len(sorted_vals) - 1)]
    return result

def _rate_check(ip: str) -> bool:
    now = int(pd.Timestamp.utcnow().timestamp())
    slot = now // _RATE_WINDOW_SEC
    ent = _RATE_BUCKET.get(ip)
    if not ent or ent.get("slot") != slot:
        _RATE_BUCKET[ip] = {"slot": slot, "count": 1}
        return True
    ent["count"] += 1
    return ent["count"] <= _RATE_CAP

def _normalize_country_name(name: str) -> str:
    mapping = {
        "United States": "United States of America",
        "USA": "United States of America",
        "US": "United States of America",
        "China": "People's Republic of China",
        "Russia": "Russian Federation",
        "UK": "United Kingdom",
        "South Korea": "Republic of Korea",
        "North Korea": "Democratic People's Republic of Korea",
    }
    return mapping.get(name, name)

@app.post("/query")
def query(req: QueryRequest, request: Request):
    """
    Filter/aggregate a file by common dimensions.
    
    Supports advanced query features:
    - **Aggregations**: sum, avg, min, max, count, distinct
    - **HAVING clauses**: Post-aggregation filtering  
    - **Computed columns**: Mathematical expressions
    
    Example:
    ```json
    {
        "file_id": "transport-country-year",
        "select": ["country_name", "year"],
        "where": {"year": 2020},
        "aggregations": {"MtCO2": "sum"},
        "group_by": ["country_name"],
        "having": {"MtCO2_sum": {"gte": 100}},
        "order_by": "MtCO2_sum DESC",
        "limit": 10
    }
    ```
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()
    
    # Setup logging context
    query_context = {
        "file_id": req.file_id,
        "select_count": len(req.select),
        "where_count": len(req.where),
        "has_group_by": bool(req.group_by),
    }
    
    # Log query start
    _logger.info(
        f"Query request: file_id={req.file_id}, select={len(req.select)} cols, {len(req.where)} filters",
        extra={"request_id": request_id, "query_context": query_context}
    )
    
    # 1. Security: Validate file_id
    valid_file_id, file_id_error = _validate_file_id(req.file_id)
    if not valid_file_id:
        _logger.warning(f"Invalid file_id: {file_id_error}", extra={"request_id": request_id})
        return _error_response(
            "invalid_file_id",
            file_id_error,
            "Use /list_files to see available file_ids",
            {"file_id": req.file_id},
            ["Check available files via /list_files endpoint"]
        )
    
    # 2. Security: Validate query complexity
    valid_complexity, complexity_error, complexity_info = _validate_query_complexity(req)
    if not valid_complexity:
        _logger.warning(f"Query complexity validation failed: {complexity_error}", extra={"request_id": request_id})
        return _error_response(
            "query_too_complex",
            complexity_error,
            "Reduce query complexity",
            complexity_info,
            ["Reduce number of columns or filters", "Use pagination for large datasets"]
        )
    
    # 3. Rate limiting
    try:
        ip = request.client.host if request and request.client else "unknown"
        if not _rate_check(ip):
            _logger.warning(f"Rate limit exceeded for IP: {ip}", extra={"request_id": request_id})
            return _error_response(
                "rate_limited",
                "Too many requests",
                "Reduce frequency",
                {"ip": ip, "limit": _RATE_CAP, "window_sec": _RATE_WINDOW_SEC},
                [f"Limit: {_RATE_CAP} requests per {_RATE_WINDOW_SEC} seconds"]
            )
    except Exception as e:
        _logger.warning(f"Rate check error: {e}", extra={"request_id": request_id})

    # 4. Limit guard
    if req.limit is not None and int(req.limit) > 5000:
        return _error_response(
            "limit_too_high",
            f"Limit {req.limit} exceeds maximum of 5000",
            "Use limit <= 5000",
            {"requested_limit": req.limit, "max_limit": 5000},
            ["Use pagination with offset for large result sets"]
        )
    
    # 5. Resolve and validate file_id
    fid = _resolve_file_id(req.file_id)
    file_meta = _get_file_meta(fid)
    if not file_meta:
        _logger.warning(f"File not found: {fid}", extra={"request_id": request_id})
        return _error_response(
            "file_not_found",
            f"File '{fid}' not found in manifest",
            "Check available files via /list_files endpoint",
            {"requested_file_id": fid},
            ["Use /list_files to see available datasets", "Check file_id spelling"]
        )
    
    # 6. Query intent validation (warnings and suggestions)
    is_valid_intent, warning_msg, suggestions_dict, suggestions_list = _validate_query_intent(req, file_meta)
    if not is_valid_intent:
        _logger.warning(f"Query intent validation failed: {warning_msg}", extra={"request_id": request_id})
        return _error_response(
            "invalid_query_intent",
            warning_msg or "Query validation failed",
            "Review query parameters",
            suggestions_dict,
            suggestions_list or []
        )
    
    # 7. Detect query patterns (for logging and suggestions)
    patterns = _detect_query_patterns(req)

    # Attempt DuckDB pushdown for performance
    df: pd.DataFrame
    try:
        db_start = time.time()
        df_push = _duckdb_pushdown(
            file_meta, req.select, req.where, req.group_by, req.order_by, 
            req.limit, req.offset, req.aggregations, req.having
        )
        if df_push is not None:
            df = df_push
            _logger.debug(
                f"DuckDB pushdown successful ({time.time() - db_start:.3f}s)",
                extra={"request_id": request_id}
            )
        else:
            df = _load_table(file_meta).copy()
            _logger.debug(
                f"Loaded table via fallback ({time.time() - db_start:.3f}s)",
                extra={"request_id": request_id}
            )
    except Exception as e:
        error_str = str(e)
        _logger.error(
            f"Database read failed: {error_str}",
            extra={"request_id": request_id, "file_id": fid},
            exc_info=True
        )
        
        # Check if this is an invalid column error from DuckDB
        column_error = _parse_duckdb_column_error(error_str)
        if column_error:
            bad_cols, candidate_cols = column_error
            
            # Check which columns in the request are actually invalid
            requested_cols = set(req.select) | set(req.group_by)
            if req.order_by:
                order_col = req.order_by.split()[0]
                requested_cols.add(order_col)
            
            # Find which requested columns are actually bad
            bad_select = [c for c in req.select if c in bad_cols]
            bad_group = [c for c in req.group_by if c in bad_cols]
            bad_order = []
            if req.order_by and req.order_by.split()[0] in bad_cols:
                bad_order.append(req.order_by.split()[0])
            
            detail = f"Invalid columns in select: {bad_select}" if bad_select else ""
            if bad_group:
                detail += f" Invalid columns in group_by: {bad_group}" if detail else f"Invalid columns in group_by: {bad_group}"
            if bad_order:
                detail += f" Invalid column in order_by: {bad_order}" if detail else f"Invalid column in order_by: {bad_order}"
            
            if not detail:
                detail = f"Invalid columns: {bad_cols}"
            
            # Use candidate columns as available columns if we got them
            available_cols = candidate_cols[:20] if candidate_cols else []
            
            _logger.warning(
                f"Invalid columns in DuckDB query: {bad_cols}",
                extra={"request_id": request_id, "file_id": fid, "available_columns": available_cols}
            )
            
            return _error_response(
                "invalid_columns",
                detail,
                "Check available columns via /get_schema endpoint",
                {
                    "bad_select": bad_select,
                    "bad_group_by": bad_group,
                    "bad_order_by": bad_order,
                    "available_columns": sorted(available_cols)
                },
                ["Use /get_schema/{file_id} to see all available columns", "Check column name spelling"]
            )
        
        # Not a column error, return generic read_failed
        _record_error("read_failed")
        return _error_response(
            "read_failed",
            error_str,
            f"Check database path: {file_meta.get('path', 'unknown')}",
            {"file_id": fid, "path": file_meta.get('path', 'unknown')},
            ["Verify database file exists", "Check database connectivity", "Review /health endpoint"]
        )

    # Explicit typing
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    for c in ["emissions_tonnes", "MtCO2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Validate select/group_by columns
    # NOTE: If using aggregations, validation happens after aggregations are applied in fallback path
    # If pushdown succeeded, df already has aggregated columns, so validation is fine
    # If pushdown failed, we'll validate after applying aggregations
    skip_early_validation = (req.aggregations or req.having) and file_meta.get("engine") != "duckdb"
    
    if not skip_early_validation:
        # Note: When using aggregations with DuckDB pushdown, group_by columns are included in SELECT
        # So they should be present in the result dataframe
        valid_cols = set(df.columns)
        
        # Also include aggregated column names in valid_cols for validation
        if req.aggregations:
            for col, func in req.aggregations.items():
                func_upper = func.upper()
                if func_upper == "DISTINCT":
                    valid_cols.add(f"{col}_distinct_count")
                elif func_upper == "COUNT":
                    valid_cols.add(f"{col}_count")
                elif func_upper in ("AVG", "MEAN"):
                    valid_cols.add(f"{col}_avg")
                elif func_upper == "SUM":
                    valid_cols.add(f"{col}_sum")
                elif func_upper == "MIN":
                    valid_cols.add(f"{col}_min")
                elif func_upper == "MAX":
                    valid_cols.add(f"{col}_max")
        
        bad_select = [c for c in req.select if c not in valid_cols]
        bad_group = [c for c in req.group_by if c not in valid_cols]
        
        # Don't fail validation for group_by columns if we're using aggregations and they might be in result
        # (They should be, but if pushdown failed and we fall back, validation might fail)
        if req.aggregations and bad_group:
            # Only fail if group_by columns are actually missing from result
            # This is a warning, not an error, because aggregations should include them
            _logger.warning(
                f"Group_by columns not in result (may be expected with aggregations): {bad_group}",
                extra={"request_id": request_id, "file_id": fid}
            )
            # Don't treat this as an error - continue processing
        
        if bad_select or (bad_group and not req.aggregations):
            detail = f"Invalid columns in select: {bad_select}" if bad_select else ""
            if bad_group:
                detail += f" Invalid columns in group_by: {bad_group}" if detail else f"Invalid columns in group_by: {bad_group}"
            _logger.warning(
                f"Invalid columns requested: select={bad_select}, group_by={bad_group}",
                extra={"request_id": request_id, "file_id": fid}
            )
            return _error_response(
                "invalid_columns",
                detail,
                "Check available columns via /get_schema endpoint",
                {"bad_select": bad_select, "bad_group_by": bad_group, "available_columns": sorted(list(valid_cols))[:20]},
                ["Use /get_schema/{file_id} to see all available columns", "Check column name spelling"]
            )

    # Normalize country name in filters
    if "country_name" in req.where and isinstance(req.where["country_name"], str):
        req.where["country_name"] = _normalize_country_name(req.where["country_name"])

    # Filters
    for k, v in req.where.items():
        df = _apply_filter(df, k, v)

    # Projection (but preserve aggregated columns and group_by columns)
    if req.select:
        # When using aggregations, preserve aggregated columns and group_by columns
        if req.aggregations or req.group_by:
            # Get all columns to keep: select + group_by + aggregated columns
            keep = list(req.select)
            if req.group_by:
                keep.extend([c for c in req.group_by if c not in keep])
            if req.aggregations:
                for col, func in req.aggregations.items():
                    func_upper = func.upper()
                    agg_col_name = {
                        "SUM": f"{col}_sum",
                        "AVG": f"{col}_avg", "MEAN": f"{col}_avg",
                        "COUNT": f"{col}_count",
                        "DISTINCT": f"{col}_distinct_count",
                        "MIN": f"{col}_min",
                        "MAX": f"{col}_max"
                    }.get(func_upper, f"{col}_{func.lower()}")
                    if agg_col_name not in keep:
                        keep.append(agg_col_name)
            # Only keep columns that actually exist in df
            keep = [c for c in keep if c in df.columns]
            if keep:
                df = df[keep]
        else:
            # Normal projection when no aggregations
            keep = [c for c in req.select if c in df.columns]
            df = df[keep] if keep else df

    # Grouping + aggregations (fallback path for non-DuckDB)
    if req.group_by or req.aggregations:
        if req.group_by:
            # Use custom aggregations if provided, else default
            if req.aggregations:
                agg_map = {}
                count_on_groupby = {}  # Track COUNT aggregations on group_by columns
                
                for col, func in req.aggregations.items():
                    if col in df.columns:
                        func_lower = func.lower()
                        # For COUNT on a column that's in group_by, we'll handle separately
                        if func_lower == "count" and col in req.group_by:
                            # COUNT on a grouped column - count rows per group
                            # We'll use size() or count a dummy column
                            count_on_groupby[col] = func
                        elif func_lower in ("sum", "avg", "mean", "min", "max", "count"):
                            agg_map[col] = func_lower
                        elif func_lower == "distinct":
                            agg_map[col] = "nunique"  # Pandas equivalent
                        else:
                            agg_map[col] = "sum"  # Default
                
                if agg_map or count_on_groupby:
                    # If we need to count group_by columns, add a dummy column to count
                    if count_on_groupby:
                        # Use the first available column for counting, or create a dummy
                        dummy_col = None
                        for c in df.columns:
                            if c not in req.group_by and c not in agg_map:
                                dummy_col = c
                                break
                        if dummy_col is None:
                            # Create a dummy column of 1s to count
                            dummy_col = "_temp_count"
                            df[dummy_col] = 1
                        
                        # Count the dummy column for each group_by column that needs counting
                        # Add dummy column to agg_map (only once, even if multiple count_on_groupby)
                        if count_on_groupby and dummy_col not in agg_map:
                            agg_map[dummy_col] = "count"
                    
                    if agg_map:
                        df = df.groupby(req.group_by, dropna=False).agg(agg_map).reset_index()
                        # reset_index() preserves group_by columns, so country_name should be in df now
                    
                    # Handle renaming for COUNT on group_by columns
                    if count_on_groupby and dummy_col:
                        for col in count_on_groupby:
                            if dummy_col in df.columns:
                                df = df.rename(columns={dummy_col: f"{col}_count"})
                                break  # Only rename once for the first count_on_groupby
                
                # Rename aggregated columns to match SQL naming
                for col, func in req.aggregations.items():
                    if col in df.columns:
                        func_lower = func.lower()
                        if func_lower == "distinct":
                            df = df.rename(columns={col: f"{col}_distinct_count"})
                        elif func_lower == "count":
                            df = df.rename(columns={col: f"{col}_count"})
                        elif func_lower in ("avg", "mean"):
                            df = df.rename(columns={col: f"{col}_avg"})
                        elif func_lower == "sum":
                            df = df.rename(columns={col: f"{col}_sum"})
                        elif func_lower == "min":
                            df = df.rename(columns={col: f"{col}_min"})
                        elif func_lower == "max":
                            df = df.rename(columns={col: f"{col}_max"})
            else:
                # Default aggregation for AGG_COLUMNS
                agg_map = {c: "sum" for c in AGG_COLUMNS if c in df.columns}
                if agg_map:
                    df = df.groupby(req.group_by, dropna=False).agg(agg_map).reset_index()
        
        # Apply HAVING clause (post-aggregation filtering)
        if req.having:
            for key, val in req.having.items():
                if key in df.columns:
                    df = _apply_filter(df, key, val)
    
    # Validate columns AFTER aggregations if we skipped earlier validation
    if skip_early_validation:
        valid_cols = set(df.columns)
        # Add aggregated column names to valid_cols
        if req.aggregations:
            for col, func in req.aggregations.items():
                func_upper = func.upper()
                if func_upper == "DISTINCT":
                    valid_cols.add(f"{col}_distinct_count")
                elif func_upper == "COUNT":
                    valid_cols.add(f"{col}_count")
                elif func_upper in ("AVG", "MEAN"):
                    valid_cols.add(f"{col}_avg")
                elif func_upper == "SUM":
                    valid_cols.add(f"{col}_sum")
                elif func_upper == "MIN":
                    valid_cols.add(f"{col}_min")
                elif func_upper == "MAX":
                    valid_cols.add(f"{col}_max")
        
        bad_select = [c for c in req.select if c not in valid_cols and c not in req.group_by]
        bad_group = [c for c in req.group_by if c not in valid_cols]
        
        if bad_select or bad_group:
            detail = f"Invalid columns in select: {bad_select}" if bad_select else ""
            if bad_group:
                detail += f" Invalid columns in group_by: {bad_group}" if detail else f"Invalid columns in group_by: {bad_group}"
            _logger.warning(
                f"Invalid columns after aggregation: select={bad_select}, group_by={bad_group}",
                extra={"request_id": request_id, "file_id": fid}
            )
            return _error_response(
                "invalid_columns",
                detail,
                "Check available columns via /get_schema endpoint",
                {"bad_select": bad_select, "bad_group_by": bad_group, "available_columns": sorted(list(valid_cols))[:20]},
                ["Use /get_schema/{file_id} to see all available columns", "Check column name spelling"]
            )

    # Ordering
    if req.order_by:
        parts = req.order_by.split()
        col = parts[0]
        ascending = not (len(parts) > 1 and parts[1].upper() == "DESC")
        
        # Check if column exists, or if it's an aggregated column name
        col_exists = col in df.columns
        
        # If not found and we have aggregations, check if it's an aggregated column name
        if not col_exists and req.aggregations:
            # Try to find matching aggregated column
            for orig_col, func in req.aggregations.items():
                func_upper = func.upper()
                expected_names = {
                    "SUM": f"{orig_col}_sum",
                    "AVG": f"{orig_col}_avg", "MEAN": f"{orig_col}_avg",
                    "COUNT": f"{orig_col}_count",
                    "DISTINCT": f"{orig_col}_distinct_count",
                    "MIN": f"{orig_col}_min",
                    "MAX": f"{orig_col}_max"
                }
                if func_upper in expected_names:
                    expected_name = expected_names[func_upper]
                    # Column name in order_by might be the original or the aggregated name
                    if col == orig_col and expected_name in df.columns:
                        col = expected_name
                        col_exists = True
                        break
                    elif col == expected_name and expected_name in df.columns:
                        col_exists = True
                        break
        
        if not col_exists:
            _logger.warning(f"Invalid order_by column: {col}", extra={"request_id": request_id})
            return _error_response(
                "invalid_order_by",
                f"Column '{col}' not found for ordering",
                f"Use format: 'column_name' or 'column_name DESC'",
                {"requested_column": col, "available_columns": sorted(list(df.columns))[:20]},
                [f"Available columns: {', '.join(sorted(list(df.columns))[:10])}..."]
            )
        df = df.sort_values(col, ascending=ascending)

    # Add computed columns (after all processing)
    if req.computed_columns:
        try:
            df = _build_computed_columns_sql(req.computed_columns, df)
        except Exception as e:
            _logger.warning(f"Error computing columns: {e}", extra={"request_id": request_id})
            return _error_response(
                "computed_column_error",
                str(e),
                "Check computed column expressions",
                {"computed_columns": list(req.computed_columns.keys())},
                ["Verify column names in expressions", "Use simple math operations only"]
            )
    
    # Pagination slicing (offset then limit) if not already applied by pushdown
    if file_meta.get("engine") != "duckdb":
        try:
            off = int(req.offset or 0)
            if off > 0:
                df = df.iloc[off:]
        except Exception:
            pass

    # Assist mode fallbacks if empty
    if req.assist and len(df) == 0:
        # 1) fuzzy contains on string filters
        fuzzy_where = {}
        for k, v in req.where.items():
            if isinstance(v, str) and v.strip():
                fuzzy_where[k] = {"contains": v.strip()}
            else:
                fuzzy_where[k] = v
        if fuzzy_where and fuzzy_where != req.where:
            try:
                df2 = _load_table(file_meta).copy()
                if "year" in df2.columns:
                    df2["year"] = pd.to_numeric(df2["year"], errors="coerce").astype("Int64")
                if "month" in df2.columns:
                    df2["month"] = pd.to_numeric(df2["month"], errors="coerce").astype("Int64")
                for c in ["emissions_tonnes", "MtCO2"]:
                    if c in df2.columns:
                        df2[c] = pd.to_numeric(df2[c], errors="coerce")
                for k, v in fuzzy_where.items():
                    df2 = _apply_filter(df2, k, v)
                if req.select:
                    keep = [c for c in req.select if c in df2.columns]
                    df2 = df2[keep] if keep else df2
                if req.group_by:
                    agg_map = {c: "sum" for c in AGG_COLUMNS if c in df2.columns}
                    if agg_map:
                        df2 = df2.groupby(req.group_by, dropna=False).agg(agg_map).reset_index()
                if req.order_by:
                    parts = req.order_by.split()
                    col = parts[0]
                    ascending = not (len(parts) > 1 and parts[1].upper() == "DESC")
                    if col in df2.columns:
                        df2 = df2.sort_values(col, ascending=ascending)
                if len(df2) > 0:
                    return _response(df2, fid, req.limit)
            except Exception:
                pass

        # 2) roll level up city->admin1->country based on file_id naming
        def _level(fid: str) -> str:
            if "-city-" in fid or "_city_" in fid:
                return "city"
            if "-admin1-" in fid or "_admin1_" in fid:
                return "admin1"
            return "country"
        def _roll_up(fid: str) -> Optional[str]:
            if _level(fid) == "city":
                return fid.replace("-city-", "-admin1-").replace("_city_", "_admin1_")
            if _level(fid) == "admin1":
                return fid.replace("-admin1-", "-country-").replace("_admin1_", "_country_")
            return None
        up_fid = _roll_up(fid)
        if up_fid:
            up_meta = _get_file_meta(up_fid)
            if up_meta:
                try:
                    df3 = _load_table(up_meta).copy()
                    if "year" in df3.columns:
                        df3["year"] = pd.to_numeric(df3["year"], errors="coerce").astype("Int64")
                    if "month" in df3.columns:
                        df3["month"] = pd.to_numeric(df3["month"], errors="coerce").astype("Int64")
                    for c in ["emissions_tonnes", "MtCO2"]:
                        if c in df3.columns:
                            df3[c] = pd.to_numeric(df3[c], errors="coerce")
                    # strip lower-level filters when rolling up
                    where2 = dict(req.where)
                    if _level(up_fid) == "admin1":
                        where2.pop("city_name", None)
                        where2.pop("city_id", None)
                    if _level(up_fid) == "country":
                        for k in ["city_name","city_id","admin1_name","admin1_geoid"]:
                            where2.pop(k, None)
                    for k, v in where2.items():
                        df3 = _apply_filter(df3, k, v)
                    if req.select:
                        keep = [c for c in req.select if c in df3.columns]
                        df3 = df3[keep] if keep else df3
                    if req.group_by:
                        agg_map = {c: "sum" for c in AGG_COLUMNS if c in df3.columns}
                        if agg_map:
                            df3 = df3.groupby(req.group_by, dropna=False).agg(agg_map).reset_index()
                    if req.order_by:
                        parts = req.order_by.split()
                        col = parts[0]
                        ascending = not (len(parts) > 1 and parts[1].upper() == "DESC")
                        if col in df3.columns:
                            df3 = df3.sort_values(col, ascending=ascending)
                    if len(df3) > 0:
                        out = _response(df3, up_fid, req.limit)
                        out.setdefault("meta", {})["proxy"] = {
                            "method": "rollup",
                            "detail": {"from_level": _level(fid), "to_level": _level(up_fid)},
                            "disclaimer": "Requested place not found; showing higher-level aggregate.",
                        }
                        return out
                except Exception:
                    pass

        # Proxy mode (optional)
        if req.proxy:
            # 2a) nearest-year for same entity
            if "year" in req.where and isinstance(req.where["year"], int):
                try:
                    df_all = _load_table(file_meta).copy()
                    y = int(req.where["year"])
                    if "year" in df_all.columns:
                        df_all["year"] = pd.to_numeric(df_all["year"], errors="coerce").astype("Int64")
                    other = {k: v for k, v in req.where.items() if k != "year"}
                    for k, v in other.items():
                        df_all = _apply_filter(df_all, k, v)
                    if len(df_all) > 0:
                        df_all["dy"] = (df_all["year"].astype(int) - y).abs()
                        near = df_all.sort_values("dy").drop(columns=["dy"])
                        if len(near) > 0:
                            out = _response(near, fid, req.limit)
                            out.setdefault("meta", {})["proxy"] = {
                                "method": "nearest_year",
                                "detail": {"from_year": y, "to_year": int(near.iloc[0]["year"])},
                                "disclaimer": "Nearest available year used due to missing target year.",
                            }
                            return out
                except Exception:
                    pass

            # 2b) nearest-k same-level proxy (name similarity heuristic)
            try:
                k = int(req.max_proxy_k or 3)
                level = "country"
                if "-city-" in fid or "_city_" in fid:
                    level = "city"
                elif "-admin1-" in fid or "_admin1_" in fid:
                    level = "admin1"
                place_col = {"city": "city_name", "admin1": "admin1_name", "country": "country_name"}[level]
                qname = req.where.get(place_col) if isinstance(req.where.get(place_col), str) else None
                if qname:
                    df_same = _load_table(file_meta).copy()
                    for c in ["emissions_tonnes", "MtCO2"]:
                        if c in df_same.columns:
                            df_same[c] = pd.to_numeric(df_same[c], errors="coerce")
                    tokens = [t for t in str(qname).split() if len(t) >= 3]
                    mask = False
                    for t in tokens:
                        mask = (df_same[place_col].astype(str).str.contains(t, case=False, na=False)) | mask
                    pool = df_same[mask] if isinstance(mask, pd.Series) else df_same.head(0)
                    if len(pool) == 0:
                        pool = df_same.sort_values("emissions_tonnes", ascending=False).head(50)
                    pool = pool.sort_values("emissions_tonnes", ascending=False).head(k)
                    if len(pool) > 0:
                        out = _response(pool, fid, req.limit)
                        out.setdefault("meta", {})["proxy"] = {
                            "method": "nearest_k",
                            "detail": {"k": int(min(k, len(pool))), "from_level": level, "to_level": level},
                            "disclaimer": f"No exact {level} match; approximating from similar {level}s (k={int(min(k, len(pool)))}).",
                        }
                        return out
            except Exception:
                pass

        # 3) final preview with no filters
        try:
            if file_meta.get("engine") == "duckdb":
                df4 = _duckdb_pushdown(file_meta, req.select, {}, [], None, 5, None, None, None)
                if df4 is None:
                    df4 = _load_table(file_meta).copy()
            else:
                df4 = _load_table(file_meta).copy()
            return _response(df4, fid, 5)
        except Exception:
            pass

    # Handle empty results with helpful suggestions
    if len(df) == 0:
        _logger.info(
            f"Query returned no results",
            extra={"request_id": request_id, "file_id": fid, "where": req.where}
        )
        
        # Build helpful error response for empty results
        empty_suggestions = []
        empty_context = {"file_id": fid, "filters_applied": list(req.where.keys())}
        suggestions_dict = {}
        
        # Get actual suggestions for filter columns
        for filter_col, filter_val in req.where.items():
            if isinstance(filter_val, str) and filter_val.strip():
                # Get suggestions for this column
                try:
                    col_suggestions = _get_suggestions_for_column(file_meta, filter_col, query=filter_val, limit=5)
                    if col_suggestions.get("suggestions"):
                        suggestions_dict[filter_col] = {
                            "requested": filter_val,
                            "available": col_suggestions["suggestions"],
                            "total_available": col_suggestions.get("total_available", 0)
                        }
                        empty_suggestions.append(
                            f"For '{filter_col}', try: {', '.join(col_suggestions['suggestions'][:3])}"
                        )
                    else:
                        # Column exists but no values match - suggest removing filter
                        all_vals = _get_distinct_values(file_meta, filter_col, limit=10)
                        if all_vals:
                            suggestions_dict[filter_col] = {
                                "requested": filter_val,
                                "sample_values": all_vals,
                                "total_available": len(all_vals)
                            }
                            empty_suggestions.append(
                                f"For '{filter_col}', sample values: {', '.join(all_vals[:5])}"
                            )
                except Exception as e:
                    _logger.warning(f"Error getting suggestions for {filter_col}: {e}", extra={"request_id": request_id})
                    # Fallback to generic suggestion
                    empty_suggestions.append(f"Try a different value for '{filter_col}'")
            elif isinstance(filter_val, dict) and "in" in filter_val and isinstance(filter_val["in"], list):
                # For IN filters, suggest available values
                try:
                    all_vals = _get_distinct_values(file_meta, filter_col, limit=20)
                    if all_vals:
                        suggestions_dict[filter_col] = {
                            "requested_values": filter_val["in"],
                            "sample_available": all_vals[:10],
                            "total_available": len(all_vals)
                        }
                        empty_suggestions.append(
                            f"For '{filter_col}', available values include: {', '.join(all_vals[:5])}"
                        )
                except Exception:
                    empty_suggestions.append(f"Try different values for '{filter_col}'")
            elif isinstance(filter_val, dict) and "between" in filter_val:
                # For range filters, suggest valid range
                try:
                    all_vals = _get_distinct_values(file_meta, filter_col, limit=100)
                    if all_vals and len(all_vals) > 0:
                        # Convert to numeric if possible
                        try:
                            numeric_vals = sorted([float(v) for v in all_vals if v and str(v).replace('.','').replace('-','').isdigit()])
                            if numeric_vals:
                                suggestions_dict[filter_col] = {
                                    "requested_range": filter_val["between"],
                                    "valid_range": [numeric_vals[0], numeric_vals[-1]],
                                    "total_available": len(numeric_vals)
                                }
                                empty_suggestions.append(
                                    f"For '{filter_col}', valid range: {numeric_vals[0]} to {numeric_vals[-1]}"
                                )
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    empty_suggestions.append(f"Try a different range for '{filter_col}'")
            else:
                # Generic filter suggestion
                empty_suggestions.append(f"Try removing or adjusting the '{filter_col}' filter")
        
        # Generic suggestions if no specific suggestions found
        if not empty_suggestions:
            empty_suggestions.append("Try removing or relaxing filters")
            if req.where:
                empty_suggestions.append(f"Current filters: {list(req.where.keys())}")
        
        # Suggest using assist mode
        if not req.assist:
            empty_suggestions.append("Enable assist mode for automatic fallbacks")
        
        empty_context["suggestions"] = suggestions_dict
        
        # Record empty result as query metric (with 0 rows)
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        _record_query_metric(fid, execution_time_ms, 0, success=False)
        _record_error("no_results")
        _record_endpoint_usage("/query")
        
        return _error_response(
            "no_results",
            f"No data found matching the specified criteria",
            "Try adjusting filters or using assist mode",
            empty_context,
            empty_suggestions
        )
    
    # Build response with warnings if any
    response = _response(df, fid, req.limit)
    
    # Add warnings if query validation detected issues
    if warning_msg:
        response.setdefault("meta", {})["warnings"] = warning_msg
        if suggestions_dict:
            response["meta"]["suggestions"] = suggestions_dict
        if suggestions_list:
            response.setdefault("suggestions", []).extend(suggestions_list)
    
    # Add query metadata
    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    response["meta"]["query_patterns"] = patterns
    response["meta"]["execution_time_ms"] = execution_time_ms
    
    # Record query metrics
    row_count = len(df)
    _record_query_metric(fid, execution_time_ms, row_count, success=True)
    _record_endpoint_usage("/query")
    
    # Log successful query
    _logger.info(
        f"Query successful: {len(df)} rows returned in {(time.time() - start_time):.3f}s",
        extra={
            "request_id": request_id,
            "file_id": fid,
            "row_count": len(df),
            "execution_time": time.time() - start_time,
            "patterns": patterns
        }
    )
    
    # Trigger webhook event for query completion
    _trigger_webhook_event("query_complete", {
        "request_id": request_id,
        "file_id": fid,
        "row_count": row_count,
        "execution_time_ms": execution_time_ms,
        "success": True
    })
    
    return response

@app.post("/metrics/yoy")
def yoy(req: DeltaRequest, request: Request):
    """
    Convenience endpoint to compute year-over-year deltas
    (e.g., biggest drop from base_year to compare_year).
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.time()
    
    _logger.info(
        f"YoY request: file_id={req.file_id}, years={req.base_year}-{req.compare_year}",
        extra={"request_id": request_id}
    )
    
    # Validate file_id
    valid, error = _validate_file_id(req.file_id)
    if not valid:
        return _error_response("invalid_file_id", error, "Use /list_files to see available file_ids")
    
    fid = _resolve_file_id(req.file_id)
    file_meta = _get_file_meta(fid)
    if not file_meta:
        _logger.warning(f"File not found for YoY: {fid}", extra={"request_id": request_id})
        return _error_response(
            "file_not_found",
            f"File '{fid}' not found in manifest",
            "Check available files via /list_files endpoint",
            {"requested_file_id": fid},
            ["Use /list_files to see available datasets"]
        )

    # Try DuckDB pushdown first
    try:
        df_push = _duckdb_yoy(
            file_meta,
            req.key_col,
            req.value_col,
            req.base_year,
            req.compare_year,
            req.where,
            req.top_n,
            req.direction,
        )
        if df_push is not None:
            rows = []
            for _, r in df_push.iterrows():
                rows.append({
                    "key": r["key"],
                    "base": float(r["base"]) if r["base"] is not None else None,
                    "compare": float(r["compare"]) if r["compare"] is not None else None,
                    "delta": float(r["delta"]) if r["delta"] is not None else None,
                    "pct": float(r["pct"]) if r["pct"] is not None else None,
                })
            return {
                "rows": rows,
                "row_count": len(rows),
                "base_year": req.base_year,
                "compare_year": req.compare_year,
                "meta": {
                    "units": _get_file_meta(fid).get("semantics", {}).get("units", ["tonnes CO2"]),
                    "source": _get_file_meta(fid).get("semantics", {}).get("source", "EDGAR v2024"),
                    "table_id": fid,
                    "spatial_resolution": _get_file_meta(fid).get("semantics", {}).get("spatial_resolution"),
                    "temporal_resolution": _get_file_meta(fid).get("semantics", {}).get("temporal_resolution")
                }
            }
    except Exception:
        pass

    # Fallback to pandas path
    try:
        df = _load_table(file_meta).copy()
    except Exception as e:
        _logger.error(
            f"YoY read failed: {str(e)}",
            extra={"request_id": request_id, "file_id": fid},
            exc_info=True
        )
        return _error_response(
            "read_failed",
            str(e),
            f"Check database path: {file_meta.get('path', 'unknown')}",
            {"file_id": fid},
            ["Verify database file exists", "Check database connectivity"]
        )
    # Explicit typing
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    for c in ["emissions_tonnes", "MtCO2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply provided filters, but enforce year IN [base, compare]
    filters = dict(req.where)
    filters["year"] = {"in": [req.base_year, req.compare_year]}
    for k, v in filters.items():
        df = _apply_filter(df, k, v)

    # Keep only needed columns
    needed = [req.key_col, "year", req.value_col]
    df = df[[c for c in needed if c in df.columns]]

    # Build pivot per key
    by_key: Dict[str, Dict[int, float]] = {}
    for _, r in df.iterrows():
        key = r[req.key_col]
        yr = int(r["year"])
        val = float(r[req.value_col])
        by_key.setdefault(key, {})[yr] = val

    rows = []
    for k, years in by_key.items():
        b = years.get(req.base_year)
        c = years.get(req.compare_year)
        if b is None or c is None:
            continue
        delta = b - c
        pct = (delta / b) * 100.0 if b else None
        rows.append({"key": k, "base": b, "compare": c, "delta": delta, "pct": pct})

    # Sort by requested direction
    rows.sort(key=lambda x: x["delta"], reverse=(req.direction == "drop"))

    # Shape response similar to /query
    return {
        "rows": rows[: req.top_n],
        "row_count": len(rows),
        "base_year": req.base_year,
        "compare_year": req.compare_year,
        "meta": {
            "units": _get_file_meta(fid).get("semantics", {}).get("units", ["tonnes CO2"]),
            "source": _get_file_meta(fid).get("semantics", {}).get("source", "EDGAR v2024"),
            "table_id": fid,
            "spatial_resolution": _get_file_meta(fid).get("semantics", {}).get("spatial_resolution"),
            "temporal_resolution": _get_file_meta(fid).get("semantics", {}).get("temporal_resolution")
        }
    }

# ---------------------------------------------------------------------
# Convenience endpoints: rankings and trends
# ---------------------------------------------------------------------
class RankingRequest(BaseModel):
    file_id: str
    where: Dict[str, Any] = {}
    year: Optional[int] = None
    metric: str = "MtCO2"  # or emissions_tonnes
    top_n: int = 10
    direction: str = "DESC"


@app.post("/metrics/rankings")
def rankings(req: RankingRequest):
    fid = _resolve_file_id(req.file_id)
    file_meta = _get_file_meta(fid)
    if not file_meta:
        return _error_response("file_not_found", f"File '{fid}' not found in manifest")
    try:
        df = _load_table(file_meta).copy()
    except Exception as e:
        return _error_response("read_failed", str(e))
    # typing
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ["emissions_tonnes", "MtCO2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # filters
    if req.year is not None:
        df = _apply_filter(df, "year", req.year)
    for k, v in req.where.items():
        df = _apply_filter(df, k, v)
    # sort and limit
    metric = req.metric if req.metric in df.columns else ("MtCO2" if "MtCO2" in df.columns else "emissions_tonnes")
    ascending = not (str(req.direction).upper() == "DESC")
    if metric not in df.columns:
        return _error_response("invalid_columns", f"Metric column '{metric}' not found", "Available columns: " + ", ".join(df.columns.tolist()))
    df = df.sort_values(metric, ascending=ascending)
    return _response(df, fid, req.top_n)


class TrendRequest(BaseModel):
    file_id: str
    where: Dict[str, Any] = {}
    metric: str = "MtCO2"
    order: str = "ASC"
    limit_years: Optional[int] = None


@app.post("/metrics/trends")
def trends(req: TrendRequest):
    fid = _resolve_file_id(req.file_id)
    file_meta = _get_file_meta(fid)
    if not file_meta:
        return _error_response("file_not_found", f"File '{fid}' not found in manifest")
    try:
        df = _load_table(file_meta).copy()
    except Exception as e:
        return _error_response("read_failed", str(e))
    if "year" not in df.columns:
        return _error_response("invalid_columns", "Column 'year' not found (required for trends)", "Check available columns via /get_schema endpoint")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ["emissions_tonnes", "MtCO2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for k, v in req.where.items():
        df = _apply_filter(df, k, v)
    metric = req.metric if req.metric in df.columns else ("MtCO2" if "MtCO2" in df.columns else "emissions_tonnes")
    # group by year
    if metric in df.columns:
        df = df.groupby(["year"], dropna=False).agg({metric: "sum"}).reset_index()
    df = df.sort_values("year", ascending=(req.order.upper() != "DESC"))
    if req.limit_years:
        df = df.tail(int(req.limit_years))
    return _response(df, fid, None)

@app.get("/metrics/query")
def get_query_metrics(request: Request):
    """
    Get query performance metrics.
    
    Returns:
        Dict with:
        - total_queries: Total number of queries recorded
        - success_rate: Percentage of successful queries
        - avg_execution_time_ms: Average execution time
        - percentiles: P50, P95, P99 execution times
        - by_file_id: Metrics broken down by file_id
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/metrics/query")
    
    queries = _METRICS_STORE.get("queries", [])
    if not queries:
        return {
            "total_queries": 0,
            "message": "No queries recorded yet"
        }
    
    # Calculate overall stats
    execution_times = [q["execution_time_ms"] for q in queries]
    successful = [q for q in queries if q.get("success", True)]
    failed = [q for q in queries if not q.get("success", True)]
    
    percentiles = _calculate_percentiles(execution_times, [50, 75, 95, 99])
    
    # Stats by file_id
    by_file_id = {}
    for query in queries:
        fid = query.get("file_id", "unknown")
        if fid not in by_file_id:
            by_file_id[fid] = {
                "count": 0,
                "execution_times": [],
                "total_rows": 0,
                "successful": 0,
                "failed": 0
            }
        
        by_file_id[fid]["count"] += 1
        by_file_id[fid]["execution_times"].append(query["execution_time_ms"])
        by_file_id[fid]["total_rows"] += query.get("row_count", 0)
        
        if query.get("success", True):
            by_file_id[fid]["successful"] += 1
        else:
            by_file_id[fid]["failed"] += 1
    
    # Calculate per-file stats
    for fid, stats in by_file_id.items():
        if stats["execution_times"]:
            stats["avg_execution_time_ms"] = sum(stats["execution_times"]) / len(stats["execution_times"])
            stats["percentiles"] = _calculate_percentiles(stats["execution_times"], [50, 95, 99])
        else:
            stats["avg_execution_time_ms"] = 0
            stats["percentiles"] = {50: 0, 95: 0, 99: 0}
        stats["success_rate"] = (stats["successful"] / stats["count"] * 100) if stats["count"] > 0 else 0
        del stats["execution_times"]  # Remove raw data
    
    result = {
        "total_queries": len(queries),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": (len(successful) / len(queries) * 100) if queries else 0,
        "avg_execution_time_ms": sum(execution_times) / len(execution_times) if execution_times else 0,
        "percentiles": percentiles,
        "by_file_id": by_file_id,
        "server_start_time": _METRICS_STORE.get("start_time")
    }
    
    _logger.debug(f"Query metrics retrieved", extra={"request_id": request_id, "total_queries": len(queries)})
    return result


@app.get("/metrics/usage")
def get_usage_metrics(request: Request):
    """
    Get endpoint usage statistics.
    
    Returns:
        Dict with:
        - endpoints: Usage counts per endpoint
        - total_requests: Total number of requests
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/metrics/usage")
    
    endpoints = _METRICS_STORE.get("endpoints", {})
    total = sum(endpoints.values())
    
    return {
        "endpoints": endpoints,
        "total_requests": total,
        "server_start_time": _METRICS_STORE.get("start_time")
    }


@app.get("/metrics/errors")
def get_error_metrics(request: Request):
    """
    Get error statistics.
    
    Returns:
        Dict with:
        - errors: Error counts by type
        - total_errors: Total number of errors
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/metrics/errors")
    
    errors = _METRICS_STORE.get("errors", {})
    total = sum(errors.values())
    
    return {
        "errors": errors,
        "total_errors": total,
        "server_start_time": _METRICS_STORE.get("start_time")
    }


@app.get("/data-coverage")
def data_coverage():
    """Get data coverage information for all datasets"""
    return {
        "cities_dataset": _get_cities_data_coverage(),
        "country_dataset": {
            "status": "comprehensive",
            "total_countries": "200+",
            "coverage_period": "2000-2023",
            "includes_major_countries": True
        },
        "recommendations": [
            "Use country-level data for comprehensive coverage",
            "City-level data available for 8 countries only",
            "Consider expanding cities dataset for better coverage"
        ]
    }

# ---------------------------------------------------------------------
# Webhook Models and Helpers
# ---------------------------------------------------------------------
class WebhookRegistrationRequest(BaseModel):
    """Request to register a webhook."""
    url: str
    events: List[str]  # e.g., ["query_complete", "data_update", "error"]
    secret: Optional[str] = None  # Optional secret for signature verification
    active: bool = True
    description: Optional[str] = None


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook."""
    url: Optional[str] = None
    events: Optional[List[str]] = None
    secret: Optional[str] = None
    active: Optional[bool] = None
    description: Optional[str] = None


def _validate_webhook_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate webhook URL format."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            return False, "URL must use http or https protocol"
        if not parsed.netloc:
            return False, "URL must have a valid hostname"
        return True, None
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def _generate_webhook_signature(payload: str, secret: str) -> str:
    """Generate HMAC SHA256 signature for webhook payload."""
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def _deliver_webhook(webhook_id: str, event: str, payload: Dict[str, Any]):
    """Deliver webhook asynchronously with retry logic."""
    with _WEBHOOK_LOCK:
        if webhook_id not in _WEBHOOKS_STORE:
            return
        webhook = _WEBHOOKS_STORE[webhook_id]
    
    # Check if webhook is active (default to True if not set)
    if not webhook.get("active", True):
        _logger.debug(f"Skipping webhook {webhook_id} - not active")
        return
    
    if event not in webhook.get("events", []):
        return
    
    url = webhook.get("url")
    secret = webhook.get("secret")
    
    # Prepare webhook payload
    webhook_payload = {
        "event": event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "webhook_id": webhook_id,
        "data": payload
    }
    
    # Add signature if secret is provided
    payload_str = json.dumps(webhook_payload, sort_keys=True)
    headers = {"Content-Type": "application/json"}
    if secret:
        signature = _generate_webhook_signature(payload_str, secret)
        headers["X-Webhook-Signature"] = f"sha256={signature}"
    
    # Retry configuration
    max_retries = 3
    retry_delays = [1, 5, 15]  # seconds
    
    def _attempt_delivery(attempt: int) -> Tuple[bool, Optional[str]]:
        """Attempt webhook delivery."""
        try:
            import requests
            response = requests.post(
                url,
                json=webhook_payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return True, None
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except requests.exceptions.RequestException as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    # Attempt delivery with retries
    last_error = None
    for attempt in range(max_retries):
        success, error = _attempt_delivery(attempt)
        if success:
            # Record successful delivery
            with _WEBHOOK_LOCK:
                if webhook_id not in _WEBHOOK_DELIVERY_HISTORY:
                    _WEBHOOK_DELIVERY_HISTORY[webhook_id] = []
                _WEBHOOK_DELIVERY_HISTORY[webhook_id].append({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event": event,
                    "status": "success",
                    "attempt": attempt + 1
                })
                # Keep only last 100 deliveries per webhook
                if len(_WEBHOOK_DELIVERY_HISTORY[webhook_id]) > 100:
                    _WEBHOOK_DELIVERY_HISTORY[webhook_id] = _WEBHOOK_DELIVERY_HISTORY[webhook_id][-100:]
            return
        
        last_error = error
        if attempt < max_retries - 1:
            time.sleep(retry_delays[attempt])
    
    # Record failed delivery
    with _WEBHOOK_LOCK:
        if webhook_id not in _WEBHOOK_DELIVERY_HISTORY:
            _WEBHOOK_DELIVERY_HISTORY[webhook_id] = []
        _WEBHOOK_DELIVERY_HISTORY[webhook_id].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "status": "failed",
            "error": last_error,
            "attempts": max_retries
        })


def _trigger_webhook_event(event: str, payload: Dict[str, Any]):
    """Trigger webhook event for all registered webhooks."""
    with _WEBHOOK_LOCK:
        webhook_ids = list(_WEBHOOKS_STORE.keys())
    
    for webhook_id in webhook_ids:
        _WEBHOOK_EXECUTOR.submit(_deliver_webhook, webhook_id, event, payload)


class BatchQueryRequest(BaseModel):
    """Request for batch query execution."""
    queries: List[QueryRequest]
    stop_on_error: bool = False  # If True, stop processing on first error
    parallel: bool = False  # If True, execute queries in parallel


@app.post("/batch/query")
def batch_query(req: BatchQueryRequest, request: Request):
    """
    Execute multiple queries in a single request.
    
    Returns results for each query with success/error status.
    Useful for executing multiple related queries efficiently.
    
    Args:
        queries: List of QueryRequest objects to execute (max 50)
        stop_on_error: If True, stop processing remaining queries on first error
    
    Returns:
        Dict with:
        - summary: Total queries, successful, failed counts
        - results: List of query results or errors
    
    Example:
    ```json
    {
        "queries": [
            {"file_id": "transport-country-year", "where": {"year": 2020}, "limit": 5},
            {"file_id": "power-country-year", "where": {"year": 2020}, "limit": 5}
        ],
        "stop_on_error": false
    }
    ```
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/batch/query")
    
    if not req.queries:
        return _error_response(
            "invalid_request",
            "No queries provided",
            "Provide at least one query in the queries array",
            {},
            ["Include queries array with at least one QueryRequest"]
        )
    
    if len(req.queries) > 50:  # Limit batch size
        return _error_response(
            "batch_too_large",
            f"Batch size {len(req.queries)} exceeds maximum of 50",
            "Split into smaller batches",
            {"batch_size": len(req.queries), "max_batch_size": 50},
            ["Execute queries in batches of 50 or fewer"]
        )
    
    results = []
    successful = 0
    failed = 0
    
    def _execute_single_query(i: int, query_req: QueryRequest) -> Dict[str, Any]:
        """Execute a single query and return result with index."""
        try:
            result = query(query_req, request)
            
            if isinstance(result, dict) and "error" in result:
                return {
                    "index": i,
                    "status": "error",
                    "error": result.get("error"),
                    "detail": result.get("detail"),
                    "query": {
                        "file_id": query_req.file_id,
                        "select": query_req.select[:3] if query_req.select else []
                    }
                }
            else:
                return {
                    "index": i,
                    "status": "success",
                    "data": result
                }
        except Exception as e:
            _logger.error(
                f"Batch query {i} failed: {e}",
                extra={"request_id": request_id, "query_index": i},
                exc_info=True
            )
            return {
                "index": i,
                "status": "error",
                "error": "execution_failed",
                "detail": str(e),
                "query": {
                    "file_id": query_req.file_id,
                    "select": query_req.select[:3] if query_req.select else []
                }
            }
    
    # Execute queries in parallel or sequentially
    if req.parallel and len(req.queries) > 1:
        # Parallel execution using ThreadPoolExecutor
        batch_executor = ThreadPoolExecutor(max_workers=min(10, len(req.queries)))
        future_to_index = {
            batch_executor.submit(_execute_single_query, i, query_req): i
            for i, query_req in enumerate(req.queries)
        }
        
        for future in as_completed(future_to_index):
            result = future.result()
            results.append(result)
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
                if req.stop_on_error:
                    # Cancel remaining futures
                    for f in future_to_index:
                        if not f.done():
                            f.cancel()
                    break
        
        batch_executor.shutdown(wait=True)
        # Sort results by index to maintain order
        results.sort(key=lambda x: x["index"])
    else:
        # Sequential execution (original logic)
        for i, query_req in enumerate(req.queries):
            result = _execute_single_query(i, query_req)
            results.append(result)
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
                if req.stop_on_error:
                    break
    
    # Trigger webhook event for batch completion
    _trigger_webhook_event("batch_complete", {
        "batch_id": request_id,
        "total_queries": len(req.queries),
        "successful": successful,
        "failed": failed,
        "parallel": req.parallel
    })
    
    return {
        "summary": {
            "total": len(req.queries),
            "successful": successful,
            "failed": failed,
            "processed": len(results),
            "parallel": req.parallel
        },
        "results": results
    }


# ---------------------------------------------------------------------
# Enhanced Batch Operations
# ---------------------------------------------------------------------
class BatchExportRequest(BaseModel):
    """Request for batch export (multiple datasets)."""
    queries: List[QueryRequest]
    format: str = "csv"  # csv, json
    filename: Optional[str] = None


@app.post("/batch/export")
def batch_export(req: BatchExportRequest, request: Request):
    """
    Export multiple queries as a single ZIP file.
    
    Each query result is exported as a separate file within the ZIP.
    Useful for downloading multiple datasets at once.
    
    Args:
        queries: List of QueryRequest objects to export (max 20)
        format: Export format - "csv" or "json"
        filename: Optional custom filename (default: "climategpt_export_YYYYMMDD_HHMMSS.zip")
    
    Returns:
        ZIP file with exported datasets
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/batch/export")
    
    if not req.queries:
        return _error_response(
            "invalid_request",
            "No queries provided",
            "Provide at least one query in the queries array",
            {},
            ["Include queries array with at least one QueryRequest"]
        )
    
    if len(req.queries) > 20:  # Lower limit for exports
        return _error_response(
            "batch_too_large",
            f"Export batch size {len(req.queries)} exceeds maximum of 20",
            "Split into smaller export batches",
            {"batch_size": len(req.queries), "max_batch_size": 20},
            ["Execute exports in batches of 20 or fewer"]
        )
    
    if req.format not in ("csv", "json"):
        return _error_response(
            "invalid_format",
            f"Format '{req.format}' not supported",
            "Use 'csv' or 'json'",
            {"requested_format": req.format, "supported_formats": ["csv", "json"]},
            ["Supported formats: csv, json"]
        )
    
    import zipfile
    import io
    
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, query_req in enumerate(req.queries):
                try:
                    result = query(query_req, request)
                    
                    if isinstance(result, dict) and "error" in result:
                        # Create error file
                        error_content = json.dumps(result, indent=2)
                        zip_file.writestr(
                            f"query_{i}_error.txt",
                            error_content.encode("utf-8")
                        )
                    else:
                        # Export successful query
                        rows = result.get("rows", [])
                        file_id = query_req.file_id.replace("-", "_").replace("/", "_")
                        
                        if req.format == "csv":
                            # Convert to CSV
                            df = pd.DataFrame(rows)
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            zip_file.writestr(
                                f"{file_id}_query_{i}.csv",
                                csv_buffer.getvalue().encode("utf-8")
                            )
                        else:  # json
                            # Export as JSON
                            json_content = json.dumps(rows, indent=2)
                            zip_file.writestr(
                                f"{file_id}_query_{i}.json",
                                json_content.encode("utf-8")
                            )
                except Exception as e:
                    _logger.error(
                        f"Batch export query {i} failed: {e}",
                        extra={"request_id": request_id, "query_index": i},
                        exc_info=True
                    )
                    error_content = json.dumps({
                        "error": "export_failed",
                        "detail": str(e),
                        "query_index": i
                    }, indent=2)
                    zip_file.writestr(
                        f"query_{i}_error.txt",
                        error_content.encode("utf-8")
                    )
        
        # Generate filename
        if not req.filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            req.filename = f"climategpt_export_{timestamp}.zip"
        elif not req.filename.endswith(".zip"):
            req.filename += ".zip"
        
        zip_buffer.seek(0)
        
        # Trigger webhook event
        _trigger_webhook_event("batch_export_complete", {
            "batch_id": request_id,
            "total_queries": len(req.queries),
            "format": req.format,
            "filename": req.filename
        })
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={req.filename}",
                "X-Batch-Id": request_id
            }
        )
    except Exception as e:
        _logger.error(
            f"Batch export failed: {e}",
            extra={"request_id": request_id},
            exc_info=True
        )
        return _error_response(
            "export_failed",
            f"Failed to create export: {str(e)}",
            "Try exporting fewer queries or check server logs",
            {"error": str(e)},
            ["Reduce batch size", "Check query validity"]
        )


class BatchValidationRequest(BaseModel):
    """Request for batch query validation."""
    queries: List[QueryRequest]


@app.post("/batch/validate")
def batch_validate(req: BatchValidationRequest, request: Request):
    """
    Validate multiple queries without executing them.
    
    Returns validation results for each query, including:
    - Query validity
    - Estimated row counts (if possible)
    - Warnings and suggestions
    
    Args:
        queries: List of QueryRequest objects to validate (max 50)
    
    Returns:
        Dict with validation results for each query
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/batch/validate")
    
    if not req.queries:
        return _error_response(
            "invalid_request",
            "No queries provided",
            "Provide at least one query in the queries array",
            {},
            ["Include queries array with at least one QueryRequest"]
        )
    
    if len(req.queries) > 50:
        return _error_response(
            "batch_too_large",
            f"Batch size {len(req.queries)} exceeds maximum of 50",
            "Split into smaller batches",
            {"batch_size": len(req.queries), "max_batch_size": 50},
            ["Validate queries in batches of 50 or fewer"]
        )
    
    validation_results = []
    
    for i, query_req in enumerate(req.queries):
        try:
            # Validate file_id
            fid = _resolve_file_id(query_req.file_id)
            file_meta = _get_file_meta(fid)
            
            if not file_meta:
                validation_results.append({
                    "index": i,
                    "valid": False,
                    "error": "file_not_found",
                    "detail": f"File '{fid}' not found in manifest"
                })
                continue
            
            # Validate query intent
            is_valid_intent, warning_msg, suggestions_dict, suggestions_list = _validate_query_intent(query_req, file_meta)
            
            # Basic validation
            validation_results.append({
                "index": i,
                "valid": True,
                "file_id": fid,
                "warnings": warning_msg if not is_valid_intent else None,
                "suggestions": suggestions_dict if suggestions_dict else None,
                "hints": suggestions_list if suggestions_list else []
            })
        except Exception as e:
            _logger.error(
                f"Batch validation query {i} failed: {e}",
                extra={"request_id": request_id, "query_index": i},
                exc_info=True
            )
            validation_results.append({
                "index": i,
                "valid": False,
                "error": "validation_failed",
                "detail": str(e)
            })
    
    return {
        "summary": {
            "total": len(req.queries),
            "valid": sum(1 for r in validation_results if r.get("valid", False)),
            "invalid": sum(1 for r in validation_results if not r.get("valid", False))
        },
        "results": validation_results
    }


# ---------------------------------------------------------------------
# Webhook Endpoints
# ---------------------------------------------------------------------
@app.post("/webhooks")
def register_webhook(req: WebhookRegistrationRequest, request: Request):
    """
    Register a new webhook.
    
    Webhooks receive notifications when specific events occur:
    - query_complete: When a query completes successfully
    - batch_complete: When a batch operation completes
    - batch_export_complete: When a batch export completes
    - error: When an error occurs
    
    Args:
        url: Webhook URL (must be http or https)
        events: List of events to subscribe to
        secret: Optional secret for signature verification
        active: Whether webhook is active (default: true)
        description: Optional description
    
    Returns:
        Dict with webhook_id and webhook details
    """
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/webhooks")
    
    # Validate URL
    valid_url, url_error = _validate_webhook_url(req.url)
    if not valid_url:
        return _error_response(
            "invalid_webhook_url",
            url_error or "Invalid webhook URL",
            "Provide a valid http or https URL",
            {"url": req.url},
            ["Use a valid URL format", "URL must use http or https protocol"]
        )
    
    # Validate events
    valid_events = {"query_complete", "batch_complete", "batch_export_complete", "error"}
    invalid_events = [e for e in req.events if e not in valid_events]
    if invalid_events:
        return _error_response(
            "invalid_webhook_events",
            f"Invalid events: {invalid_events}",
            f"Valid events: {', '.join(sorted(valid_events))}",
            {"invalid_events": invalid_events, "valid_events": sorted(valid_events)},
            [f"Use only valid events: {', '.join(sorted(valid_events))}"]
        )
    
    # Generate webhook ID
    webhook_id = str(uuid.uuid4())
    
    # Store webhook
    with _WEBHOOK_LOCK:
        _WEBHOOKS_STORE[webhook_id] = {
            "id": webhook_id,
            "url": req.url,
            "events": req.events,
            "secret": req.secret,
            "active": req.active,
            "description": req.description,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        _WEBHOOK_DELIVERY_HISTORY[webhook_id] = []
    
    _logger.info(
        f"Webhook registered: {webhook_id}",
        extra={"request_id": request_id, "webhook_id": webhook_id, "url": req.url}
    )
    
    return {
        "webhook_id": webhook_id,
        "webhook": _WEBHOOKS_STORE[webhook_id]
    }


@app.get("/webhooks")
def list_webhooks(request: Request):
    """List all registered webhooks."""
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/webhooks")
    
    with _WEBHOOK_LOCK:
        webhooks = list(_WEBHOOKS_STORE.values())
    
    # Don't expose secrets in list
    for webhook in webhooks:
        if "secret" in webhook:
            webhook["secret"] = "***" if webhook.get("secret") else None
    
    return {
        "total": len(webhooks),
        "webhooks": webhooks
    }


@app.get("/webhooks/{webhook_id}")
def get_webhook(webhook_id: str, request: Request):
    """Get details of a specific webhook."""
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/webhooks/{webhook_id}")
    
    with _WEBHOOK_LOCK:
        if webhook_id not in _WEBHOOKS_STORE:
            return _error_response(
                "webhook_not_found",
                f"Webhook '{webhook_id}' not found",
                "Check webhook ID or register a new webhook",
                {"webhook_id": webhook_id},
                ["Use GET /webhooks to list all webhooks"]
            )
        webhook = _WEBHOOKS_STORE[webhook_id].copy()
    
    # Don't expose secret
    if "secret" in webhook:
        webhook["secret"] = "***" if webhook.get("secret") else None
    
    return webhook


@app.put("/webhooks/{webhook_id}")
def update_webhook(webhook_id: str, req: WebhookUpdateRequest, request: Request):
    """Update an existing webhook."""
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/webhooks/{webhook_id}")
    
    with _WEBHOOK_LOCK:
        if webhook_id not in _WEBHOOKS_STORE:
            return _error_response(
                "webhook_not_found",
                f"Webhook '{webhook_id}' not found",
                "Check webhook ID",
                {"webhook_id": webhook_id},
                ["Use GET /webhooks to list all webhooks"]
            )
        
        webhook = _WEBHOOKS_STORE[webhook_id]
        
        # Validate URL if provided
        if req.url is not None:
            valid_url, url_error = _validate_webhook_url(req.url)
            if not valid_url:
                return _error_response(
                    "invalid_webhook_url",
                    url_error or "Invalid webhook URL",
                    "Provide a valid http or https URL",
                    {"url": req.url},
                    ["Use a valid URL format"]
                )
            webhook["url"] = req.url
        
        # Validate events if provided
        if req.events is not None:
            valid_events = {"query_complete", "batch_complete", "batch_export_complete", "error"}
            invalid_events = [e for e in req.events if e not in valid_events]
            if invalid_events:
                return _error_response(
                    "invalid_webhook_events",
                    f"Invalid events: {invalid_events}",
                    f"Valid events: {', '.join(sorted(valid_events))}",
                    {"invalid_events": invalid_events},
                    [f"Use only valid events: {', '.join(sorted(valid_events))}"]
                )
            webhook["events"] = req.events
        
        # Update other fields
        if req.secret is not None:
            webhook["secret"] = req.secret
        if req.active is not None:
            webhook["active"] = req.active
        if req.description is not None:
            webhook["description"] = req.description
        
        webhook["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    _logger.info(
        f"Webhook updated: {webhook_id}",
        extra={"request_id": request_id, "webhook_id": webhook_id}
    )
    
    updated_webhook = webhook.copy()
    if "secret" in updated_webhook:
        updated_webhook["secret"] = "***" if updated_webhook.get("secret") else None
    
    return updated_webhook


@app.delete("/webhooks/{webhook_id}")
def delete_webhook(webhook_id: str, request: Request):
    """Delete a webhook."""
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/webhooks/{webhook_id}")
    
    with _WEBHOOK_LOCK:
        if webhook_id not in _WEBHOOKS_STORE:
            return _error_response(
                "webhook_not_found",
                f"Webhook '{webhook_id}' not found",
                "Check webhook ID",
                {"webhook_id": webhook_id},
                ["Use GET /webhooks to list all webhooks"]
            )
        
        del _WEBHOOKS_STORE[webhook_id]
        if webhook_id in _WEBHOOK_DELIVERY_HISTORY:
            del _WEBHOOK_DELIVERY_HISTORY[webhook_id]
    
    _logger.info(
        f"Webhook deleted: {webhook_id}",
        extra={"request_id": request_id, "webhook_id": webhook_id}
    )
    
    return {"message": "Webhook deleted", "webhook_id": webhook_id}


@app.get("/webhooks/{webhook_id}/history")
def get_webhook_history(webhook_id: str, request: Request, limit: int = 50):
    """Get delivery history for a webhook."""
    request_id = getattr(request.state, "request_id", "unknown")
    _record_endpoint_usage("/webhooks/{webhook_id}/history")
    
    with _WEBHOOK_LOCK:
        if webhook_id not in _WEBHOOKS_STORE:
            return _error_response(
                "webhook_not_found",
                f"Webhook '{webhook_id}' not found",
                "Check webhook ID",
                {"webhook_id": webhook_id},
                ["Use GET /webhooks to list all webhooks"]
            )
        
        history = _WEBHOOK_DELIVERY_HISTORY.get(webhook_id, [])
    
    # Return most recent deliveries
    history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
    
    return {
        "webhook_id": webhook_id,
        "total_deliveries": len(_WEBHOOK_DELIVERY_HISTORY.get(webhook_id, [])),
        "history": history
    }


@app.get("/tools")
def tools():
    return [{
        "type": "function",
        "function": {
            "name": "query",
            "description": "Filter/aggregate EDGAR emissions datasets",
            "parameters": QueryRequest.model_json_schema()
        }
    },{
        "type": "function",
        "function": {
            "name": "metrics.yoy",
            "description": "Compute YoY deltas",
            "parameters": DeltaRequest.model_json_schema()
        }
    },{
        "type": "function",
        "function": {
            "name": "metrics.rankings",
            "description": "Top-N rankings by metric",
            "parameters": RankingRequest.model_json_schema()
        }
    },{
        "type": "function",
        "function": {
            "name": "metrics.trends",
            "description": "Yearly trend aggregates",
            "parameters": TrendRequest.model_json_schema()
        }
    }]

# ---------------------------------------------------------------------
# Local runner (so: uv run python mcp_server.py works)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8010"))
    uvicorn.run("mcp_server:app", host="127.0.0.1", port=port, reload=True)