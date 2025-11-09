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
import re
import time
import hashlib
import uuid
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Set
from datetime import datetime, timedelta
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
from queue import Queue, Empty, Full
from contextlib import contextmanager

# ---------------------------------------------------------------------
# Logging Infrastructure (from mcp_server.py)
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
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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

logger = _setup_logging()

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

# ---------------------------------------------------------------------
# Security and Input Validation (from mcp_server.py)
# ---------------------------------------------------------------------
# Project root for path resolution
_PROJECT_ROOT = Path(__file__).parent

# Valid characters for identifiers (file_id, column names)
_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
_MAX_QUERY_COMPLEXITY = {
    "max_columns": 50,
    "max_filters": 20,
    "max_list_items": 100,
    "max_string_length": 500,
    "max_query_size": 10000,  # bytes
}

# Config defaults (env-driven)
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
# Enhanced Validation Functions (from mcp_server.py)
# ---------------------------------------------------------------------
def _validate_file_id_enhanced(file_id: str) -> Tuple[bool, Optional[str]]:
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


def _validate_column_name_enhanced(col: str) -> Tuple[bool, Optional[str]]:
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


# ---------------------------------------------------------------------
# Error handling helpers (from mcp_server.py)
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


def _error_response(code: str, detail: str, hint: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None,
                   suggestions: Optional[List[str]] = None) -> Dict[str, Any]:
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

    return response


# ---------------------------------------------------------------------
# Query Validation and Intent Detection (from mcp_server.py)
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


def _validate_query_complexity(
    select: List[str],
    where: Dict[str, Any],
    group_by: List[str],
    order_by: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Validate query complexity to prevent DoS."""
    issues = []

    # Check column count
    if len(select) > _MAX_QUERY_COMPLEXITY["max_columns"]:
        issues.append(f"Too many columns in select (max {_MAX_QUERY_COMPLEXITY['max_columns']})")

    # Check filter count
    if len(where) > _MAX_QUERY_COMPLEXITY["max_filters"]:
        issues.append(f"Too many filters (max {_MAX_QUERY_COMPLEXITY['max_filters']})")

    # Check group_by count
    if len(group_by) > _MAX_QUERY_COMPLEXITY["max_columns"]:
        issues.append(f"Too many group_by columns (max {_MAX_QUERY_COMPLEXITY['max_columns']})")

    # Validate all column names
    all_columns = set(select) | set(group_by)
    if order_by:
        order_col = order_by.split()[0]
        all_columns.add(order_col)

    for col in all_columns:
        valid, error = _validate_column_name_enhanced(col)
        if not valid:
            issues.append(f"Invalid column name '{col}': {error}")

    # Validate filter values
    for key, value in where.items():
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
# Phase 3: Advanced Query Features (from mcp_server.py)
# ---------------------------------------------------------------------

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
        # Validate column name (basic validation - schema check happens later)
        valid_col, col_error = _validate_column_name_enhanced(col)
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
        valid, error = _validate_column_name_enhanced(key)
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


# Note: Computed columns are complex and require pandas
# They may not be needed for the MCP stdio server initially
# Keeping stub for future implementation
def _validate_computed_expression(expression: str, available_columns: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate computed column expression for security.
    Placeholder for future implementation.
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

    return True, None


# ---------------------------------------------------------------------
# Phase 5: Suggestions & Intelligence (from mcp_server.py)
# ---------------------------------------------------------------------

def _get_distinct_values(file_meta: Dict[str, Any], column: str, limit: int = 100) -> List[str]:
    """
    Get distinct values for a column from a table.
    Returns list of distinct values (sorted, limited).
    """
    if not file_meta:
        return []

    # Validate column name (enhanced validation)
    valid, error = _validate_column_name_enhanced(column)
    if not valid:
        return []

    try:
        if file_meta.get("engine") == "duckdb":
            uri = file_meta.get("path")
            db_path, _, table = uri[len("duckdb://"):].partition("#")
            if not table:
                return []

            # Validate table name
            valid_table, _ = _validate_column_name_enhanced(table)
            if not valid_table:
                return []

            # Security: validate column again for SQL
            sql = f'SELECT DISTINCT "{column}" FROM {table} WHERE "{column}" IS NOT NULL ORDER BY "{column}" LIMIT {limit}'
            with _get_db_connection() as conn:
                result = conn.execute(sql).fetchall()
                if result:
                    values = [str(row[0]) for row in result]
                    return sorted(values)[:limit]
    except Exception as e:
        logger.warning(f"Error getting distinct values for {column}: {e}")
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


def _get_suggestions_for_column(file_meta: Dict[str, Any], column: str,
                                query: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
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


@lru_cache(maxsize=1)
def _coverage_index() -> Dict[str, List[str]]:
    """Build coverage index for all datasets (cached)"""
    idx = {"city": set(), "admin1": set(), "country": set()}

    for file_meta in MANIFEST.get("files", []):
        try:
            if file_meta.get("engine") == "duckdb":
                uri = file_meta.get("path")
                db_path, _, table = uri[len("duckdb://"):].partition("#")
                if not table:
                    continue

                with _get_db_connection() as conn:
                    # Check which columns exist
                    table_info_sql = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
                    columns_result = conn.execute(table_info_sql).fetchall()
                    cols = {row[0] for row in columns_result}

                    if "city_name" in cols:
                        result = conn.execute(f'SELECT DISTINCT city_name FROM {table} WHERE city_name IS NOT NULL').fetchall()
                        idx["city"].update(str(row[0]) for row in result)
                    if "admin1_name" in cols:
                        result = conn.execute(f'SELECT DISTINCT admin1_name FROM {table} WHERE admin1_name IS NOT NULL').fetchall()
                        idx["admin1"].update(str(row[0]) for row in result)
                    if "country_name" in cols:
                        result = conn.execute(f'SELECT DISTINCT country_name FROM {table} WHERE country_name IS NOT NULL').fetchall()
                        idx["country"].update(str(row[0]) for row in result)
        except Exception as e:
            logger.warning(f"Error building coverage index for {file_meta.get('file_id', 'unknown')}: {e}")
            continue

    return {k: sorted(v) for k, v in idx.items()}


def _top_matches(name: str, pool: List[str], k: int = 5) -> List[str]:
    """Find top matching strings from a pool"""
    nm = (name or "").lower()
    scored = []
    for p in pool:
        pl = p.lower()
        score = 0 if pl == nm else (1 if nm in pl else 2)
        scored.append((p, score, len(p)))
    scored.sort(key=lambda x: (x[1], x[2]))
    return [p for p, _, _ in scored[:k]]


# ---------------------------------------------------------------------
# Phase 2 & 4: Remaining Validation and Data Handling
# ---------------------------------------------------------------------

def _get_file_meta(file_id: str) -> Optional[Dict[str, Any]]:
    """Get file metadata from manifest"""
    return next((f for f in MANIFEST.get("files", []) if f.get("file_id") == file_id), None)


def _validate_query_intent(
    file_id: str,
    where: Dict[str, Any],
    select: List[str],
    file_meta: Optional[Dict[str, Any]] = None,
    assist: bool = True
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[List[str]]]:
    """
    Validate query and detect potential issues before execution.
    Returns: (is_valid, warning_message, suggestions_dict, suggestions_list)
    """
    warnings = []
    suggestions_dict: Dict[str, Any] = {}
    suggestions_list: List[str] = []

    if not file_meta:
        return False, "File metadata not found", None, ["Check available datasets"]

    # Check temporal coverage
    if "year" in where:
        year_val = where["year"]
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
    if "city" in file_id and "country_name" in where:
        country = where.get("country_name")
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
    if not where and assist:
        warnings.append("No filters specified - returning sample data")
        suggestions_list.append("Add filters like 'year' or 'country_name' to narrow results")

    # Check if select columns exist in manifest
    if file_meta.get("columns") and select:
        manifest_cols = {col.get("name") for col in file_meta.get("columns", []) if isinstance(col, dict)}
        missing_cols = [c for c in select if c not in manifest_cols]
        if missing_cols:
            warnings.append(f"Some requested columns may not exist: {missing_cols}")
            suggestions_list.append(f"Available columns: {', '.join(sorted(manifest_cols)[:10])}...")

    warning_msg = "; ".join(warnings) if warnings else None
    return True, warning_msg, suggestions_dict if suggestions_dict else None, suggestions_list if suggestions_list else None


def _detect_query_patterns(
    where: Dict[str, Any],
    group_by: List[str],
    order_by: Optional[str],
    limit: Optional[int]
) -> Dict[str, Any]:
    """Detect query patterns to provide better suggestions."""
    patterns = {
        "is_top_n": False,
        "is_comparison": False,
        "is_trend": False,
        "has_temporal_filter": "year" in where or "month" in where,
        "has_spatial_filter": any(k in where for k in ["country_name", "admin1_name", "city_name"]),
        "needs_aggregation": bool(group_by),
    }

    # Detect top N pattern
    if order_by and "DESC" in order_by.upper():
        if limit and limit <= 20:
            patterns["is_top_n"] = True

    # Detect comparison pattern
    if "year" in where and isinstance(where["year"], dict) and "in" in where["year"]:
        if len(where["year"]["in"]) == 2:
            patterns["is_comparison"] = True

    # Detect trend pattern
    if "year" in where or (group_by and "year" in group_by):
        patterns["is_trend"] = True

    return patterns


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
    """Validate file_id format - calls enhanced version"""
    return _validate_file_id_enhanced(file_id)

def _find_file_meta(file_id: str):
    """Find file metadata in manifest - calls _get_file_meta"""
    return _get_file_meta(file_id)

def _get_table_name(file_meta: dict) -> Optional[str]:
    """Extract table name from file metadata"""
    path = file_meta.get("path", "")
    if path.startswith("duckdb://"):
        return path.split("#")[1] if "#" in path else None
    return None

def _build_where_sql(where: dict[str, Any]) -> tuple[str, list]:
    """
    Build WHERE clause SQL with parameterized queries (enhanced version).
    Supports: equality, in, between, comparisons, contains/ILIKE
    """
    if not where:
        return "", []

    conditions = []
    params = []

    for key, value in where.items():
        if isinstance(value, list):
            # List values are treated as IN operator
            placeholders = ",".join(["?"] * len(value))
            conditions.append(f"{key} IN ({placeholders})")
            params.extend(value)
        elif isinstance(value, dict):
            # Support various operators
            if "in" in value and isinstance(value["in"], list):
                placeholders = ",".join(["?"] * len(value["in"]))
                conditions.append(f"{key} IN ({placeholders})")
                params.extend(value["in"])
            elif "between" in value and isinstance(value["between"], (list, tuple)) and len(value["between"]) == 2:
                conditions.append(f"{key} BETWEEN ? AND ?")
                params.extend(list(value["between"]))
            elif "gte" in value:
                conditions.append(f"{key} >= ?")
                params.append(value["gte"])
            elif "lte" in value:
                conditions.append(f"{key} <= ?")
                params.append(value["lte"])
            elif "gt" in value or "$gt" in value:
                val = value.get("gt", value.get("$gt"))
                conditions.append(f"{key} > ?")
                params.append(val)
            elif "lt" in value or "$lt" in value:
                val = value.get("lt", value.get("$lt"))
                conditions.append(f"{key} < ?")
                params.append(val)
            elif "ne" in value or "$ne" in value:
                val = value.get("ne", value.get("$ne"))
                conditions.append(f"{key} != ?")
                params.append(val)
            elif "contains" in value:
                # Case-insensitive substring search
                conditions.append(f"CAST({key} AS VARCHAR) ILIKE ?")
                params.append(f"%{value['contains']}%")
        else:
            # Simple equality
            conditions.append(f"{key} = ?")
            params.append(value)

    sql = " WHERE " + " AND ".join(conditions) if conditions else ""
    return sql, params

def _validate_column_name(column: str, file_meta: dict) -> tuple[bool, Optional[str]]:
    """Validate column name exists in dataset schema (prevents SQL injection)"""
    # First, do security validation
    valid, error = _validate_column_name_enhanced(column)
    if not valid:
        return False, error

    # Then check against schema
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









