# ClimateGPT MCP Architecture

## Overview

ClimateGPT now uses **TRUE MCP (Model Context Protocol)** for data access. This document explains the architecture and how the components work together.

## Architecture Diagram

```
┌──────────────────────────────────────┐
│  ClimateGPT Streamlit UI             │
│  (enhanced_climategpt_with_personas) │
│  Port: 8501                          │
└───────────────┬──────────────────────┘
                │
                │ HTTP REST API
                │ (http://localhost:8010)
                │
                ▼
┌──────────────────────────────────────┐
│  MCP HTTP Bridge                     │
│  (mcp_http_bridge.py)                │
│  Port: 8010                          │
│                                      │
│  - Exposes HTTP endpoints            │
│  - Translates to MCP protocol        │
└───────────────┬──────────────────────┘
                │
                │ MCP Protocol (JSON-RPC 2.0)
                │ via stdin/stdout
                │
                ▼
┌──────────────────────────────────────┐
│  TRUE MCP Server                     │
│  (mcp_server_stdio.py)               │
│                                      │
│  - Implements MCP protocol           │
│  - Provides MCP tools                │
└───────────────┬──────────────────────┘
                │
                │ SQL Queries
                │
                ▼
┌──────────────────────────────────────┐
│  DuckDB Databases                    │
│  (data/warehouse/*.duckdb)           │
│                                      │
│  - Transport emissions               │
│  - Power emissions                   │
│  - Other sectors                     │
└──────────────────────────────────────┘
```

## Components

### 1. ClimateGPT Streamlit UI

**File:** `enhanced_climategpt_with_personas.py`

- Web-based user interface
- Handles user questions in natural language
- Calls LLM to generate MCP tool calls
- Makes HTTP requests to MCP Bridge
- Displays results with persona-based formatting

### 2. MCP HTTP Bridge

**File:** `mcp_http_bridge.py`

- **Purpose:** Allows HTTP clients to communicate with MCP stdio server
- **Protocol Translation:** HTTP REST ↔ MCP JSON-RPC 2.0
- **Lifecycle:** Manages the MCP server subprocess

#### HTTP Endpoints → MCP Tools Mapping

| HTTP Endpoint | MCP Tool | Description |
|--------------|----------|-------------|
| `GET /health` | - | Health check |
| `GET /list_files` | `list_emissions_datasets` | List available datasets |
| `GET /get_schema/{file_id}` | `get_dataset_schema` | Get dataset schema |
| `POST /query` | `query_emissions` | Query emissions data |
| `POST /metrics/yoy` | `calculate_yoy_change` | Year-over-year analysis |
| `POST /batch/query` | `query_emissions` (multiple) | Batch queries |

### 3. TRUE MCP Server

**File:** `mcp_server_stdio.py`

- **Protocol:** MCP (Model Context Protocol) via stdio
- **Communication:** JSON-RPC 2.0 over stdin/stdout
- **Tools Provided:**

  1. **list_emissions_datasets** - List all datasets
  2. **get_dataset_schema** - Get dataset schema
  3. **query_emissions** - Query emissions data
  4. **calculate_yoy_change** - Year-over-year calculations
  5. **analyze_monthly_trends** - Monthly trend analysis
  6. **detect_seasonal_patterns** - Seasonal pattern detection
  7. **analyze_emissions** - General emissions analysis
  8. **compare_countries** - Country comparisons
  9. **analyze_covid_impact** - COVID impact analysis

### 4. DuckDB Databases

**Location:** `data/warehouse/`

- Multiple sector databases (transport, power, agriculture, etc.)
- Efficient columnar storage
- Fast analytical queries

## Why This Architecture?

### Problem

- **Streamlit UI** needs HTTP endpoints (runs in browser)
- **TRUE MCP Protocol** uses stdio (standard input/output)
- These two cannot directly communicate

### Solution

The **MCP HTTP Bridge** acts as a translator:

1. UI makes HTTP request → Bridge receives it
2. Bridge converts to MCP JSON-RPC → Sends to MCP server via stdin
3. MCP server processes → Returns result via stdout
4. Bridge converts back to HTTP response → UI receives it

### Benefits

✅ **TRUE MCP Protocol** - Proper MCP implementation
✅ **HTTP Compatibility** - Works with web UI
✅ **Separation of Concerns** - Clean architecture
✅ **MCP Client Support** - Can connect Claude Desktop, IDEs, etc.
✅ **Same Data Access** - All paths use MCP tools

## Starting the System

### Using the Startup Script (Recommended)

```bash
./start_climategpt.sh
```

This starts:
1. MCP Bridge (which starts MCP server internally)
2. Streamlit UI

### Manual Start (For Development)

Terminal 1 - Start MCP Bridge:
```bash
python mcp_http_bridge.py
```

Terminal 2 - Start Streamlit UI:
```bash
streamlit run enhanced_climategpt_with_personas.py
```

### Using Docker

```bash
docker-compose up
```

## Configuration

### Environment Variables

- `PORT` - HTTP bridge port (default: 8010)
- `MCP_URL` - HTTP bridge URL (default: http://127.0.0.1:8010)
- `MANIFEST_PATH` - Path to dataset manifest

### Files

- `.env` - Environment configuration
- `data/curated-2/manifest_mcp_duckdb.json` - Dataset manifest

## MCP Protocol Details

### JSON-RPC 2.0 Format

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "method": "tools/call",
  "params": {
    "name": "query_emissions",
    "arguments": {
      "file_id": "transport-country-year",
      "select": ["country_name", "year", "emissions_tonnes"],
      "limit": 10
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"rows\": [...], \"row_count\": 10}"
      }
    ]
  }
}
```

## External MCP Clients

The MCP server can also be used by external MCP clients:

### Claude Desktop

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "climategpt": {
      "command": "python",
      "args": ["/path/to/mcp_server_stdio.py"]
    }
  }
}
```

### IDEs (VSCode, etc.)

Configure your IDE's MCP client to run:
```bash
python mcp_server_stdio.py
```

## Troubleshooting

### Bridge not starting

Check logs for MCP server errors:
```bash
python mcp_http_bridge.py 2>&1 | tee bridge.log
```

### MCP server not responding

Test MCP server directly:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python mcp_server_stdio.py
```

### HTTP endpoints returning errors

Check bridge logs and verify MCP server tools match expected names.

## Migration from Old FastAPI Server

### What Changed

**Before:**
- `mcp_server.py` (FastAPI) → DuckDB directly

**After:**
- `mcp_http_bridge.py` (FastAPI) → `mcp_server_stdio.py` (MCP) → DuckDB

### Why Migrate?

- ✅ Use TRUE MCP protocol (industry standard)
- ✅ Compatible with Claude Desktop and MCP ecosystem
- ✅ Better separation of data access layer
- ✅ Follows MCP specification

### Backward Compatibility

The HTTP endpoints remain the same, so **no changes needed to the UI**.

## Future Enhancements

- [ ] Add authentication to HTTP bridge
- [ ] Implement MCP resource subscriptions
- [ ] Add more MCP tools for advanced analytics
- [ ] Support MCP prompts for query templates
- [ ] Add caching layer in bridge

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DuckDB Documentation](https://duckdb.org/docs/)
