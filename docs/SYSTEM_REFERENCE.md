# ClimateGPT - Quick Reference Guide

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ClimateGPT System Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Interfaces:                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Streamlit UI    │  │  Claude Desktop  │  │ CLI Harness  │  │
│  │  Port 8501       │  │  Stdio MCP       │  │ run_llm.py   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
│           │                      │                   │          │
│  APIs:    │                      │                   │          │
│  ┌────────▼──────────────────────▼───────────────────▼───────┐  │
│  │      MCP FastAPI Server (Port 8010)                       │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │  • /health        • /list_files                    │   │  │
│  │  │  • /get_schema    • /query                         │   │  │
│  │  │  • /metrics/yoy   • /suggestions                   │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│  Core:    │                                                     │
│  ┌────────▼─────────────────────────────────────────────────┐  │
│  │  Utilities Layer (src/utils/)                           │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ router.py      → intent to file_id mapping      │   │  │
│  │  │ intent.py      → extract sector/place/year/...  │   │  │
│  │  │ answer.py      → format results, convert units  │   │  │
│  │  │ fallbacks.py   → degrade city→admin1→country    │   │  │
│  │  │ http.py        → retry logic + backoff          │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └────────┬─────────────────────────────────────────────────┘  │
│           │                                                     │
│  Data:    │                                                     │
│  ┌────────▼─────────────────────────────────────────────────┐  │
│  │  DuckDB Connection Pool (data/warehouse/)               │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ 48 datasets: 8 sectors × 3 levels × 2 grains   │   │  │
│  │  │ Coverage: 2000-2024, countries/admin1/cities    │   │  │
│  │  │ Format: EDGAR v2024 emissions data              │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Natural Language Query
  │
  ├─→ [intent.py] Extract intent
  │    • Sector (transport, power, waste, ...)
  │    • Place (country, admin1, city)
  │    • Year range
  │    • Temporal grain (month vs year)
  │
  ├─→ [router.py] Route to dataset
  │    transport-country-year
  │
  ├─→ [mcp_server.py] Security validation
  │    • File ID check
  │    • Column validation
  │    • Filter validation
  │
  ├─→ [DuckDB] Query execution
  │    • WHERE clauses
  │    • GROUP BY aggregations
  │    • ORDER BY sorting
  │
  ├─→ [fallbacks.py] Handle no results
  │    City → Admin1 → Country
  │
  ├─→ [answer.py] Format response
  │    • Convert to MtCO2
  │    • Generate summary
  │    • Add metadata
  │
  └─→ Response to User
```

## Testing Pyramid

```
                    /\
                   /  \
                  /E2E \
                 / (8)  \
                /________\
               /          \
              /Integration \
             /   (30)       \
            /________________\
           /                  \
          /     Unit Tests     \
         /         (45)         \
        /________________________\

Horizontal: Security (19) + Performance (5)
Total: 107 tests
```

## Test Categories

| Test Type | Count | Run Time | Requires |
|-----------|-------|----------|----------|
| **Unit** | 45 | <5m | Nothing |
| **Integration** | 30 | 2-5m | Server running |
| **E2E** | 8 | 5-15m | Full stack |
| **Security** | 19 | <5m | Mixed |
| **Performance** | 5 | 5-10m | Load tools |
| **TOTAL** | **107** | **20-40m** | - |

## Quick Commands

```bash
# Setup
uv sync --dev

# All tests
uv run pytest tests/ -v

# By category
uv run pytest tests/ -m unit -v              # Unit only
uv run pytest tests/ -m integration -v       # Integration only
uv run pytest tests/ -m e2e -v              # E2E only
uv run pytest tests/ -m security -v         # Security only

# With coverage
uv run pytest tests/ --cov=src --cov-report=html

# Start server for integration tests
make serve &
sleep 3
uv run pytest tests/ -m integration -v

# Run in parallel
uv run pytest tests/ -n 4 -v

# Stop server
pkill -f "uvicorn mcp_server"
```

## Environment Setup

```bash
# Copy .env.example
cp .env.example .env

# Edit .env with:
MCP_MANIFEST_PATH=data/curated-2/manifest_mcp_duckdb.json
PORT=8010
OPENAI_API_KEY=sk-...
MODEL=gpt-4
```

## File Structure

```
ClimateGPT/
├── mcp_server.py                    # FastAPI REST API
├── mcp_server_stdio.py              # MCP protocol server
├── enhanced_climategpt_with_personas.py  # Streamlit UI
├── run_llm.py                       # CLI LLM harness
│
├── src/
│   ├── __init__.py
│   ├── cli.py                       # CLI commands
│   ├── utils/
│   │   ├── router.py                # Intent → file_id
│   │   ├── intent.py                # Extract intent
│   │   ├── answer.py                # Format results
│   │   ├── fallbacks.py             # Query degradation
│   │   ├── http.py                  # HTTP utilities
│   │   └── logging.py               # Logging setup
│   └── pipelines/
│       └── viirs.py                 # VIIRS ETL (placeholder)
│
├── tests/
│   ├── test_api.py                  # API tests
│   ├── test_json_cleaner.py         # JSON parsing
│   ├── conftest.py                  # (TO CREATE)
│   └── utils.py                     # (TO CREATE)
│
├── data/
│   ├── curated-2/
│   │   └── manifest_mcp_duckdb.json # Dataset metadata
│   └── warehouse/
│       └── climategpt.duckdb        # DuckDB database
│
├── docs/
│   ├── README_MCP.md                # MCP details
│   ├── TESTING_GUIDE.md             # Manual testing
│   ├── QUICK_START.md               # Setup guide
│   └── CI_CD_SETUP_GUIDE.md         # CI/CD guide
│
├── .env.example                     # Environment template
├── pytest.ini                       # Pytest config
├── pyproject.toml                   # Dependencies
├── Makefile                         # Build commands
└── ARCHITECTURE_AND_TESTING_PLAN.md # Detailed plan (1,166 lines)
```

## Key Concepts

### Intent Parsing
Converts natural language to structured data:
```python
input:  "How much power did Germany emit monthly in 2023?"
output: {
  "sector": "power",
  "place": "Germany",
  "grain": "month",
  "year": 2023,
  "level": "country"
}
```

### Routing
Maps intent to dataset file_id:
```python
intent = {"sector": "power", "level": "country", "grain": "month"}
file_id = route_file_id(intent)  # → "power-country-month"
```

### Query Flow
```
WHERE filters → GROUP BY → aggregations → ORDER BY → LIMIT
```

### Fallback Degradation
If city-level query returns no results:
1. Try admin1-level (state/province)
2. Try country-level
3. Expand WHERE clause (fuzzy matching)

### Answer Generation
```python
Emissions Data → Convert to MtCO2 → Format Summary → Add Metadata
```

## Dataset Reference

### Available Sectors (8)
- transport
- power
- waste
- agriculture
- buildings
- fuel-exploitation
- industrial-combustion
- industrial-processes

### Available Levels (3)
- country (e.g., "United States of America")
- admin1 (e.g., "California" state/province)
- city (e.g., "New York City")

### Available Grains (2)
- year (annual data 2000-2024)
- month (monthly data with month column)

### Table Naming
```
{sector}-{level}-{grain}

Examples:
  transport-country-year
  power-admin1-month
  agriculture-city-year
```

## Security Checklist

- File ID must match: `^[a-zA-Z0-9_\-\.]+$`
- Column names validated against schema
- Filter values type-checked
- Query complexity limited (50 cols, 20 filters)
- Computed columns: no import, exec, etc.
- Rate limiting: 60 requests per 5 minutes per IP
- All input logged with request ID

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Health check | <100ms | No DB access |
| List files | <500ms | 48 datasets |
| Simple query | <1s | limit=10 |
| Aggregation | <2s | GROUP BY |
| YoY metrics | <2s | Two years |
| Large result | <30s | limit=100,000 |

## Testing Checklist

- [ ] Unit tests pass (45)
- [ ] Integration tests pass (30)
- [ ] E2E tests pass (8)
- [ ] Security tests pass (19)
- [ ] Performance tests pass (5)
- [ ] Coverage > 80%
- [ ] CI/CD green
- [ ] Documentation complete

## Common Issues & Solutions

### Issue: "Manifest not found"
```bash
Solution: Check MCP_MANIFEST_PATH in .env
ls -la data/curated-2/manifest_mcp_duckdb.json
```

### Issue: "Connection pool full"
```bash
Solution: Increase pool size in mcp_server.py
pool_size=10, max_overflow=5 (default)
```

### Issue: "No results found"
```bash
Solution: Check if data exists for that location/year
Try: curl "http://127.0.0.1:8010/suggestions/transport-country-year?column=country_name"
```

### Issue: "Rate limited (429)"
```bash
Solution: Wait 5 minutes or change MCP_RATE_CAP in .env
```

### Issue: "Test server won't start"
```bash
Solution: Kill existing process
lsof -i :8010
kill -9 <PID>
```

## Links & References

| Resource | URL |
|----------|-----|
| MCP Protocol | https://modelcontextprotocol.io/ |
| FastAPI Docs | https://fastapi.tiangolo.com/ |
| DuckDB Docs | https://duckdb.org/docs/ |
| Pytest Docs | https://docs.pytest.org/ |
| Streamlit Docs | https://docs.streamlit.io/ |
| EDGAR Database | https://edgar.jrc.ec.europa.eu/ |

## Implementation Phases

### Week 1: Foundation
- Setup fixtures and utilities
- Implement unit tests (20)
- Setup CI/CD

### Week 2: Integration
- Implement integration tests (20)
- Test all endpoints
- Test connection pool

### Week 3: Advanced
- E2E workflow tests
- Security suite
- Performance tests

### Week 4: Polish
- Coverage reporting
- Documentation
- Optimization

## Success Metrics

- **Coverage**: >80%
- **All tests pass**: 107/107
- **CI/CD green**: All checks passing
- **Performance**: <2s median query time
- **Uptime**: No crashes under 50 concurrent requests
- **Security**: Zero vulnerabilities detected

---

**Last Updated**: 2025-11-02
**Documents**: 
- TESTING_PLAN_SUMMARY.md (quick overview)
- ARCHITECTURE_AND_TESTING_PLAN.md (detailed - 1,166 lines)
- QUICK_REFERENCE.md (this file - cheat sheet)
