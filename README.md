# ClimateGPT

AI-powered emissions data analysis system for EDGAR v2024 datasets. Query historical CO₂ emissions data through a conversational interface powered by an LLM and MCP (Model Context Protocol) server.

## Features

- **Multi-sector emissions data**: Transport, Power Industry, and other EDGAR sectors
- **Geographic granularity**: Country, admin-1 (state/province), and city-level data
- **Temporal analysis**: Monthly and annual data from 2000-2024
- **Conversational interface**: Natural language queries powered by LLM
- **MCP server**: Standardized data access via Model Context Protocol
- **Interactive UI**: Streamlit-based chat interface with persona modes

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (optional, for containerized deployment)

### Local Development

1. **Start the MCP server** (port 8010):

```bash
make serve
```

2. **In a second terminal, start the UI** (port 8501):

```bash
make ui
```

3. Open http://localhost:8501 in your browser

### Docker Deployment

```bash
docker compose up --build
```

This will start both services:
- **server**: FastAPI MCP server on port 8010
- **ui**: Streamlit interface on port 8501

## Architecture

### MCP Server (`mcp_server.py`)

FastAPI-based server providing:
- `/health` - Health check endpoint
- `/list_files` - List available datasets
- `/get_schema/{id}` - Get dataset schema
- `/query` - Execute SQL queries on emissions data
- `/metrics/yoy` - Year-over-year growth metrics
- `/metrics/rankings` - Country/region rankings
- `/metrics/trends` - Time-series trend analysis

Features:
- Rate limiting (60 requests per 5 minutes per IP)
- Gzip compression
- DuckDB backend for fast analytical queries
- Configurable via environment variables

### UI (`enhanced_climategpt_with_personas.py`)

Streamlit chat interface with:
- Multiple persona modes (Analyst, Technical, Policy Advisor)
- Chat-first layout with inline controls
- CSV export of query results
- Status indicators and error handling

## Configuration

Set environment variables as needed:

```bash
# MCP Server
export MCP_MANIFEST_PATH=data/curated-2/manifest_mcp_duckdb.json
export MCP_RATE_CAP=60  # requests per 5 minutes per IP
export PORT=8010

# LLM Configuration
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key
export MODEL=gpt-4

# Query defaults
export ASSIST_DEFAULT=smart
export PROXY_DEFAULT=spatial
export PROXY_MAX_K=10
export PROXY_RADIUS_KM=100
```

See `docker-compose.yml` for container-specific configuration.

## Testing

### Unit & Integration Tests

Run the test suite:

```bash
make test
```

Or with pytest directly:

```bash
uv run pytest -v
```

### LLM Comparative Testing

The system has been tested with multiple LLM backends. Key findings:
- **Default LLM**: 100% success rate, 5.7s average response time (recommended for production)
- **Llama Q5_K_M**: 80% success rate, 10.4s average response time (viable for development/testing)
- All tool calls and natural language summarization working correctly

See `docs/TESTING_RESULTS.md` for detailed comparison results.

For automated LLM testing tools, see the `testing/` directory which includes:
- Test harness with 50 question bank covering all sectors and query types
- Analysis and visualization scripts
- LM Studio setup guides

## Data Sources

This project uses EDGAR (Emissions Database for Global Atmospheric Research) v2024 datasets:
- CO₂ emissions by sector (transport, power industry, etc.)
- Global coverage with spatial resolution
- Monthly temporal resolution (2000-2024)

## Usage Notes

- Use exact country names (e.g., "United States of America" not "USA")
- All emissions values are in tonnes CO₂; large numbers displayed as MtCO₂
- No forecasts or per-capita metrics (by design)
- Queries are limited by rate limiting to prevent abuse

## Documentation

Additional documentation is available in the `docs/` folder:
- `docs/QUICK_START.md` - Detailed setup guide
- `docs/SYSTEM_REFERENCE.md` - System architecture and quick reference
- `docs/TESTING_GUIDE.md` - Testing procedures and methodology
- `docs/TESTING_RESULTS.md` - Testing results and LLM comparison findings
- `docs/CI_CD_SETUP_GUIDE.md` - CI/CD deployment
- `docs/README_MCP.md` - MCP protocol details

For automated testing tools and scripts, see the `testing/` directory.

## Development

### Project Structure

```
.
├── mcp_server.py                          # FastAPI MCP server
├── mcp_server_stdio.py                    # MCP stdio protocol server
├── enhanced_climategpt_with_personas.py   # Streamlit UI
├── run_llm.py                             # LLM integration harness
├── src/
│   └── utils/                             # Core utilities
│       ├── router.py                      # Intent to dataset routing
│       ├── intent.py                      # Intent extraction
│       ├── answer.py                      # Response formatting
│       ├── fallbacks.py                   # Query fallback logic
│       └── http.py                        # HTTP utilities
├── data/
│   ├── curated-2/                         # Processed datasets
│   │   └── manifest_mcp_duckdb.json       # Dataset manifest
│   ├── warehouse/                         # DuckDB databases
│   └── geo/                               # Geographic data
├── testing/                               # LLM testing infrastructure
│   ├── test_harness.py                    # Automated test runner
│   ├── analyze_results.py                 # Results analysis
│   ├── test_question_bank.json            # 50 test questions
│   └── test_results/                      # Test outputs
├── tests/                                 # Unit/integration tests
├── docs/                                  # Documentation
│   ├── QUICK_START.md                     # Setup guide
│   ├── SYSTEM_REFERENCE.md                # Architecture reference
│   ├── TESTING_GUIDE.md                   # Testing procedures
│   └── TESTING_RESULTS.md                 # LLM comparison results
├── Dockerfile.server                      # Server container
├── Dockerfile.ui                          # UI container
├── docker-compose.yml                     # Multi-container setup
├── Makefile                               # Development commands
├── pyproject.toml                         # Dependencies
└── uv.lock                                # Locked dependencies
```

### Dependencies

Managed via `pyproject.toml` with pinned versions for reproducibility:
- FastAPI + Uvicorn (API server)
- Streamlit (UI)
- DuckDB (analytical database)
- OpenAI (LLM integration)
- Pandas, NumPy (data processing)
- GeoPandas, Shapely (spatial operations)

Install all dependencies:

```bash
uv sync
```

## License

See project repository for license information.

## Contributing

Contributions welcome. Please ensure:
- All tests pass (`make test`)
- Code follows project style (ruff/black)
- Documentation is updated

## Support

For issues or questions, please refer to the documentation in the `docs/` folder or open an issue on the project repository.
