# src/cli.py
import os
import sys
import click

REQUIRED_ENV = "ClimateGPT_Key"

@click.group()
def cli():
    """Command-line entrypoint for ClimateGPT tasks."""
    pass

@cli.command(name="viirs_etl")
def viirs_etl():
    from src.pipelines.viirs import run  # your real module
    run()

@cli.command(name="mcp_viirs")
@click.option("--port", default=8080, show_default=True, help="Port to expose the MCP server on.")
def mcp_viirs(port: int):
    """
    Start the MCP server for ClimateGPT. Requires CLIMATEGPT_API_KEY in the environment.
    Exposes a lightweight HTTP server with /healthz and /metadata endpoints.
    """
    api_key = os.getenv(REQUIRED_ENV)
    if not api_key or api_key.strip() == "__TO_BE_FILLED__":
        click.echo(
            f"[MCP] Missing {REQUIRED_ENV} in the environment. "
            "Set it in ops/config/.env before running this command."
        )
        sys.exit(2)

    # Defer imports so container starts fast and only loads if key exists
    import uvicorn
    #from src.mcp.viirs import app

    click.echo("[MCP] Starting MCP server on 0.0.0.0:%d ..." % port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    cli()

# src/mcp/__init__.py
# src/mcp/viirs.py
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="ClimateGPT MCP - VIIRS", version="0.1.0")

@app.get("/healthz")
def healthz():
    api_present = bool(os.getenv("CLIMATEGPT_API_KEY"))
    return JSONResponse({"status": "ok", "api_key_present": api_present})

@app.get("/metadata")
def metadata():
    """
    Minimal placeholder metadata endpoint.
    In your real server, emit the MCP metadata used by the LLM/tooling:
      - title, domain tags, coverage
      - schema/fields
      - semantic/keywords
    """
    meta = {
        "title": "VIIRS Thermal Hotspots & Fire Activity",
        "domain_tags": ["climate", "satellite", "viirs", "thermal_hotspots"],
        "coverage": {"spatial": "global", "temporal": "daily updates"},
        "schema": {"tables": ["hotspots"], "fields": ["lat", "lon", "acq_date", "confidence"]},
        "keywords": ["VIIRS", "hotspots", "fire", "thermal", "satellite"],
        "source": "NASA VIIRS (placeholder)",
    }
    return JSONResponse(meta)