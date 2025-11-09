#!/usr/bin/env python3
"""
HTTP-to-MCP Bridge Server

This bridge allows HTTP clients (like the Streamlit UI) to communicate
with the TRUE MCP protocol server (mcp_server_stdio.py) which uses stdio.

Architecture:
  HTTP Client (Streamlit) → FastAPI Bridge → MCP Server (stdio) → DuckDB
"""

import json
import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ClimateGPT MCP Bridge",
    description="HTTP-to-MCP protocol bridge for ClimateGPT",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP Server Process
mcp_process: Optional[asyncio.subprocess.Process] = None
request_counter = 0


# ============================================================================
# MCP Protocol Communication
# ============================================================================

async def send_mcp_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a JSON-RPC 2.0 request to MCP server via stdio"""
    global request_counter, mcp_process

    if not mcp_process:
        raise HTTPException(status_code=503, detail="MCP server not initialized")

    request_counter += 1
    request_id = f"http-bridge-{request_counter}"

    # Build JSON-RPC 2.0 request
    jsonrpc_request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params
    }

    # Send request to MCP server stdin
    request_json = json.dumps(jsonrpc_request) + "\n"
    logger.info(f"→ MCP Request: {method} (id={request_id})")
    logger.debug(f"Request payload: {request_json.strip()}")

    try:
        mcp_process.stdin.write(request_json.encode())
        await mcp_process.stdin.drain()
    except Exception as e:
        logger.error(f"Failed to send request to MCP server: {e}")
        raise HTTPException(status_code=503, detail=f"MCP communication error: {e}")

    # Read response from MCP server stdout
    try:
        response_line = await asyncio.wait_for(
            mcp_process.stdout.readline(),
            timeout=30.0
        )
        response_json = response_line.decode().strip()

        if not response_json:
            raise HTTPException(status_code=503, detail="Empty response from MCP server")

        logger.debug(f"← MCP Response: {response_json}")
        response = json.loads(response_json)

        # Check for JSON-RPC error
        if "error" in response:
            error = response["error"]
            logger.error(f"MCP server error: {error}")
            raise HTTPException(
                status_code=400,
                detail=error.get("message", "MCP server error")
            )

        # Return the result
        return response.get("result", {})

    except asyncio.TimeoutError:
        logger.error("MCP server response timeout")
        raise HTTPException(status_code=504, detail="MCP server timeout")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from MCP server: {e}")
        raise HTTPException(status_code=502, detail="Invalid response from MCP server")


# ============================================================================
# HTTP Endpoints (REST API compatibility with existing ClimateGPT UI)
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if mcp_process and mcp_process.returncode is None:
        return {"status": "healthy", "mcp_server": "running"}
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "mcp_server": "not running"}
    )


@app.get("/list_files")
async def list_files():
    """List available datasets via MCP list_emissions_datasets tool"""
    try:
        result = await send_mcp_request("tools/call", {
            "name": "list_emissions_datasets",
            "arguments": {}
        })

        # Parse the result content
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                return json.loads(text_content)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_schema/{file_id}")
async def get_schema(file_id: str):
    """Get schema for a specific dataset via MCP get_dataset_schema tool"""
    try:
        result = await send_mcp_request("tools/call", {
            "name": "get_dataset_schema",
            "arguments": {"file_id": file_id}
        })

        # Parse the result content
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                # Extract text from content
                text_content = content[0].get("text", "{}")
                return json.loads(text_content)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryRequest(BaseModel):
    file_id: str
    select: list
    where: Optional[dict] = None
    group_by: Optional[list] = None
    order_by: Optional[str] = None
    limit: Optional[int] = None


@app.post("/query")
async def query_data(request: QueryRequest):
    """Query dataset via MCP query_emissions tool"""
    try:
        result = await send_mcp_request("tools/call", {
            "name": "query_emissions",
            "arguments": request.dict()
        })

        # Parse the result content
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                return json.loads(text_content)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class YoYMetricsRequest(BaseModel):
    file_id: str
    key_col: str
    value_col: str
    base_year: int
    compare_year: int
    top_n: Optional[int] = 10
    direction: Optional[str] = "rise"


@app.post("/metrics/yoy")
async def yoy_metrics(request: YoYMetricsRequest):
    """Year-over-year metrics via MCP calculate_yoy_change tool"""
    try:
        result = await send_mcp_request("tools/call", {
            "name": "calculate_yoy_change",
            "arguments": request.dict()
        })

        # Parse the result content
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "{}")
                return json.loads(text_content)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating YoY metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchQueryRequest(BaseModel):
    queries: list


@app.post("/batch/query")
async def batch_query(request: BatchQueryRequest):
    """Batch query - executes multiple queries via MCP and aggregates results"""
    try:
        results = []

        # Execute each query separately via MCP
        for query in request.queries:
            try:
                result = await send_mcp_request("tools/call", {
                    "name": "query_emissions",
                    "arguments": query
                })

                # Parse the result content
                data = {}
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, list) and len(content) > 0:
                        text_content = content[0].get("text", "{}")
                        data = json.loads(text_content)

                results.append({
                    "status": "success",
                    "data": data
                })

            except Exception as e:
                logger.error(f"Batch query item failed: {e}")
                results.append({
                    "status": "error",
                    "error": str(e)
                })

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MCP Server Lifecycle Management
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Start the MCP server process on bridge startup"""
    global mcp_process

    logger.info("🚀 Starting MCP Bridge Server...")

    # Find the mcp_server_stdio.py script
    mcp_script = Path(__file__).parent / "mcp_server_stdio.py"

    if not mcp_script.exists():
        logger.error(f"MCP server script not found: {mcp_script}")
        raise RuntimeError("MCP server script not found")

    # Start MCP server as subprocess
    try:
        logger.info(f"📡 Starting MCP server: {mcp_script}")
        mcp_process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(mcp_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        # Wait a bit for initialization
        await asyncio.sleep(2)

        # Check if process is still running
        if mcp_process.returncode is not None:
            stderr = await mcp_process.stderr.read()
            logger.error(f"MCP server failed to start: {stderr.decode()}")
            raise RuntimeError("MCP server failed to start")

        logger.info("✅ MCP server started successfully")

        # Initialize MCP protocol
        await send_mcp_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "climategpt-http-bridge",
                "version": "1.0.0"
            }
        })

        logger.info("✅ MCP protocol initialized")

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the MCP server process on bridge shutdown"""
    global mcp_process

    if mcp_process:
        logger.info("🛑 Stopping MCP server...")
        try:
            mcp_process.terminate()
            await asyncio.wait_for(mcp_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("MCP server did not terminate gracefully, killing...")
            mcp_process.kill()
            await mcp_process.wait()

        logger.info("✅ MCP server stopped")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8010"))

    logger.info(f"🌍 Starting ClimateGPT MCP Bridge on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
