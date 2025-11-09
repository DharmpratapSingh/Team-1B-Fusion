#!/bin/bash
# Run MCP server with mcp dev (requires venv1 activated)
# Usage: ./run_mcp_dev.sh

cd "$(dirname "$0")"

# Check if venv1 exists
if [ ! -d ".venv1" ]; then
    echo "❌ Error: .venv1 not found"
    echo "Please create venv1 first: python3.13 -m venv .venv1"
    exit 1
fi

# Activate venv1
source .venv1/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "✅ Using: $PYTHON_VERSION"
echo "✅ Python path: $(which python)"
echo "✅ MCP path: $(which mcp)"

# Check if mcp is installed
if ! command -v mcp &> /dev/null; then
    echo "❌ Error: mcp command not found"
    echo "Installing mcp[cli]..."
    pip install "mcp[cli]"
fi

# Note about mcp dev limitation
echo ""
echo "⚠️  Note: mcp dev only supports FastMCP servers, not low-level Server class"
echo "   Your server uses low-level Server, so mcp dev will show an error."
echo "   Use 'python mcp_server_stdio.py' directly instead."
echo ""

# Try to run (will show error but that's expected)
echo "Attempting mcp dev (will show FastMCP error - this is expected)..."
mcp dev mcp_server_stdio.py









