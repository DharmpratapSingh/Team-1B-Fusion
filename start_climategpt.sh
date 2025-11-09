#!/bin/bash
# ClimateGPT Startup Script
echo "🚀 Starting ClimateGPT Application..."

# Clear cache
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Start MCP HTTP Bridge (wraps mcp_server_stdio.py)
echo "📡 Starting MCP Bridge Server..."
python mcp_http_bridge.py &
MCP_PID=$!

# Wait for MCP Server to start
sleep 3

# Check if MCP Bridge is running
if curl -s http://localhost:8010/health > /dev/null; then
    echo "✅ MCP Bridge started successfully (using TRUE MCP protocol)"
else
    echo "❌ MCP Bridge failed to start"
    exit 1
fi

# Start Streamlit App
echo "🌐 Starting Streamlit App..."
streamlit run enhanced_climategpt_with_personas.py

# Cleanup on exit
trap "kill $MCP_PID 2>/dev/null" EXIT
