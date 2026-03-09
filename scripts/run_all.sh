#!/bin/bash
# T-501: Start all services (MCP server + Agent backend + Frontend)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source venv/bin/activate

cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $MCP_PID $AGENT_PID $FRONT_PID 2>/dev/null || true
    wait $MCP_PID $AGENT_PID $FRONT_PID 2>/dev/null || true
    echo "All services stopped."
}
trap cleanup EXIT

# 1. MCP Server (needs MODEL_PATH)
if [ -z "${MODEL_PATH:-}" ]; then
    export MODEL_PATH="$(pwd)/output/qwen3-14b-telco-Q4_K_M.gguf"
fi
echo "=== Starting MCP Server (port ${MCP_PORT:-8000}) ==="
echo "    Model: $MODEL_PATH"
cd mcp_server && python server.py &
MCP_PID=$!
cd "$PROJECT_DIR"

echo "    Waiting for model to load (~60s)..."
sleep 65
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo "ERROR: MCP server failed to start. Check MODEL_PATH."
    exit 1
fi
echo "    MCP Server ready ✓"

# 2. Agent Backend
echo "=== Starting Agent Backend (port ${API_PORT:-8080}) ==="
cd agent && python api_server.py &
AGENT_PID=$!
cd "$PROJECT_DIR"
sleep 3
echo "    Agent Backend ready ✓"

# 3. Frontend
echo "=== Starting Frontend (port 3000) ==="
cd frontend && npm run dev &
FRONT_PID=$!
cd "$PROJECT_DIR"
sleep 5

echo ""
echo "============================================"
echo "  All services running!"
echo "  MCP Server:    http://localhost:${MCP_PORT:-8000}  (PID $MCP_PID)"
echo "  Agent Backend: http://localhost:${API_PORT:-8080}  (PID $AGENT_PID)"
echo "  Frontend:      http://localhost:3000               (PID $FRONT_PID)"
echo ""
echo "  Open http://localhost:3000 in your browser"
echo "  Press Ctrl+C to stop all services"
echo "============================================"
wait
