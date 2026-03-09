"""T-206: Smart MCP Server - FastMCP with SSE transport."""
import os
import json
import logging
from mcp.server.fastmcp import FastMCP
from inference_engine import TelcoInferenceEngine
from post_processing_filter import apply_filter
from log_store import LogStore

# Logging: human-readable to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
# JSON to file
fh = logging.FileHandler("mcp_server.log")
fh.setFormatter(logging.Formatter('{"ts":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'))
logging.getLogger().addHandler(fh)

logger = logging.getLogger("MCPServer")

MODEL_PATH = os.environ["MODEL_PATH"]
GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "telco_grammar.gbnf")
PORT = int(os.getenv("MCP_PORT", "8000"))

engine = TelcoInferenceEngine.get_instance(MODEL_PATH, GRAMMAR_PATH)
log_store = LogStore()
mcp = FastMCP("TelcoEdgeFilter")


@mcp.tool()
def analyze_network_telemetry(imsi: str, context: str = "general_failure") -> str:
    """Retrieves network signaling logs for a subscriber IMSI, filters noise
    using a local fine-tuned SLM, and returns a clean JSON summary of failure
    events for Root Cause Analysis."""
    logger.info(f"Tool called: imsi={imsi}, context={context}")
    raw_logs = log_store.get_logs(imsi)
    result = engine.analyze(raw_logs)
    filtered_events, rules = apply_filter(result["events"])
    filtered_json = json.dumps({"events": filtered_events})

    response = {
        "events": filtered_events,
        "metadata": {
            "raw_log_bytes": result["raw_log_bytes"],
            "filtered_output_bytes": len(filtered_json),
            "inference_latency_sec": result["inference_latency_sec"],
            "filter_rules_applied": rules,
        },
    }
    logger.info(f"Result: {len(filtered_events)} events, rules={rules}")
    return json.dumps(response)


if __name__ == "__main__":
    logger.info(f"Starting MCP Server on port {PORT}")
    mcp.run(transport="sse")
