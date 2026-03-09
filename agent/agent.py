"""T-302: Strands agent with Bedrock frontier model + MCP tool binding."""
import json
import time
import logging
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from mcp.client.sse import sse_client
from config import MODEL_ID, MCP_URL, AWS_REGION, TOKEN_PRICES

logger = logging.getLogger("TelcoAgent")

SYSTEM_PROMPT = (
    "You are a Telco Network Operations AI Assistant specializing in Root Cause Analysis "
    "for 5G networks. When a user reports a network issue:\n"
    "1. Extract the subscriber identifier (IMSI) and symptom description from the query.\n"
    "2. Use the analyze_network_telemetry tool to retrieve filtered diagnostic data.\n"
    "3. Interpret the diagnostic JSON and provide a clear, actionable diagnosis.\n"
    "4. Include a specific recommendation (e.g., dispatch field tech, escalate to core team).\n"
    "Always explain your reasoning in plain language suitable for a NOC operator."
)


class TelcoRCAAgent:
    def __init__(self):
        self.model = BedrockModel(model_id=MODEL_ID, region_name=AWS_REGION)
        self.mcp_client = MCPClient(lambda: sse_client(url=MCP_URL))
        self.mcp_client.start()
        tools = self.mcp_client.list_tools()
        logger.info(f"MCP tools discovered: {[t.name for t in tools]}")

        self.agent = Agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
        )
        model_prefix = MODEL_ID.split(":")[0] if ":" in MODEL_ID else MODEL_ID
        self.price_per_1k = TOKEN_PRICES.get(model_prefix, TOKEN_PRICES["default"])
        logger.info(f"Agent ready: model={MODEL_ID}")

    def process_query(self, message: str) -> dict:
        start = time.time()
        response = self.agent(message)
        total_time = round(time.time() - start, 2)
        metrics = self._extract_metrics(response, total_time)
        return {"response": str(response), "metrics": metrics}

    def _extract_metrics(self, response, total_time: float) -> dict | None:
        try:
            for result in getattr(response, "tool_results", []):
                content = result.content if hasattr(result, "content") else str(result)
                data = json.loads(content)
                if "metadata" in data:
                    m = data["metadata"]
                    raw_tok = m["raw_log_bytes"] // 4
                    filt_tok = m["filtered_output_bytes"] // 4
                    return {
                        "raw_log_bytes": m["raw_log_bytes"],
                        "raw_log_tokens_est": raw_tok,
                        "filtered_output_bytes": m["filtered_output_bytes"],
                        "filtered_output_tokens_est": filt_tok,
                        "token_reduction_pct": round((1 - filt_tok / max(raw_tok, 1)) * 100, 1),
                        "estimated_cost_savings_usd": round((raw_tok - filt_tok) / 1000 * self.price_per_1k, 4),
                        "slm_inference_latency_sec": m["inference_latency_sec"],
                        "total_response_time_sec": total_time,
                    }
        except Exception as e:
            logger.debug(f"Metrics extraction failed: {e}")
        return None

    def shutdown(self):
        self.mcp_client.stop()
