"""T-301: Centralized configuration from environment variables."""
import os

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-4-5-haiku-latest-v1:0")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Per 1K input tokens — for cost savings estimation in comparison panel
TOKEN_PRICES = {
    "anthropic.claude-4-5-haiku": 0.001,
    "amazon.nova-2-pro": 0.0008,
    "anthropic.claude-4-5-sonnet": 0.003,
    "default": 0.001,
}
