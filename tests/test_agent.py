"""T-404: Unit tests for agent configuration and API server structure."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))


class TestConfig:
    def test_config_imports(self):
        from config import MODEL_ID, MCP_URL, AWS_REGION, API_PORT, TOKEN_PRICES
        assert isinstance(MODEL_ID, str)
        assert isinstance(MCP_URL, str)
        assert isinstance(AWS_REGION, str)
        assert isinstance(API_PORT, int)
        assert isinstance(TOKEN_PRICES, dict)

    def test_default_model_id(self):
        from config import MODEL_ID
        assert "anthropic" in MODEL_ID or "amazon" in MODEL_ID

    def test_default_mcp_url(self):
        from config import MCP_URL
        assert "localhost" in MCP_URL
        assert "8000" in MCP_URL

    def test_token_prices_has_default(self):
        from config import TOKEN_PRICES
        assert "default" in TOKEN_PRICES
        assert TOKEN_PRICES["default"] > 0


class TestAPIServerStructure:
    def test_api_server_importable(self):
        """Verify api_server.py defines the FastAPI app (without starting it)."""
        # We can't fully import because it tries to connect to MCP,
        # but we can verify the file is valid Python
        import ast
        path = os.path.join(os.path.dirname(__file__), "..", "agent", "api_server.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        # Check it defines the expected endpoints
        func_names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert "chat" in func_names
        assert "health" in func_names

    def test_agent_module_valid(self):
        """Verify agent.py is valid Python with expected class."""
        import ast
        path = os.path.join(os.path.dirname(__file__), "..", "agent", "agent.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert "TelcoRCAAgent" in class_names
