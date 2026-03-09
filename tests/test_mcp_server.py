"""T-403: Unit tests for MCP server components (log store, grammar, schema)."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from log_store import LogStore, IMSI_MAP

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_server", "telco_grammar.gbnf")


class TestLogStore:
    @pytest.fixture(scope="class")
    def store(self):
        return LogStore(os.path.join(DATA_DIR, "synthetic_logs.json"))

    def test_loads_all_scenarios(self, store):
        assert len(store.scenarios) == 8

    def test_imsi_prefix_mapping(self, store):
        for prefix, scenario in IMSI_MAP.items():
            logs = store.get_logs(f"{prefix}000")
            assert len(logs) > 0, f"No logs for prefix {prefix} ({scenario})"

    def test_unknown_imsi_returns_logs(self, store):
        logs = store.get_logs("111999")
        assert len(logs) > 0

    def test_logs_contain_timestamps(self, store):
        logs = store.get_logs("999123")
        assert "2025-" in logs

    def test_normal_logs_no_errors(self, store):
        logs = store.get_logs("777789")
        assert "FAILURE" not in logs


class TestSyntheticLogs:
    def test_json_valid(self):
        with open(os.path.join(DATA_DIR, "synthetic_logs.json")) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_all_scenarios_have_samples(self):
        with open(os.path.join(DATA_DIR, "synthetic_logs.json")) as f:
            data = json.load(f)
        for scenario, logs in data.items():
            assert isinstance(logs, list), f"{scenario} should be a list"
            assert len(logs) >= 2, f"{scenario} needs ≥2 samples, got {len(logs)}"


class TestGrammar:
    def test_grammar_file_exists(self):
        assert os.path.isfile(GRAMMAR_PATH)

    def test_grammar_has_root_rule(self):
        with open(GRAMMAR_PATH) as f:
            content = f.read()
        assert "root" in content
        assert "events" in content
        assert "error_code" in content
        assert "severity" in content

    def test_grammar_severity_constrained(self):
        with open(GRAMMAR_PATH) as f:
            content = f.read()
        assert "high" in content
        assert "medium" in content
        assert "low" in content
