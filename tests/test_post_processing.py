"""T-402: Unit tests for post-processing filter - all 4 rules + edge cases."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from post_processing_filter import apply_filter


def test_rule1_upf_cascade():
    events = [
        {"error_code": "UPF_DEGRADATION", "severity": "high"},
        {"error_code": "RRC_SETUP_FAILURE", "severity": "high"},
        {"error_code": "5GMM_CAUSE_CONGESTION", "severity": "medium"},
    ]
    filtered, rules = apply_filter(events)
    assert len(filtered) == 1
    assert filtered[0]["error_code"] == "UPF_DEGRADATION"
    assert "UPF_CASCADE" in rules


def test_rule2_core_cascade():
    events = [
        {"error_code": "5GMM_CAUSE_NETWORK_FAILURE", "severity": "high"},
        {"error_code": "RRC_SETUP_FAILURE", "severity": "high"},
        {"error_code": "PDU_SESSION_FAILURE", "severity": "high"},
    ]
    filtered, rules = apply_filter(events)
    assert all(not e["error_code"].startswith("RRC_") for e in filtered)
    assert "CORE_CASCADE" in rules


def test_rule3_auth_isolation():
    events = [
        {"error_code": "AUTHENTICATION_FAILURE", "severity": "high"},
        {"error_code": "REGISTRATION_REJECTED", "severity": "high"},
        {"error_code": "RRC_SETUP_FAILURE", "severity": "medium"},
    ]
    filtered, rules = apply_filter(events)
    assert len(filtered) == 2
    assert {e["error_code"] for e in filtered} == {"AUTHENTICATION_FAILURE", "REGISTRATION_REJECTED"}
    assert "AUTH_ISOLATION" in rules


def test_rule4_dedup():
    events = [
        {"error_code": "TCP_RETRANSMISSION", "severity": "high"},
        {"error_code": "TCP_RETRANSMISSION", "severity": "high"},
    ]
    filtered, rules = apply_filter(events)
    assert len(filtered) == 1
    assert "DEDUP" in rules


def test_no_rules_applied():
    events = [{"error_code": "TCP_RETRANSMISSION", "severity": "high"}]
    filtered, rules = apply_filter(events)
    assert filtered == events
    assert rules == []


def test_empty_events():
    filtered, rules = apply_filter([])
    assert filtered == []
    assert rules == []
