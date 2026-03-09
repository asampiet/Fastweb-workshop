"""T-405: End-to-end integration tests. Requires all 3 services running."""
import pytest
import requests

BASE_URL = "http://localhost:8080"


def is_backend_running():
    try:
        requests.get(f"{BASE_URL}/api/health", timeout=3)
        return True
    except Exception:
        return False


skip_if_offline = pytest.mark.skipif(
    not is_backend_running(), reason="Backend not running (requires all services)"
)


@skip_if_offline
def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "bedrock_model" in data


@skip_if_offline
def test_youtube_buffering():
    """Demo scenario 1: transport jitter detected, high token reduction."""
    resp = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "Subscriber 999123 complains YouTube buffers constantly despite full signal bars"
    }, timeout=60)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["response"]) > 50
    assert data["metrics"] is not None
    assert data["metrics"]["token_reduction_pct"] > 90


@skip_if_offline
def test_call_drop():
    """Demo scenario 2: radio failure detected."""
    resp = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "Subscriber 888456 experiencing frequent call drops in downtown area"
    }, timeout=60)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["response"]) > 50
    assert data["metrics"] is not None


@skip_if_offline
def test_no_issue():
    """Demo scenario 3: normal logs, no anomalies."""
    resp = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "Check subscriber 777789 for issues, they called about billing"
    }, timeout=60)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["response"]) > 20


@skip_if_offline
def test_auth_failure():
    """Demo scenario 4: authentication failure isolated."""
    resp = requests.post(f"{BASE_URL}/api/chat", json={
        "message": "Subscriber 666012 cannot register on the network after SIM swap"
    }, timeout=60)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["response"]) > 50


@skip_if_offline
def test_invalid_request():
    resp = requests.post(f"{BASE_URL}/api/chat", json={"message": ""}, timeout=10)
    assert resp.status_code == 422  # Pydantic validation error
