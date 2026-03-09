"""T-202: Deterministic post-processing filter for sympathetic noise removal."""
import logging

logger = logging.getLogger("PostProcessingFilter")

UPF_SYMPATHETIC = {"RRC_SETUP_FAILURE", "RRC_REESTABLISHMENT_FAILURE", "5GMM_CAUSE_CONGESTION"}
AUTH_KEEP = {"AUTHENTICATION_FAILURE", "REGISTRATION_REJECTED"}


def apply_filter(events: list[dict]) -> tuple[list[dict], list[str]]:
    """Apply deterministic rules to remove sympathetic noise.
    Returns (filtered_events, applied_rule_names)."""
    codes = {e["error_code"] for e in events}
    applied = []
    filtered = list(events)

    # Rule 1: UPF Degradation Cascade
    if "UPF_DEGRADATION" in codes:
        before = len(filtered)
        filtered = [e for e in filtered if e["error_code"] not in UPF_SYMPATHETIC]
        if len(filtered) < before:
            applied.append("UPF_CASCADE")

    # Rule 2: Core Network Failure Cascade
    codes_now = {e["error_code"] for e in filtered}
    if "5GMM_CAUSE_NETWORK_FAILURE" in codes_now or "PDU_SESSION_FAILURE" in codes_now:
        before = len(filtered)
        filtered = [e for e in filtered if not e["error_code"].startswith("RRC_")]
        if len(filtered) < before:
            applied.append("CORE_CASCADE")

    # Rule 3: Authentication Failure Isolation
    codes_now = {e["error_code"] for e in filtered}
    if "AUTHENTICATION_FAILURE" in codes_now:
        before = len(filtered)
        filtered = [e for e in filtered if e["error_code"] in AUTH_KEEP]
        if len(filtered) < before:
            applied.append("AUTH_ISOLATION")

    # Rule 4: Deduplication
    seen = set()
    deduped = []
    for e in filtered:
        if e["error_code"] not in seen:
            seen.add(e["error_code"])
            deduped.append(e)
    if len(deduped) < len(filtered):
        applied.append("DEDUP")
    filtered = deduped

    logger.info(f"Filter: {len(events)} → {len(filtered)} events, rules={applied}")
    return filtered, applied
