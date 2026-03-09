"""T-203: Synthetic log store - loads logs from JSON, serves by IMSI prefix."""
import json
import random
import logging
import os

logger = logging.getLogger("LogStore")

IMSI_MAP = {
    "999": "transport_jitter",
    "888": "radio_failure",
    "777": "normal",
    "666": "authentication_failure",
    "555": "core_network_failure",
    "444": "handover_failure",
    "333": "congestion",
    "222": "qos_violation",
}


class LogStore:
    """Loads synthetic logs from JSON file, serves by IMSI prefix."""

    def __init__(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_logs.json")
        with open(path) as f:
            self.scenarios = json.load(f)
        total = sum(len(v) if isinstance(v, list) else 1 for v in self.scenarios.values())
        logger.info(f"Loaded {total} log entries across {len(self.scenarios)} scenarios")

    def get_logs(self, imsi: str) -> str:
        """Return raw logs for an IMSI. Prefix determines scenario."""
        prefix = imsi[:3]
        scenario = IMSI_MAP.get(prefix, random.choice(list(self.scenarios.keys())))
        logs = self.scenarios[scenario]
        return random.choice(logs) if isinstance(logs, list) else logs
