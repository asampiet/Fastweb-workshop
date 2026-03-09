#!/usr/bin/env python3
"""T-104: Validate fine-tuned model against test set. Reports F1/Precision/Recall."""
import argparse
import json
import sys
import os
import time
from collections import defaultdict

# Import from mcp_server (built in Phase 2)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from inference_engine import TelcoInferenceEngine
from post_processing_filter import apply_filter

def parse_ground_truth(assistant_text: str) -> set:
    """Extract expected error codes from the assistant response in training data."""
    try:
        data = json.loads(assistant_text)
        return {e["error_code"] for e in data.get("events", [])}
    except (json.JSONDecodeError, KeyError):
        return set()

def extract_assistant_text(text: str) -> str:
    """Extract the assistant turn from a pre-formatted ChatML text field."""
    marker = "<|im_start|>assistant\n"
    idx = text.find(marker)
    if idx == -1:
        return "{}"
    start = idx + len(marker)
    end = text.find("<|im_end|>", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()

def extract_user_text(text: str) -> str:
    """Extract the user turn (raw logs) from a pre-formatted ChatML text field."""
    start_marker = "<|im_start|>user\n"
    end_marker = "<|im_end|>"
    idx = text.find(start_marker)
    if idx == -1:
        return ""
    start = idx + len(start_marker)
    end = text.find(end_marker, start)
    return text[start:end].strip() if end != -1 else text[start:].strip()

def main():
    p = argparse.ArgumentParser(description="Validate model on test set")
    p.add_argument("--model-path", required=True, help="Path to GGUF model (absolute)")
    p.add_argument("--grammar-path", default="mcp_server/telco_grammar.gbnf")
    p.add_argument("--test-data", default="data/test_data.json")
    p.add_argument("--with-filter", action="store_true", help="Apply post-processing filter")
    p.add_argument("--limit", type=int, default=0, help="Limit examples (0=all)")
    args = p.parse_args()

    print(f"Loading model: {args.model_path}")
    engine = TelcoInferenceEngine.get_instance(args.model_path, args.grammar_path)

    print(f"Loading test data: {args.test_data}")
    test_data = json.load(open(args.test_data))
    if args.limit > 0:
        test_data = test_data[:args.limit]
    print(f"Examples: {len(test_data)}, filter: {'ON' if args.with_filter else 'OFF'}")

    totals = {"tp": 0, "fp": 0, "fn": 0, "exact": 0}
    per_scenario = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "count": 0})
    latencies = []

    for i, example in enumerate(test_data):
        raw_logs = extract_user_text(example["text"])
        expected = parse_ground_truth(extract_assistant_text(example["text"]))

        result = engine.analyze(raw_logs)
        events = result["events"]
        if args.with_filter:
            events, _ = apply_filter(events)

        predicted = {e["error_code"] for e in events}
        tp = len(predicted & expected)
        fp = len(predicted - expected)
        fn = len(expected - predicted)

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        if predicted == expected:
            totals["exact"] += 1
        latencies.append(result["inference_latency_sec"])

        scenario = example.get("scenario", "unknown")
        per_scenario[scenario]["tp"] += tp
        per_scenario[scenario]["fp"] += fp
        per_scenario[scenario]["fn"] += fn
        per_scenario[scenario]["count"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_data)}] latency={result['inference_latency_sec']:.1f}s")

    # Compute metrics
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    em = totals["exact"] / len(test_data) * 100
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    # Report
    mode = "WITH filter" if args.with_filter else "RAW (no filter)"
    print(f"\n{'='*50}")
    print(f"  Model Validation Report ({mode})")
    print(f"{'='*50}")
    print(f"  Examples:      {len(test_data)}")
    print(f"  F1 Score:      {f1:.3f}")
    print(f"  Precision:     {precision:.3f}")
    print(f"  Recall:        {recall:.3f}")
    print(f"  Exact Match:   {em:.1f}%")
    print(f"  TP/FP/FN:      {tp}/{fp}/{fn}")
    print(f"  Avg Latency:   {avg_lat:.2f}s")
    print()

    # Per-scenario breakdown
    print(f"  {'Scenario':<25} {'Count':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Status'}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*10}")
    for scenario in sorted(per_scenario.keys()):
        s = per_scenario[scenario]
        status = "✅ PERFECT" if s["fp"] == 0 and s["fn"] == 0 else "⚠️" if s["fp"] + s["fn"] < 10 else "❌"
        print(f"  {scenario:<25} {s['count']:>5} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5} {status}")

    # Pass/fail
    threshold = 0.90 if args.with_filter else 0.85
    result = "PASS" if f1 >= threshold else "FAIL"
    print(f"\n  Result: {result} (F1 {f1:.3f} {'≥' if f1 >= threshold else '<'} {threshold} threshold)")

if __name__ == "__main__":
    main()
