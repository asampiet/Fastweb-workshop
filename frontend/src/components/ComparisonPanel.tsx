"use client";
import { useMetrics } from "@/context/MetricsContext";

function fmt(n: number): string {
  return n.toLocaleString();
}

export default function ComparisonPanel() {
  const { metrics } = useMetrics();

  if (!metrics) {
    return (
      <div className="p-6 text-center text-gray-400 text-sm">
        <p className="text-2xl mb-2">📊</p>
        <p>Run a query to see Semantic Compression metrics</p>
      </div>
    );
  }

  const pct = metrics.token_reduction_pct;

  return (
    <div className="p-5 space-y-5 text-sm">
      <h2 className="font-semibold text-gray-800 text-base">📊 Semantic Compression</h2>

      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-gray-500">Raw Logs</span>
          <span className="font-mono">{fmt(metrics.raw_log_bytes)} bytes</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Raw Tokens (est.)</span>
          <span className="font-mono">~{fmt(metrics.raw_log_tokens_est)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Filtered Output</span>
          <span className="font-mono">{fmt(metrics.filtered_output_bytes)} bytes</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Filtered Tokens (est.)</span>
          <span className="font-mono">~{fmt(metrics.filtered_output_tokens_est)}</span>
        </div>
      </div>

      <div>
        <div className="flex justify-between mb-1">
          <span className="text-gray-500">Token Reduction</span>
          <span className="font-semibold text-green-600">{pct}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-green-500 h-3 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
      </div>

      <div className="border-t pt-4 space-y-2">
        <div className="flex justify-between">
          <span className="text-gray-500">💰 Cost Saved / Query</span>
          <span className="font-semibold text-green-600">${metrics.estimated_cost_savings_usd.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">⏱️ SLM Inference</span>
          <span className="font-mono">{metrics.slm_inference_latency_sec}s</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">⏱️ Total Response</span>
          <span className="font-mono">{metrics.total_response_time_sec}s</span>
        </div>
      </div>
    </div>
  );
}
