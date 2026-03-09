"use client";
import { createContext, useContext, useState, ReactNode } from "react";

export interface Metrics {
  raw_log_bytes: number;
  raw_log_tokens_est: number;
  filtered_output_bytes: number;
  filtered_output_tokens_est: number;
  token_reduction_pct: number;
  estimated_cost_savings_usd: number;
  slm_inference_latency_sec: number;
  total_response_time_sec: number;
}

interface MetricsContextType {
  metrics: Metrics | null;
  setMetrics: (m: Metrics | null) => void;
}

const MetricsContext = createContext<MetricsContextType>({ metrics: null, setMetrics: () => {} });

export function MetricsProvider({ children }: { children: ReactNode }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  return <MetricsContext.Provider value={{ metrics, setMetrics }}>{children}</MetricsContext.Provider>;
}

export const useMetrics = () => useContext(MetricsContext);
