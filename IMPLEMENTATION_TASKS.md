# Implementation Task List

**Date:** 2026-03-09
**Source:** DESIGN_SPECIFICATION.md v1.0

---

## Phase 0: Project Scaffolding & Infrastructure

### T-001: Initialize project structure
- **Files:** All directories as per §10 of requirements
- **Action:** Create directory tree: `fine_tuning/`, `mcp_server/`, `agent/`, `frontend/`, `tests/`, `scripts/`
- **Acceptance:** `tree` output matches design spec §9 file inventory
- **Estimate:** 15 min

### T-002: Create setup_instance.sh
- **Files:** `scripts/setup_instance.sh`
- **Ref:** Design §6.1
- **Action:** System packages, Python venv, pip installs (torch, transformers, peft, trl, bitsandbytes, datasets, accelerate, fastapi, uvicorn, mcp, strands-agents, boto3, llama-cpp-python with CUDA), Node.js 20, llama.cpp build with CUDA
- **Acceptance:** Script runs on fresh Deep Learning AMI without errors
- **Estimate:** 30 min

### T-003: Create download_model.sh
- **Files:** `scripts/download_model.sh`
- **Ref:** Design §6.2
- **Action:** Download Qwen3-14B from HuggingFace Hub to `models/Qwen3-14B/`
- **Acceptance:** Model files present, tokenizer loadable
- **Estimate:** 10 min

### T-004: Create requirements.txt
- **Files:** `requirements.txt`
- **Action:** Pin all Python dependencies with versions
- **Acceptance:** `pip install -r requirements.txt` succeeds
- **Estimate:** 10 min

---

## Phase 1: Fine-Tuning Pipeline (C1)

### T-101: Implement train.py
- **Files:** `fine_tuning/train.py`
- **Ref:** Design §3.2
- **Action:** CLI arg parser, dataset loading + validation (assert 1300), BitsAndBytesConfig (4-bit NF4), model load, LoraConfig (r=16, α=32, all linear layers), SFTTrainer, save adapter to `output/adapter/`, print summary
- **Deps:** T-002, T-003
- **Acceptance:** Runs on g6.12xlarge, loss < 0.30, adapter files saved
- **Estimate:** 60 min

### T-102: Implement merge_adapter.py
- **Files:** `fine_tuning/merge_adapter.py`
- **Ref:** Design §3.3
- **Action:** Load base model FP16, load adapter, merge_and_unload(), save to `output/merged_model/`
- **Deps:** T-101
- **Acceptance:** Merged model loadable by transformers
- **Estimate:** 20 min

### T-103: Implement convert_to_gguf.sh
- **Files:** `fine_tuning/convert_to_gguf.sh`
- **Ref:** Design §3.4
- **Action:** convert_hf_to_gguf.py → FP16 GGUF, quantize → Q4_K_M, cleanup intermediate
- **Deps:** T-102, T-002 (llama.cpp build)
- **Acceptance:** `output/qwen3-14b-telco-Q4_K_M.gguf` exists, ~9 GB
- **Estimate:** 20 min

### T-104: Implement validate.py
- **Files:** `fine_tuning/validate.py`
- **Ref:** Design §3.5
- **Action:** Load GGUF + grammar, iterate 1000 test examples, compute TP/FP/FN, F1/Precision/Recall/EM, per-scenario breakdown, latency report, PASS/FAIL against thresholds
- **Deps:** T-103, T-201 (needs inference_engine + grammar), T-202 (needs filter)
- **Acceptance:** Prints report, F1 ≥ 0.85 raw / ≥ 0.90 filtered
- **Estimate:** 45 min


---

## Phase 2: Smart MCP Server (C2)

### T-201: Implement inference_engine.py
- **Files:** `mcp_server/inference_engine.py`
- **Ref:** Design §4.3
- **Action:** `TelcoInferenceEngine` singleton class, loads GGUF via `Llama()` with `n_gpu_layers=-1`, loads GBNF grammar via `LlamaGrammar.from_file()`, `analyze()` method builds ChatML prompt, calls `llm()` with grammar/temp=0.1/repeat_penalty=1.2/max_tokens=512, returns events + raw_log_bytes + latency, error handling returns error JSON
- **Deps:** T-002 (llama-cpp-python installed)
- **Acceptance:** Loads model, returns valid JSON for sample input, GPU detected
- **Estimate:** 45 min

### T-202: Implement post_processing_filter.py
- **Files:** `mcp_server/post_processing_filter.py`
- **Ref:** Design §4.4
- **Action:** `apply_filter(events) → (filtered, rules)`. Rule 1: UPF cascade (remove RRC_SETUP_FAILURE, RRC_REESTABLISHMENT_FAILURE, 5GMM_CAUSE_CONGESTION if UPF_DEGRADATION present). Rule 2: Core cascade (remove RRC_ prefixed if 5GMM_CAUSE_NETWORK_FAILURE or PDU_SESSION_FAILURE). Rule 3: Auth isolation (keep only AUTHENTICATION_FAILURE + REGISTRATION_REJECTED). Rule 4: Dedup. Log applied rules.
- **Deps:** None (pure Python)
- **Acceptance:** All 6 test cases from Design §7.2 pass
- **Estimate:** 30 min

### T-203: Implement log_store.py
- **Files:** `mcp_server/log_store.py`
- **Ref:** Design §4.5
- **Action:** `LogStore` class, loads `data/synthetic_logs.json` at init, IMSI prefix → scenario mapping (999→transport_jitter, 888→radio_failure, etc.), `get_logs(imsi)` returns log string, unknown IMSI → random scenario
- **Deps:** T-204 (needs synthetic_logs.json)
- **Acceptance:** Returns different logs for different IMSI prefixes
- **Estimate:** 20 min

### T-204: Create synthetic_logs.json
- **Files:** `data/synthetic_logs.json`
- **Action:** Create JSON with 8 scenario keys, each containing 3-5 sample raw log strings covering the scenario's error patterns. Logs must include timestamps, protocol layers, error codes, cell IDs matching training data patterns.
- **Deps:** None
- **Acceptance:** Valid JSON, all 8 scenarios present, logs realistic
- **Estimate:** 30 min

### T-205: Create telco_grammar.gbnf
- **Files:** `mcp_server/telco_grammar.gbnf`
- **Ref:** Design §4.7
- **Action:** GBNF grammar: root→object→events array→event-obj (error_code string + severity enum high/medium/low)
- **Deps:** None
- **Acceptance:** Grammar parseable by LlamaGrammar.from_file()
- **Estimate:** 10 min

### T-206: Implement server.py
- **Files:** `mcp_server/server.py`
- **Ref:** Design §4.6
- **Action:** Read MODEL_PATH env var, init TelcoInferenceEngine singleton, init LogStore, create FastMCP("TelcoEdgeFilter"), register `analyze_network_telemetry` tool (fetches logs → inference → filter → return JSON with metadata), dual logging (stdout human-readable + file JSON), start SSE on MCP_PORT
- **Deps:** T-201, T-202, T-203, T-204, T-205
- **Acceptance:** Server starts, tool discoverable via MCP, returns valid JSON for test query
- **Estimate:** 30 min

---

## Phase 3: Web GUI with Strands Agent (C3)

### T-301: Implement config.py
- **Files:** `agent/config.py`
- **Ref:** Design §5.4
- **Action:** Read env vars: BEDROCK_MODEL_ID (default Haiku), MCP_SERVER_URL (default localhost:8000/sse), AWS_REGION, API_PORT. TOKEN_PRICES dict for cost estimation.
- **Deps:** None
- **Acceptance:** Imports without error, defaults correct
- **Estimate:** 5 min

### T-302: Implement agent.py
- **Files:** `agent/agent.py`
- **Ref:** Design §5.2
- **Action:** `TelcoRCAAgent` class. Init: create BedrockModel, MCPClient (SSE), start MCP client, discover tools, create Strands Agent with system prompt + tools. `process_query(message)`: invoke agent, extract metrics from tool results (raw_log_bytes, filtered_output_bytes, latency, token estimates, cost savings), return response + metrics dict. `shutdown()`: stop MCP client.
- **Deps:** T-301, T-206 (MCP server must be running for tool discovery)
- **Acceptance:** Agent connects to MCP, discovers tool, processes test query via Bedrock
- **Estimate:** 60 min

### T-303: Implement api_server.py
- **Files:** `agent/api_server.py`
- **Ref:** Design §5.3
- **Action:** FastAPI app. POST `/api/chat`: Pydantic ChatRequest (message 1-2000 chars), calls agent.process_query(), returns ChatResponse (response + metrics). GET `/api/health`: returns status + model ID. CORS middleware. Error handling: 502 for Bedrock errors, 503 for MCP errors. Uvicorn on API_PORT.
- **Deps:** T-302
- **Acceptance:** Health endpoint returns 200, chat endpoint processes query
- **Estimate:** 30 min

### T-304: Initialize Next.js frontend
- **Files:** `frontend/package.json`, `frontend/next.config.js`, `frontend/tailwind.config.js`, `frontend/tsconfig.json`, `frontend/src/app/layout.tsx`
- **Action:** `npx create-next-app@latest` with TypeScript + Tailwind + App Router. Configure layout with MetricsProvider wrapper.
- **Deps:** T-002 (Node.js installed)
- **Acceptance:** `npm run dev` starts on port 3000
- **Estimate:** 15 min

### T-305: Implement MetricsContext.tsx
- **Files:** `frontend/src/context/MetricsContext.tsx`
- **Ref:** Design §5.5.1
- **Action:** React Context with Metrics interface (8 fields), MetricsProvider component, useMetrics hook
- **Deps:** T-304
- **Acceptance:** Context importable, state updates propagate
- **Estimate:** 10 min

### T-306: Implement API proxy route
- **Files:** `frontend/src/app/api/chat/route.ts`
- **Ref:** Design §5.5.3
- **Action:** POST handler, reads BACKEND_URL from env (default localhost:8080), proxies request body to Python backend, returns response
- **Deps:** T-304
- **Acceptance:** Proxy forwards requests correctly
- **Estimate:** 10 min

### T-307: Implement MessageBubble.tsx
- **Files:** `frontend/src/components/MessageBubble.tsx`
- **Ref:** Design §5.5.4
- **Action:** Props: role (user|agent), content (string). User: right-aligned blue. Agent: left-aligned gray with markdown rendering (bold, code blocks, lists).
- **Deps:** T-304
- **Acceptance:** Renders both roles correctly
- **Estimate:** 20 min

### T-308: Implement ChatInterface.tsx
- **Files:** `frontend/src/components/ChatInterface.tsx`
- **Ref:** Design §5.5.4
- **Action:** State: messages[], isLoading. Input field + send button. On submit: append user msg, POST /api/chat, append agent msg, call setMetrics from context. Loading indicator. Auto-scroll to bottom. Error display.
- **Deps:** T-305, T-306, T-307
- **Acceptance:** Sends message, displays response, updates metrics context
- **Estimate:** 40 min

### T-309: Implement ComparisonPanel.tsx
- **Files:** `frontend/src/components/ComparisonPanel.tsx`
- **Ref:** Design §5.5.5
- **Action:** Reads metrics from useMetrics(). Empty state: "Run a query to see metrics". With data: raw bytes, filtered bytes, token reduction % with progress bar, cost savings, SLM latency, total time.
- **Deps:** T-305
- **Acceptance:** Shows placeholder when no data, updates on query
- **Estimate:** 30 min

### T-310: Implement page.tsx
- **Files:** `frontend/src/app/page.tsx`
- **Ref:** Design §5.5.2
- **Action:** Responsive layout: ChatInterface (2/3 desktop) + ComparisonPanel (1/3 sidebar desktop, below on mobile). Wrap in MetricsProvider.
- **Deps:** T-308, T-309
- **Acceptance:** Responsive layout works on desktop and mobile viewports
- **Estimate:** 15 min


---

## Phase 4: Testing

### T-401: Implement test_training.py
- **Files:** `tests/test_training.py`
- **Action:** Test dataset loads (1300 examples), has positive + negative, follows ChatML format, system prompt present in each example
- **Deps:** T-101
- **Acceptance:** All tests pass with pytest
- **Estimate:** 20 min

### T-402: Implement test_post_processing.py
- **Files:** `tests/test_post_processing.py`
- **Ref:** Design §7.2
- **Action:** 6 test cases: UPF cascade, core cascade, auth isolation, dedup, no rules, empty events
- **Deps:** T-202
- **Acceptance:** All 6 tests pass
- **Estimate:** 20 min

### T-403: Implement test_mcp_server.py
- **Files:** `tests/test_mcp_server.py`
- **Action:** Test server starts (mock model), tool registered, tool returns valid JSON schema, grammar file parseable
- **Deps:** T-206
- **Acceptance:** All tests pass
- **Estimate:** 30 min

### T-404: Implement test_agent.py
- **Files:** `tests/test_agent.py`
- **Action:** Test agent init, health endpoint returns 200, chat endpoint validates input (rejects empty, accepts valid)
- **Deps:** T-303
- **Acceptance:** All tests pass
- **Estimate:** 20 min

### T-405: Implement test_e2e.py
- **Files:** `tests/test_e2e.py`
- **Ref:** Design §7.3
- **Action:** 3 integration tests: YouTube buffering (token reduction >90%), no issue (agent reports no anomalies), health check. Requires all services running.
- **Deps:** T-206, T-303
- **Acceptance:** All tests pass against running services
- **Estimate:** 30 min

---

## Phase 5: Integration & Launch Scripts

### T-501: Create run_all.sh
- **Files:** `scripts/run_all.sh`
- **Ref:** Design §6.3
- **Action:** Start MCP server (wait 65s for model load), start agent backend (wait 3s), start frontend. Print URLs. Trap Ctrl+C to kill all.
- **Deps:** T-206, T-303, T-310
- **Acceptance:** All 3 services start and respond
- **Estimate:** 15 min

### T-502: Create README.md
- **Files:** `README.md`
- **Action:** Workshop overview, prerequisites, setup steps, execution order (8 phases from requirements §11), demo scenarios, troubleshooting
- **Deps:** All tasks
- **Acceptance:** A participant can follow README to complete the workshop
- **Estimate:** 30 min

---

## Summary

| Phase | Tasks | Estimated Time |
|-------|-------|---------------|
| Phase 0: Scaffolding | T-001 to T-004 | 1h 5min |
| Phase 1: Fine-Tuning | T-101 to T-104 | 2h 25min |
| Phase 2: MCP Server | T-201 to T-206 | 2h 45min |
| Phase 3: Web GUI | T-301 to T-310 | 3h 35min |
| Phase 4: Testing | T-401 to T-405 | 2h 0min |
| Phase 5: Integration | T-501 to T-502 | 45min |
| **Total** | **31 tasks** | **~12h 35min** |

---

## Dependency Graph

```
T-001 ──► T-002 ──► T-003 ──► T-101 ──► T-102 ──► T-103 ─┐
                                                            │
T-004                                                       ▼
                                                     T-104 (validate)
T-205 (grammar) ──┐                                        ▲
T-204 (logs)   ──┐│                                        │
                 ││                                        │
                 ▼▼                                        │
T-201 ──► T-203 ──► T-206 (MCP server) ──► T-302 ──► T-303
T-202 ──────────────┘  │                              │
                       │                              ▼
                       │                    T-306 ──► T-308 ──► T-310
                       │                    T-307 ──┘           ▲
                       │                    T-305 ──► T-309 ────┘
                       │                    T-304 ──┘
                       │
                       ▼
              T-401, T-402, T-403, T-404, T-405 ──► T-501 ──► T-502
```

**Critical path:** T-002 → T-003 → T-101 → T-102 → T-103 → T-201 → T-206 → T-302 → T-303 → T-310 → T-501

---

*End of Implementation Task List*
