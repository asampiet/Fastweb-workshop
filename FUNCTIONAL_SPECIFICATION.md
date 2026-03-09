# Functional Specification: Telco Autonomous Operations Workshop

**Document Version:** 1.0
**Date:** 2026-03-09
**Classification:** Functional Specification
**Author:** Angelo Sampietro
**Source:** REQUIREMENTS_DOCUMENT.md v1.0

---

## 1. Introduction

### 1.1 Purpose
This document translates the workshop requirements into detailed functional behaviors, use cases, interface contracts, data models, and error handling specifications. It serves as the authoritative reference for the design and implementation phases.

### 1.2 Scope
Three components on a single EC2 GPU instance (g6.12xlarge recommended):
- **C1:** Fine-Tuning Pipeline (Qwen3-14B QLoRA → GGUF conversion)
- **C2:** Smart MCP Server (llama.cpp inference + post-processing filter + MCP SSE)
- **C3:** Web GUI with Strands Agent (React/Next.js + Python backend + Bedrock)

### 1.3 Actors

| Actor | Description |
|-------|-------------|
| **Workshop Participant** | Executes scripts, builds components, runs the demo |
| **End User (Demo)** | Interacts with the GUI chat interface during the demo |
| **Strands Agent** | Autonomous software agent orchestrating Bedrock + MCP tools |
| **Frontier Model** | Amazon Bedrock LLM (Claude 4.5 Haiku / Nova 2 Pro) performing reasoning |
| **Local SLM** | Fine-tuned Qwen3-14B running on llama.cpp performing semantic filtering |

---

## 2. Use Cases

### UC-1: Fine-Tune the Base Model

| Field | Value |
|-------|-------|
| **Actor** | Workshop Participant |
| **Preconditions** | EC2 instance running, Python env configured, Qwen3-14B downloaded, training dataset at `data/training_data.json` |
| **Trigger** | Participant runs `python fine_tuning/train.py` |

**Main Flow:**
1. Script loads training dataset (1,300 examples) and validates format
2. Script loads Qwen3-14B in 4-bit NF4 quantization via bitsandbytes
3. Script attaches QLoRA adapters (r=16, α=32) to all linear layers
4. SFTTrainer runs for 1 epoch (~325 steps, ~31 min on g6.12xlarge)
5. Script logs loss and token accuracy per step to stdout
6. Script saves LoRA adapter to `output/adapter/`
7. Script prints final training metrics summary

**Postconditions:** LoRA adapter saved; training loss < 0.30; token accuracy > 90%

**Alternative Flows:**
- **AF-1a:** CUDA OOM → Script logs error with VRAM usage, suggests reducing max_seq_length to 1024
- **AF-1b:** Dataset file missing → Script exits with clear error message and expected path

---

### UC-2: Convert Model to GGUF

| Field | Value |
|-------|-------|
| **Actor** | Workshop Participant |
| **Preconditions** | UC-1 completed, LoRA adapter at `output/adapter/` |
| **Trigger** | Participant runs `python fine_tuning/merge_adapter.py` then `bash fine_tuning/convert_to_gguf.sh` |

**Main Flow:**
1. `merge_adapter.py` loads base model + LoRA adapter
2. Merges adapter into base model via `merge_and_unload()`
3. Saves merged model in HuggingFace format to `output/merged_model/`
4. `convert_to_gguf.sh` runs `llama.cpp/convert_hf_to_gguf.py` → FP16 GGUF
5. Script runs `llama.cpp/tools/quantize/quantize` with Q4_K_M profile
6. Outputs final model to `output/qwen3-14b-telco-Q4_K_M.gguf` (~9 GB)

**Postconditions:** GGUF file exists and is loadable by llama.cpp

**Alternative Flows:**
- **AF-2a:** Disk space insufficient → Script checks available space before conversion, exits with warning
- **AF-2b:** llama.cpp not built → Script checks for quantize binary, prints build instructions

---

### UC-3: Validate Model Quality

| Field | Value |
|-------|-------|
| **Actor** | Workshop Participant |
| **Preconditions** | UC-2 completed, GGUF model available, test dataset at `data/test_data.json` |
| **Trigger** | Participant runs `python fine_tuning/validate.py` |

**Main Flow:**
1. Script loads GGUF model via llama-cpp-python with GBNF grammar
2. Iterates through 1,000 test examples
3. For each example: runs inference → applies post-processing filter → compares to ground truth
4. Computes TP, FP, FN per example
5. Calculates aggregate F1, Precision, Recall, Exact Match
6. Generates per-scenario breakdown table
7. Reports average inference latency
8. Prints PASS/FAIL against minimum thresholds (F1 ≥ 0.85 raw, ≥ 0.90 filtered)

**Postconditions:** Validation report printed; model meets minimum quality thresholds

**Alternative Flows:**
- **AF-3a:** Model below threshold → Script prints WARNING with per-scenario failure analysis and suggests retraining with adjusted hyperparameters

---

### UC-4: Start Smart MCP Server

| Field | Value |
|-------|-------|
| **Actor** | Workshop Participant |
| **Preconditions** | UC-2 completed, GGUF model available |
| **Trigger** | Participant runs `python mcp_server/server.py` |

**Main Flow:**
1. Server loads GGUF model into GPU via llama-cpp-python (~60s)
2. Server initializes GBNF grammar from `telco_grammar.gbnf`
3. Server registers `analyze_network_telemetry` tool with FastMCP
4. Server starts SSE transport on port 8000
5. Server logs "Ready" with model info and VRAM usage

**Postconditions:** MCP server listening on `http://localhost:8000/sse`, tool discoverable

**Alternative Flows:**
- **AF-4a:** Port 8000 in use → Server logs error, suggests alternative port via `MCP_PORT` env var
- **AF-4b:** GPU not available → Server falls back to CPU inference, logs warning about degraded latency

---

### UC-5: Run Diagnostic Query (End-to-End Demo)

| Field | Value |
|-------|-------|
| **Actor** | End User (Demo) |
| **Preconditions** | MCP server running (UC-4), Agent backend running, Frontend running |
| **Trigger** | User types a natural language query in the chat interface |

**Main Flow:**
1. Frontend sends `POST /api/chat {"message": "..."}` to Python backend
2. Backend passes message to Strands agent
3. Strands agent sends message + system prompt + tool definitions to Bedrock
4. Frontier model reasons about the query, extracts IMSI and context
5. Frontier model invokes `analyze_network_telemetry(imsi, context)` tool
6. Strands agent forwards tool call to MCP server via SSE
7. MCP server fetches synthetic raw logs for the IMSI
8. MCP server runs logs through llama.cpp with GBNF grammar → raw JSON
9. MCP server applies post-processing filter → clean JSON
10. MCP server returns clean JSON + metrics (raw size, filtered size, latency) to agent
11. Strands agent returns tool result to Bedrock
12. Frontier model synthesizes diagnosis with recommendations
13. Backend returns `{"response": "...", "metrics": {...}}` to frontend
14. Frontend displays agent response in chat + updates comparison panel

**Postconditions:** User sees diagnosis; comparison panel shows token reduction ≥ 90%

**Alternative Flows:**
- **AF-5a:** Bedrock API error → Backend returns error message; frontend displays "Service temporarily unavailable"
- **AF-5b:** MCP server unreachable → Agent reports tool failure; frontend shows "Edge inference unavailable"
- **AF-5c:** SLM inference timeout (>30s) → MCP server returns error JSON `{"events": [], "status": "error"}`
- **AF-5d:** No IMSI in query → Frontier model asks user to provide subscriber identifier
- **AF-5e:** Normal logs (no failure) → SLM returns `{"events": []}`, agent reports "No anomalies detected"

---

## 3. Data Models

### 3.1 Training Example Schema

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a 3GPP Protocol Expert specializing in Root Cause Analysis.\nAnalyze the logs provided.\n1. Ignore routine Heartbeats and KeepAlives.\n2. Filter out sympathetic noise (consequential failures).\n3. Identify the primary causal chain (NAS, NGAP, or RRC).\n4. Output ONLY valid JSON according to the schema."
    },
    {
      "role": "user",
      "content": "<raw 3GPP log lines>"
    },
    {
      "role": "assistant",
      "content": "{\"events\": [{\"error_code\": \"...\", \"severity\": \"...\"}]}"
    }
  ]
}
```

### 3.2 MCP Tool Output Schema

```json
{
  "events": [
    {
      "error_code": "string (e.g., TCP_RETRANSMISSION, RRC_SETUP_FAILURE)",
      "severity": "string (high | medium | low)"
    }
  ]
}
```

Valid states:
- **Failure detected:** `events` array has 1+ items
- **No failure:** `events` is empty array `[]`
- **Error:** `{"events": [], "status": "error", "message": "..."}`

### 3.3 Backend API Response Schema

```json
{
  "response": "string (agent's natural language diagnosis)",
  "metrics": {
    "raw_log_bytes": 45230,
    "raw_log_tokens_est": 11308,
    "filtered_output_bytes": 187,
    "filtered_output_tokens_est": 47,
    "token_reduction_pct": 99.6,
    "estimated_cost_savings_usd": 0.0158,
    "slm_inference_latency_sec": 6.2,
    "total_response_time_sec": 12.4
  }
}
```

### 3.4 Synthetic Log Data Model

Each IMSI maps to a scenario with pre-generated raw logs:

| IMSI Pattern | Scenario | Log Lines | Raw Size |
|-------------|----------|-----------|----------|
| 999xxx | transport_jitter | 50-200 | 5-40 KB |
| 888xxx | radio_failure | 50-200 | 5-40 KB |
| 777xxx | normal | 20-50 | 2-10 KB |
| 666xxx | authentication_failure | 30-80 | 3-15 KB |
| 555xxx | core_network_failure | 50-200 | 5-40 KB |
| 444xxx | handover_failure | 40-150 | 4-30 KB |
| 333xxx | congestion | 50-200 | 5-40 KB |
| 222xxx | qos_violation | 30-100 | 3-20 KB |

Unknown IMSIs return a random scenario from the pool.


---

## 4. Interface Specifications

### 4.1 MCP Server Interface (SSE Transport)

| Property | Value |
|----------|-------|
| **Endpoint** | `http://localhost:8000/sse` |
| **Protocol** | MCP 1.0 over SSE |
| **Content-Type** | `text/event-stream` |

**Tool Registration (server → client on connect):**
```json
{
  "tools": [
    {
      "name": "analyze_network_telemetry",
      "description": "Retrieves network signaling logs for a subscriber IMSI, filters noise using a local fine-tuned SLM, and returns a clean JSON summary of failure events for Root Cause Analysis.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "imsi": {"type": "string", "description": "Subscriber IMSI identifier"},
          "context": {"type": "string", "description": "Symptom description", "default": "general_failure"}
        },
        "required": ["imsi"]
      }
    }
  ]
}
```

**Tool Call (client → server):**
```json
{
  "method": "tools/call",
  "params": {
    "name": "analyze_network_telemetry",
    "arguments": {"imsi": "999123", "context": "YouTube Buffering"}
  }
}
```

**Tool Response (server → client):**
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"events\": [{\"error_code\": \"TCP_RETRANSMISSION\", \"severity\": \"high\"}]}"
    }
  ]
}
```

### 4.2 Backend REST API

**Endpoint:** `POST /api/chat`

**Request:**
```json
{
  "message": "Subscriber 999123 complains YouTube buffers constantly"
}
```

**Response (success):**
```json
{
  "response": "Based on the network analysis, the subscriber is experiencing...",
  "metrics": {
    "raw_log_bytes": 45230,
    "raw_log_tokens_est": 11308,
    "filtered_output_bytes": 187,
    "filtered_output_tokens_est": 47,
    "token_reduction_pct": 99.6,
    "estimated_cost_savings_usd": 0.0158,
    "slm_inference_latency_sec": 6.2,
    "total_response_time_sec": 12.4
  }
}
```

**Response (error):**
```json
{
  "response": "I'm unable to analyze the network at this time. The edge inference service is unavailable.",
  "metrics": null,
  "error": "MCP_CONNECTION_FAILED"
}
```

**Error Codes:**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `BEDROCK_API_ERROR` | 502 | Frontier model call failed |
| `MCP_CONNECTION_FAILED` | 503 | Cannot reach MCP server |
| `SLM_INFERENCE_TIMEOUT` | 504 | Local model inference exceeded 30s |
| `INVALID_REQUEST` | 400 | Missing or malformed message field |

### 4.3 Frontend ↔ Backend Communication

```
┌──────────┐   POST /api/chat    ┌──────────────┐   Bedrock API    ┌─────────┐
│ Frontend │ ──────────────────► │ Python       │ ───────────────► │ Bedrock │
│ (Next.js)│ ◄────────────────── │ Backend      │ ◄─────────────── │         │
│          │   JSON response     │ (Strands)    │                  └─────────┘
└──────────┘                     │              │   MCP SSE         ┌─────────┐
                                 │              │ ───────────────► │ MCP     │
                                 │              │ ◄─────────────── │ Server  │
                                 └──────────────┘                  └─────────┘
```

---

## 5. Component Functional Specifications

### 5.1 C1: Fine-Tuning Pipeline

#### 5.1.1 `fine_tuning/train.py` — Functional Behavior

**Inputs:**
- `data/training_data.json` (1,300 examples)
- Configuration: QLoRA params, training hyperparams (defaults in code, overridable via CLI args)

**Processing Steps:**

| Step | Function | Validation |
|------|----------|------------|
| 1. Load dataset | Read JSON, parse into HF Dataset | Assert count = 1,300; assert both positive and negative examples present |
| 2. Load base model | Load Qwen3-14B in 4-bit NF4 | Assert model loads; log VRAM usage |
| 3. Attach QLoRA | Apply LoRA config to target modules | Assert adapter params < 1% of total params |
| 4. Train | SFTTrainer for 1 epoch | Log loss per step; assert final loss < 0.30 |
| 5. Save adapter | Save to `output/adapter/` | Assert files exist: adapter_model.safetensors, adapter_config.json |

**Outputs:**
- LoRA adapter files at `output/adapter/`
- Training log to stdout (loss, accuracy, duration)

#### 5.1.2 `fine_tuning/merge_adapter.py` — Functional Behavior

**Inputs:** Base model (HuggingFace Hub), adapter at `output/adapter/`

**Processing:**
1. Load base model in FP16
2. Load and merge LoRA adapter
3. Save merged model to `output/merged_model/`

**Outputs:** Full merged model in HuggingFace format (~28 GB)

#### 5.1.3 `fine_tuning/convert_to_gguf.sh` — Functional Behavior

**Inputs:** Merged model at `output/merged_model/`

**Processing:**
1. Verify llama.cpp build exists (check for `quantize` binary)
2. Run `convert_hf_to_gguf.py` → `output/model-fp16.gguf`
3. Run `quantize` with Q4_K_M → `output/qwen3-14b-telco-Q4_K_M.gguf`
4. Delete intermediate FP16 GGUF to save disk space

**Outputs:** `output/qwen3-14b-telco-Q4_K_M.gguf` (~9 GB)

#### 5.1.4 `fine_tuning/validate.py` — Functional Behavior

**Inputs:** GGUF model, `data/test_data.json` (1,000 examples)

**Processing:**
1. Load model with GBNF grammar
2. For each test example:
   a. Run inference with grammar constraints
   b. Apply post-processing filter
   c. Parse ground truth from test data
   d. Compare: compute TP, FP, FN per example
3. Aggregate metrics across all examples
4. Group by scenario for breakdown

**Outputs (stdout):**
```
=== Model Validation Report ===
Total examples: 1000
F1 Score:       0.915 (raw) / 0.945 (filtered)
Precision:      0.879 (raw) / 0.975 (filtered)
Recall:         0.953 (raw) / 0.917 (filtered)
Exact Match:    62.5% (raw) / 77.3% (filtered)
Avg Latency:    6.45s

Per-Scenario Breakdown:
| Scenario              | Examples | TP   | FP  | FN  | Status    |
| core_network_failure  | 180      | 180  | 0   | 0   | ✅ PERFECT |
| authentication_failure| 100      | 100  | 0   | 0   | ✅ PERFECT |
...

Result: PASS (F1 0.945 ≥ 0.90 threshold)
```

### 5.2 C2: Smart MCP Server

#### 5.2.1 `mcp_server/inference_engine.py` — Functional Behavior

**Class: `TelcoInferenceEngine`**

**Constructor:**
- Input: `model_path` (path to GGUF file), `grammar_path` (path to .gbnf file)
- Loads model into GPU via llama-cpp-python
- Parses GBNF grammar via `LlamaGrammar.from_file()`
- Logs VRAM usage and model metadata

**Method: `analyze(raw_logs: str) -> dict`**
- Constructs prompt using system prompt + user logs in ChatML format
- Calls `llm()` with grammar constraint, temperature=0.1, max_tokens=512, repeat_penalty=1.2
- Extracts generated text
- Parses JSON (guaranteed valid by grammar)
- Returns `{"events": [...], "raw_log_bytes": N, "inference_latency_sec": T}`

**Error handling:**
- Inference timeout (>30s): returns `{"events": [], "status": "error", "message": "Inference timeout"}`
- Any exception: logs full traceback, returns error JSON

#### 5.2.2 `mcp_server/post_processing_filter.py` — Functional Behavior

**Function: `apply_filter(events: list[dict]) -> tuple[list[dict], list[str]]`**

Returns: (filtered_events, applied_rules)

**Rule execution order:**

```
Input events
    │
    ▼
Rule 1: UPF Cascade ──── if "UPF_DEGRADATION" in codes:
    │                      remove RRC_SETUP_FAILURE, RRC_REESTABLISHMENT_FAILURE,
    │                      5GMM_CAUSE_CONGESTION
    ▼
Rule 2: Core Cascade ─── if "5GMM_CAUSE_NETWORK_FAILURE" or "PDU_SESSION_FAILURE" in codes:
    │                      remove all RRC-prefixed codes
    ▼
Rule 3: Auth Isolation ── if "AUTHENTICATION_FAILURE" in codes:
    │                      keep ONLY AUTHENTICATION_FAILURE + REGISTRATION_REJECTED
    ▼
Rule 4: Deduplication ── remove duplicate error_code values (keep first)
    │
    ▼
Output events
```

**Example:**
```
Input:  [{"error_code": "UPF_DEGRADATION", "severity": "high"},
         {"error_code": "RRC_SETUP_FAILURE", "severity": "high"},
         {"error_code": "5GMM_CAUSE_CONGESTION", "severity": "medium"}]

Rule 1 fires: removes RRC_SETUP_FAILURE and 5GMM_CAUSE_CONGESTION

Output: [{"error_code": "UPF_DEGRADATION", "severity": "high"}]
Rules:  ["UPF_CASCADE"]
```

#### 5.2.3 `mcp_server/server.py` — Functional Behavior

**Startup sequence:**
1. Read config: model path (env `MODEL_PATH`), port (env `MCP_PORT`, default 8000)
2. Initialize `TelcoInferenceEngine`
3. Register MCP tool `analyze_network_telemetry`
4. Start FastMCP with SSE transport
5. Log "Server ready on port {port}"

**Tool handler: `analyze_network_telemetry(imsi, context)`**
1. Fetch synthetic raw logs for IMSI (from local data store)
2. Measure raw log size in bytes
3. Call `inference_engine.analyze(raw_logs)`
4. Call `post_processing_filter.apply_filter(events)`
5. Build response JSON with events + metadata
6. Return response string

#### 5.2.4 `mcp_server/telco_grammar.gbnf` — Grammar Definition

```
root     ::= object
object   ::= "{" ws "\"events\"" ws ":" ws array ws "}"
array    ::= "[" ws (event-obj (ws "," ws event-obj)*)? ws "]"
event-obj ::= "{" ws "\"error_code\"" ws ":" ws string ws "," ws "\"severity\"" ws ":" ws severity-val ws "}"
string   ::= "\"" ([^"\\] | "\\" .)* "\""
severity-val ::= "\"high\"" | "\"medium\"" | "\"low\""
ws       ::= [ \t\n]*
```

Note: `severity-val` is constrained to exactly three valid values, further reducing invalid outputs.


### 5.3 C3: Web GUI with Strands Agent

#### 5.3.1 `agent/agent.py` — Functional Behavior

**Class: `TelcoRCAAgent`**

**Constructor:**
- Reads `BEDROCK_MODEL_ID` from env (default: Claude 4.5 Haiku)
- Reads `MCP_SERVER_URL` from env (default: `http://localhost:8000/sse`)
- Initializes Strands Agent with system prompt (telco RCA persona)
- Connects to MCP server, discovers tools
- Logs available tools and model ID

**Method: `process_query(message: str) -> dict`**
1. Records start time
2. Passes message to Strands agent
3. Agent invokes Bedrock with message + tool definitions
4. If Bedrock requests tool call → Strands forwards to MCP server
5. Captures tool call metadata (raw bytes, filtered bytes, SLM latency)
6. Bedrock synthesizes final response
7. Records end time
8. Returns `{"response": str, "metrics": dict}`

**Metrics calculation:**
```python
metrics = {
    "raw_log_bytes": raw_size,                          # from MCP server metadata
    "raw_log_tokens_est": raw_size // 4,                # approximation
    "filtered_output_bytes": len(filtered_json),
    "filtered_output_tokens_est": len(filtered_json) // 4,
    "token_reduction_pct": round((1 - filtered/raw) * 100, 1),
    "estimated_cost_savings_usd": round((raw_tokens - filtered_tokens) * price_per_token, 4),
    "slm_inference_latency_sec": slm_latency,           # from MCP server metadata
    "total_response_time_sec": end_time - start_time
}
```

#### 5.3.2 `agent/api_server.py` — Functional Behavior

**Framework:** FastAPI

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Process a user query through the agent |
| GET | `/api/health` | Health check (returns agent + MCP status) |

**POST /api/chat flow:**
1. Validate request body (must contain `message` string, 1-2000 chars)
2. Call `TelcoRCAAgent.process_query(message)`
3. Return JSON response with `response` and `metrics`
4. On error: return appropriate HTTP status + error code

**GET /api/health response:**
```json
{
  "status": "healthy",
  "agent": "connected",
  "mcp_server": "connected",
  "bedrock_model": "anthropic.claude-4-5-haiku-latest-v1:0"
}
```

#### 5.3.3 Frontend Components — Functional Behavior

**`page.tsx` (Main Page):**
- Renders ChatInterface and ComparisonPanel side by side (desktop) or stacked (mobile)
- Manages state: `messages[]`, `currentMetrics`, `isLoading`

**`ChatInterface.tsx`:**
- Displays scrollable message list (user right-aligned, agent left-aligned)
- Input field with send button at bottom
- On submit: adds user message to list, sets `isLoading=true`, calls `POST /api/chat`, adds agent response, updates metrics, sets `isLoading=false`
- Shows typing indicator while `isLoading`
- Renders agent responses with markdown (bold, code blocks, lists)

**`ComparisonPanel.tsx`:**
- Displays metrics from most recent tool invocation
- Shows "No data yet" when no tool call has been made
- Visual elements:
  - Bar chart or gauge: raw tokens vs filtered tokens
  - Large percentage: "99.6% Token Reduction"
  - Cost savings in USD
  - Inference latency in seconds

**`MessageBubble.tsx`:**
- Renders a single message with role-based styling
- User messages: right-aligned, blue background
- Agent messages: left-aligned, gray background, markdown rendered

---

## 6. Configuration Management

### 6.1 Environment Variables

| Variable | Component | Default | Description |
|----------|-----------|---------|-------------|
| `MODEL_PATH` | MCP Server | `output/qwen3-14b-telco-Q4_K_M.gguf` | Path to GGUF model |
| `MCP_PORT` | MCP Server | `8000` | SSE server port |
| `MCP_SERVER_URL` | Agent Backend | `http://localhost:8000/sse` | MCP server SSE endpoint |
| `BEDROCK_MODEL_ID` | Agent Backend | `anthropic.claude-4-5-haiku-latest-v1:0` | Bedrock model identifier |
| `AWS_REGION` | Agent Backend | `us-east-1` | AWS region for Bedrock |
| `API_PORT` | Agent Backend | `8080` | REST API server port |
| `BACKEND_URL` | Frontend | `http://localhost:8080` | Python backend URL |

### 6.2 Configuration File: `agent/config.py`

```python
import os

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-4-5-haiku-latest-v1:0")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
API_PORT = int(os.getenv("API_PORT", "8080"))

# Token pricing for cost estimation (per 1K input tokens)
TOKEN_PRICES = {
    "anthropic.claude-4-5-haiku": 0.001,
    "amazon.nova-2-pro": 0.0008,
    "anthropic.claude-4-5-sonnet": 0.003,
    "default": 0.001
}
```

---

## 7. State & Lifecycle

### 7.1 System Startup Sequence

```
1. Start MCP Server (port 8000)
   └─ Load GGUF model → Load grammar → Register tool → Listen SSE
       └─ Status: READY (~60s)

2. Start Agent Backend (port 8080)
   └─ Connect to MCP server → Discover tools → Verify Bedrock access
       └─ Status: READY (~5s)

3. Start Frontend (port 3000)
   └─ Next.js dev server → Serve UI
       └─ Status: READY (~10s)
```

**Dependency order:** MCP Server must be running before Agent Backend starts. Frontend can start independently.

### 7.2 Request Lifecycle

```
                    Frontend          Backend           MCP Server        Bedrock
                       │                 │                  │                │
User types message ──► │                 │                  │                │
                       │─POST /api/chat─►│                  │                │
                       │                 │──invoke agent───►│                │
                       │                 │                  │                │
                       │                 │                  │──InvokeModel──►│
                       │                 │                  │                │
                       │                 │                  │◄─tool_use──────│
                       │                 │                  │                │
                       │                 │◄─tools/call──────│                │
                       │                 │                  │                │
                       │                 │  [SLM inference + filter]         │
                       │                 │                  │                │
                       │                 │──tool_result────►│                │
                       │                 │                  │──InvokeModel──►│
                       │                 │                  │◄─response──────│
                       │                 │◄─agent response──│                │
                       │◄─JSON response──│                  │                │
Display response ◄─────│                 │                  │                │
Update metrics ◄───────│                 │                  │                │
```

---

## 8. Error Handling Matrix

| Error Condition | Component | Detection | Response | User Impact |
|----------------|-----------|-----------|----------|-------------|
| Training dataset missing | C1 train.py | FileNotFoundError on load | Exit with path suggestion | Participant fixes path |
| CUDA OOM during training | C1 train.py | RuntimeError from PyTorch | Log VRAM usage, suggest lower seq_length | Participant adjusts config |
| GGUF model file missing | C2 server.py | FileNotFoundError on init | Exit with expected path | Participant runs conversion |
| GBNF grammar file missing | C2 server.py | FileNotFoundError on init | Exit with expected path | Participant checks file |
| GPU not detected | C2 inference | llama_cpp device check | Fall back to CPU, log warning | Slower inference (~30s) |
| SLM inference timeout | C2 inference | Timer > 30s | Return error JSON | Agent reports edge unavailable |
| Invalid JSON from SLM | C2 inference | Cannot happen (grammar) | N/A | N/A |
| MCP server unreachable | C3 agent | Connection refused | Return error to frontend | "Edge service unavailable" |
| Bedrock API throttled | C3 agent | ThrottlingException | Retry with backoff (3 attempts) | Slight delay |
| Bedrock API auth failure | C3 agent | AccessDeniedException | Return error to frontend | "Check AWS credentials" |
| Invalid user input | C3 api_server | Pydantic validation | 400 with error detail | "Please enter a valid message" |
| Frontend cannot reach backend | C3 frontend | fetch() network error | Display connection error banner | "Backend unavailable" |

---

## 9. Traceability Matrix

| Requirement (Section) | Functional Spec Coverage |
|-----------------------|-------------------------|
| REQ 5.1 Fine-tune objective | UC-1, §5.1.1 |
| REQ 5.3 QLoRA configuration | §5.1.1 Step 3, §6.1 env vars |
| REQ 5.4 Training dataset | §3.1, UC-1 Step 1 |
| REQ 5.6 Model conversion pipeline | UC-2, §5.1.2, §5.1.3 |
| REQ 6.2 MCP server spec | UC-4, §4.1, §5.2.3 |
| REQ 6.3 Tool definition | §4.1 tool registration |
| REQ 6.4 GBNF grammar | §5.2.4, §5.2.1 |
| REQ 6.5 Post-processing filter | §5.2.2 (all 4 rules) |
| REQ 7.2 Frontend spec | §5.3.3 |
| REQ 7.3 Backend + Strands | §5.3.1, §5.3.2, §4.2 |
| REQ 7.4 Bedrock configuration | §6.1, §6.2 |
| REQ 8.1 Automated tests | UC-3, §5.1.4 |
| REQ 8.2 Manual demo | UC-5 (4 scenarios) |
| REQ 9.1 Performance targets | §7.2 (latency in flow) |
| REQ 9.2 Reliability | §8 error handling matrix |
| REQ 10 Project structure | All §5 file references |
| REQ 13 Acceptance criteria | UC postconditions + §5.1.4 thresholds |

---

*End of Functional Specification*
