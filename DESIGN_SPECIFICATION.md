# Design Specification: Telco Autonomous Operations Workshop

**Document Version:** 1.0
**Date:** 2026-03-09
**Classification:** Design Specification
**Author:** Angelo Sampietro
**Source:** REQUIREMENTS_DOCUMENT.md v1.0, FUNCTIONAL_SPECIFICATION.md v1.0

---

## 1. Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python backend framework | FastAPI (async) | Native async for Bedrock streaming, auto OpenAPI docs, Pydantic validation |
| Frontend-backend communication | Next.js API route proxy | Single origin avoids CORS; frontend port 3000 proxies to Python port 8080 |
| Strands MCP integration | Native MCP tool loader | Auto-discovers tools from SSE endpoint; zero manual tool wiring |
| Model loading strategy | Singleton at startup | 60s load once; all requests share the loaded model instance |
| Synthetic log storage | JSON file (`data/synthetic_logs.json`) | Easy to inspect, modify, extend; loaded once at startup |
| Frontend state management | React Context API | Shared state between ChatInterface and ComparisonPanel without external deps |
| Comparison panel layout | Right sidebar (desktop), below chat (mobile) | Standard dashboard pattern; responsive via Tailwind breakpoints |
| Post-processing filter | Separate module, called by server.py | Loosely coupled; independently unit-testable; clear pipeline stages |
| Logging | Human-readable stdout + JSON to file | Stdout for workshop readability; file for automated analysis |
| Model path resolution | Absolute path via `MODEL_PATH` env var | Explicit, no ambiguity; required for production portability |

---

## 2. System Architecture Design

### 2.1 Process Architecture

```
EC2 Instance (g6.12xlarge)
│
├── Process 1: MCP Server (Python, port 8000)
│   ├── FastMCP SSE transport
│   ├── TelcoInferenceEngine (llama-cpp-python, GPU)
│   └── PostProcessingFilter
│
├── Process 2: Agent Backend (Python/FastAPI, port 8080)
│   ├── Strands Agent (MCP client → port 8000)
│   └── Bedrock client (HTTPS → AWS)
│
└── Process 3: Frontend (Node.js/Next.js, port 3000)
    ├── React UI
    └── API proxy route → port 8080
```

### 2.2 Network Topology

```
Browser ──► :3000 (Next.js)
                │
                │ /api/chat proxy
                ▼
            :8080 (FastAPI)
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
    :8000 (MCP)    Bedrock API
    (localhost)    (HTTPS, us-east-1)
```

All inter-process communication is localhost. Only Bedrock calls leave the instance.

---

## 3. Component Design: C1 — Fine-Tuning Pipeline

### 3.1 Module Design

```
fine_tuning/
├── train.py              # Entry point: orchestrates full training
├── merge_adapter.py      # Entry point: merges LoRA into base
├── convert_to_gguf.sh    # Entry point: HF → GGUF → Q4_K_M
└── validate.py           # Entry point: benchmarks on test set
```

All scripts are standalone entry points (no shared library). Each reads config from CLI args with sensible defaults.

### 3.2 train.py — Detailed Design

```python
# Pseudocode structure

def main():
    args = parse_args()  # --dataset, --output-dir, --rank, --alpha, --epochs, --lr
    
    # 1. Load & validate dataset
    dataset = load_dataset("json", data_files=args.dataset)
    assert len(dataset["train"]) == 1300
    
    # 2. Load base model (4-bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    
    # 3. Attach QLoRA
    lora_config = LoraConfig(
        r=args.rank,           # default 16
        lora_alpha=args.alpha, # default 32
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    log_trainable_params(model)  # expect < 1% of total
    
    # 4. Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,  # default 1
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,         # default 1e-4
            lr_scheduler_type="cosine",
            warmup_steps=20,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            max_seq_length=2048,
        )
    )
    trainer.train()
    
    # 5. Save
    model.save_pretrained(args.output_dir + "/adapter")
    tokenizer.save_pretrained(args.output_dir + "/adapter")
    print_summary(trainer.state.log_history)
```

**CLI interface:**
```bash
python fine_tuning/train.py \
  --dataset data/training_data.json \
  --output-dir output \
  --rank 16 --alpha 32 --epochs 1 --lr 1e-4
```

### 3.3 merge_adapter.py — Detailed Design

```python
def main():
    args = parse_args()  # --base-model, --adapter-path, --output-dir
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    
    model.save_pretrained(args.output_dir + "/merged_model")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.save_pretrained(args.output_dir + "/merged_model")
```

### 3.4 convert_to_gguf.sh — Detailed Design

```bash
#!/bin/bash
set -euo pipefail

MERGED_DIR="${1:-output/merged_model}"
OUTPUT_DIR="${2:-output}"
LLAMACPP_DIR="${3:-llama.cpp}"

echo "Step 1: Converting HF → GGUF (FP16)..."
python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$OUTPUT_DIR/model-fp16.gguf" --outtype f16

echo "Step 2: Quantizing to Q4_K_M..."
"$LLAMACPP_DIR/build/bin/llama-quantize" \
  "$OUTPUT_DIR/model-fp16.gguf" \
  "$OUTPUT_DIR/qwen3-14b-telco-Q4_K_M.gguf" Q4_K_M

echo "Step 3: Cleaning up FP16 intermediate..."
rm "$OUTPUT_DIR/model-fp16.gguf"

echo "Done: $OUTPUT_DIR/qwen3-14b-telco-Q4_K_M.gguf"
ls -lh "$OUTPUT_DIR/qwen3-14b-telco-Q4_K_M.gguf"
```

### 3.5 validate.py — Detailed Design

```python
def main():
    args = parse_args()  # --model-path, --grammar-path, --test-data, --with-filter

    engine = TelcoInferenceEngine(args.model_path, args.grammar_path)
    filter_fn = apply_filter if args.with_filter else lambda e: (e, [])
    test_data = json.load(open(args.test_data))

    results = {"tp": 0, "fp": 0, "fn": 0, "exact_match": 0, "latencies": []}
    per_scenario = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})

    for example in tqdm(test_data):
        raw_logs = example["messages"][1]["content"]
        ground_truth = parse_ground_truth(example["messages"][2]["content"])

        result = engine.analyze(raw_logs)
        filtered, _ = filter_fn(result["events"])
        predicted_codes = {e["error_code"] for e in filtered}
        expected_codes = {e["error_code"] for e in ground_truth}

        tp = len(predicted_codes & expected_codes)
        fp = len(predicted_codes - expected_codes)
        fn = len(expected_codes - predicted_codes)

        results["tp"] += tp; results["fp"] += fp; results["fn"] += fn
        results["latencies"].append(result["inference_latency_sec"])
        if predicted_codes == expected_codes:
            results["exact_match"] += 1

        scenario = example.get("scenario", "unknown")
        per_scenario[scenario]["tp"] += tp
        per_scenario[scenario]["fp"] += fp
        per_scenario[scenario]["fn"] += fn

    print_report(results, per_scenario, len(test_data))
```


---

## 4. Component Design: C2 — Smart MCP Server

### 4.1 Module Design

```
mcp_server/
├── server.py                  # FastMCP entry point + tool handler
├── inference_engine.py        # Singleton llama.cpp wrapper
├── post_processing_filter.py  # Deterministic filter (pure functions)
├── log_store.py               # Loads synthetic logs from JSON
└── telco_grammar.gbnf         # GBNF grammar file
```

### 4.2 Class Diagram

```
┌─────────────────────────┐
│     FastMCP Server       │
│  (server.py)             │
│                          │
│  @mcp.tool()             │
│  analyze_network_        │
│    telemetry(imsi,ctx)   │
└────────┬────────────────┘
         │ uses
    ┌────▼────┐    ┌──────────────┐    ┌──────────────┐
    │LogStore  │    │InferenceEngine│    │PostProcessing│
    │          │    │  (singleton)  │    │Filter        │
    │load()    │    │  analyze()   │    │apply_filter()│
    │get_logs()│    │              │    │              │
    └──────────┘    └──────────────┘    └──────────────┘
```

### 4.3 inference_engine.py — Detailed Design

```python
import time
import logging
from llama_cpp import Llama, LlamaGrammar

logger = logging.getLogger("InferenceEngine")

SYSTEM_PROMPT = """You are a 3GPP Protocol Expert specializing in Root Cause Analysis.
Analyze the logs provided.
1. Ignore routine Heartbeats and KeepAlives.
2. Filter out sympathetic noise (consequential failures).
3. Identify the primary causal chain (NAS, NGAP, or RRC).
4. Output ONLY valid JSON according to the schema."""

class TelcoInferenceEngine:
    """Singleton wrapper around llama-cpp-python with GBNF grammar."""

    _instance = None

    @classmethod
    def get_instance(cls, model_path: str, grammar_path: str) -> "TelcoInferenceEngine":
        if cls._instance is None:
            cls._instance = cls(model_path, grammar_path)
        return cls._instance

    def __init__(self, model_path: str, grammar_path: str):
        logger.info(f"Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=16384,
            n_gpu_layers=-1,  # offload all layers to GPU
            verbose=False
        )
        self.grammar = LlamaGrammar.from_file(grammar_path)
        logger.info("Model loaded, grammar initialized")

    def analyze(self, raw_logs: str) -> dict:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{raw_logs}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        raw_bytes = len(raw_logs.encode("utf-8"))
        start = time.time()

        try:
            output = self.llm(
                prompt,
                grammar=self.grammar,
                max_tokens=512,
                temperature=0.1,
                repeat_penalty=1.2,
                top_p=0.9,
                stop=["<|im_end|>"]
            )
            text = output["choices"][0]["text"].strip()
            latency = time.time() - start
            events = json.loads(text)  # guaranteed valid by grammar
            return {
                "events": events.get("events", []),
                "raw_log_bytes": raw_bytes,
                "inference_latency_sec": round(latency, 2)
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {
                "events": [],
                "status": "error",
                "message": str(e),
                "raw_log_bytes": raw_bytes,
                "inference_latency_sec": round(time.time() - start, 2)
            }
```

### 4.4 post_processing_filter.py — Detailed Design

```python
import logging

logger = logging.getLogger("PostProcessingFilter")

# Sympathetic noise codes per cascade type
UPF_SYMPATHETIC = {"RRC_SETUP_FAILURE", "RRC_REESTABLISHMENT_FAILURE", "5GMM_CAUSE_CONGESTION"}
CORE_SYMPATHETIC_PREFIX = "RRC_"
AUTH_KEEP = {"AUTHENTICATION_FAILURE", "REGISTRATION_REJECTED"}

def apply_filter(events: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Applies deterministic post-processing rules to remove sympathetic noise.
    Returns (filtered_events, list_of_applied_rule_names).
    """
    codes = {e["error_code"] for e in events}
    applied_rules = []
    filtered = list(events)  # shallow copy

    # Rule 1: UPF Degradation Cascade
    if "UPF_DEGRADATION" in codes:
        before = len(filtered)
        filtered = [e for e in filtered if e["error_code"] not in UPF_SYMPATHETIC]
        if len(filtered) < before:
            applied_rules.append("UPF_CASCADE")

    # Rule 2: Core Network Failure Cascade
    codes_after_r1 = {e["error_code"] for e in filtered}
    if "5GMM_CAUSE_NETWORK_FAILURE" in codes_after_r1 or "PDU_SESSION_FAILURE" in codes_after_r1:
        before = len(filtered)
        filtered = [e for e in filtered if not e["error_code"].startswith(CORE_SYMPATHETIC_PREFIX)]
        if len(filtered) < before:
            applied_rules.append("CORE_CASCADE")

    # Rule 3: Authentication Failure Isolation
    codes_after_r2 = {e["error_code"] for e in filtered}
    if "AUTHENTICATION_FAILURE" in codes_after_r2:
        before = len(filtered)
        filtered = [e for e in filtered if e["error_code"] in AUTH_KEEP]
        if len(filtered) < before:
            applied_rules.append("AUTH_ISOLATION")

    # Rule 4: Deduplication
    seen = set()
    deduped = []
    for e in filtered:
        if e["error_code"] not in seen:
            seen.add(e["error_code"])
            deduped.append(e)
    if len(deduped) < len(filtered):
        applied_rules.append("DEDUP")
    filtered = deduped

    logger.info(f"Filter applied: {applied_rules}, events {len(events)} → {len(filtered)}")
    return filtered, applied_rules
```

### 4.5 log_store.py — Detailed Design

```python
import json
import random
import logging

logger = logging.getLogger("LogStore")

class LogStore:
    """Loads synthetic logs from JSON file, serves by IMSI prefix."""

    def __init__(self, path: str = "data/synthetic_logs.json"):
        with open(path) as f:
            self.scenarios = json.load(f)  # {"transport_jitter": [...logs...], ...}
        self.imsi_map = {
            "999": "transport_jitter",
            "888": "radio_failure",
            "777": "normal",
            "666": "authentication_failure",
            "555": "core_network_failure",
            "444": "handover_failure",
            "333": "congestion",
            "222": "qos_violation",
        }
        logger.info(f"Loaded {sum(len(v) for v in self.scenarios.values())} log entries")

    def get_logs(self, imsi: str) -> str:
        prefix = imsi[:3]
        scenario = self.imsi_map.get(prefix)
        if scenario is None:
            scenario = random.choice(list(self.scenarios.keys()))
        logs = self.scenarios[scenario]
        if isinstance(logs, list):
            return random.choice(logs)
        return logs
```

### 4.6 server.py — Detailed Design

```python
import os, logging, json
from mcp.server.fastmcp import FastMCP
from inference_engine import TelcoInferenceEngine
from post_processing_filter import apply_filter
from log_store import LogStore

# Logging: human-readable to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
# JSON to file
file_handler = logging.FileHandler("mcp_server.log")
file_handler.setFormatter(logging.Formatter('{"ts":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("MCPServer")

MODEL_PATH = os.environ["MODEL_PATH"]  # required, absolute path
GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "telco_grammar.gbnf")
PORT = int(os.getenv("MCP_PORT", "8000"))

engine = TelcoInferenceEngine.get_instance(MODEL_PATH, GRAMMAR_PATH)
log_store = LogStore()
mcp = FastMCP("TelcoEdgeFilter")

@mcp.tool()
def analyze_network_telemetry(imsi: str, context: str = "general_failure") -> str:
    """Retrieves network signaling logs for a subscriber IMSI, filters noise
    using a local fine-tuned SLM, and returns a clean JSON summary of failure
    events for Root Cause Analysis."""
    raw_logs = log_store.get_logs(imsi)
    result = engine.analyze(raw_logs)
    filtered_events, rules = apply_filter(result["events"])

    response = {
        "events": filtered_events,
        "metadata": {
            "raw_log_bytes": result["raw_log_bytes"],
            "filtered_output_bytes": len(json.dumps({"events": filtered_events})),
            "inference_latency_sec": result["inference_latency_sec"],
            "filter_rules_applied": rules
        }
    }
    return json.dumps(response)

if __name__ == "__main__":
    logger.info(f"Starting MCP Server on port {PORT}")
    mcp.run(transport="sse")
```

### 4.7 telco_grammar.gbnf

```
root        ::= object
object      ::= "{" ws "\"events\"" ws ":" ws array ws "}"
array       ::= "[" ws (event-obj (ws "," ws event-obj)*)? ws "]"
event-obj   ::= "{" ws "\"error_code\"" ws ":" ws string ws "," ws "\"severity\"" ws ":" ws severity-val ws "}"
string      ::= "\"" ([^"\\] | "\\" .)* "\""
severity-val ::= "\"high\"" | "\"medium\"" | "\"low\""
ws          ::= [ \t\n]*
```


---

## 5. Component Design: C3 — Web GUI with Strands Agent

### 5.1 Module Design

```
agent/
├── agent.py          # TelcoRCAAgent class (Strands + MCP + Bedrock)
├── api_server.py     # FastAPI app with /api/chat and /api/health
└── config.py         # Centralized configuration from env vars

frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # Root layout with providers
│   │   ├── page.tsx            # Main page composing Chat + Panel
│   │   └── api/chat/route.ts   # Next.js API proxy to Python backend
│   ├── components/
│   │   ├── ChatInterface.tsx   # Chat messages + input
│   │   ├── MessageBubble.tsx   # Single message rendering
│   │   └── ComparisonPanel.tsx # Metrics sidebar
│   └── context/
│       └── MetricsContext.tsx  # React Context for shared metrics state
├── package.json
├── tailwind.config.js
└── next.config.js
```

### 5.2 agent.py — Detailed Design

```python
import time, json, logging
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from config import MODEL_ID, MCP_URL, AWS_REGION, TOKEN_PRICES

logger = logging.getLogger("TelcoAgent")

SYSTEM_PROMPT = """You are a Telco Network Operations AI Assistant specializing in Root Cause Analysis
for 5G networks. When a user reports a network issue:
1. Extract the subscriber identifier (IMSI) and symptom description from the query.
2. Use the analyze_network_telemetry tool to retrieve filtered diagnostic data.
3. Interpret the diagnostic JSON and provide a clear, actionable diagnosis.
4. Include a specific recommendation (e.g., dispatch field tech, escalate to core team).
Always explain your reasoning in plain language suitable for a NOC operator."""

class TelcoRCAAgent:
    def __init__(self):
        self.model = BedrockModel(model_id=MODEL_ID, region_name=AWS_REGION)
        self.mcp_client = MCPClient(url=MCP_URL, transport="sse")
        self.mcp_client.start()
        tools = self.mcp_client.list_tools()
        logger.info(f"MCP tools discovered: {[t.name for t in tools]}")

        self.agent = Agent(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools
        )
        # Resolve token price
        self.price_per_1k = TOKEN_PRICES.get(MODEL_ID.split(":")[0], TOKEN_PRICES["default"])

    def process_query(self, message: str) -> dict:
        start = time.time()
        response = self.agent(message)
        total_time = time.time() - start

        # Extract metrics from tool call results if available
        metrics = self._extract_metrics(response, total_time)
        return {"response": str(response), "metrics": metrics}

    def _extract_metrics(self, response, total_time: float) -> dict | None:
        # Parse metadata from MCP tool response embedded in agent trace
        try:
            tool_results = response.tool_results  # Strands API
            for result in tool_results:
                data = json.loads(result.content)
                if "metadata" in data:
                    meta = data["metadata"]
                    raw_tokens = meta["raw_log_bytes"] // 4
                    filtered_tokens = meta["filtered_output_bytes"] // 4
                    return {
                        "raw_log_bytes": meta["raw_log_bytes"],
                        "raw_log_tokens_est": raw_tokens,
                        "filtered_output_bytes": meta["filtered_output_bytes"],
                        "filtered_output_tokens_est": filtered_tokens,
                        "token_reduction_pct": round((1 - filtered_tokens / max(raw_tokens, 1)) * 100, 1),
                        "estimated_cost_savings_usd": round((raw_tokens - filtered_tokens) / 1000 * self.price_per_1k, 4),
                        "slm_inference_latency_sec": meta["inference_latency_sec"],
                        "total_response_time_sec": round(total_time, 2)
                    }
        except Exception:
            pass
        return None

    def shutdown(self):
        self.mcp_client.stop()
```

### 5.3 api_server.py — Detailed Design

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import TelcoRCAAgent
from config import API_PORT
import uvicorn, logging

logger = logging.getLogger("APIServer")
app = FastAPI(title="Telco RCA Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = TelcoRCAAgent()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    response: str
    metrics: dict | None = None
    error: str | None = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = agent.process_query(req.message)
        return ChatResponse(response=result["response"], metrics=result["metrics"])
    except ConnectionError:
        raise HTTPException(503, detail="MCP_CONNECTION_FAILED")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(502, detail="BEDROCK_API_ERROR")

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "agent": "connected",
        "mcp_server": "connected",
        "bedrock_model": agent.model.model_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
```

### 5.4 config.py — Detailed Design

```python
import os

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-4-5-haiku-latest-v1:0")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
API_PORT = int(os.getenv("API_PORT", "8080"))

TOKEN_PRICES = {  # per 1K input tokens
    "anthropic.claude-4-5-haiku": 0.001,
    "amazon.nova-2-pro": 0.0008,
    "anthropic.claude-4-5-sonnet": 0.003,
    "default": 0.001
}
```

### 5.5 Frontend Design

#### 5.5.1 MetricsContext.tsx

```typescript
"use client";
import { createContext, useContext, useState, ReactNode } from "react";

interface Metrics {
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

const MetricsContext = createContext<MetricsContextType>({
  metrics: null, setMetrics: () => {}
});

export function MetricsProvider({ children }: { children: ReactNode }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  return (
    <MetricsContext.Provider value={{ metrics, setMetrics }}>
      {children}
    </MetricsContext.Provider>
  );
}

export const useMetrics = () => useContext(MetricsContext);
```

#### 5.5.2 page.tsx — Layout

```typescript
"use client";
import { MetricsProvider } from "@/context/MetricsContext";
import ChatInterface from "@/components/ChatInterface";
import ComparisonPanel from "@/components/ComparisonPanel";

export default function Home() {
  return (
    <MetricsProvider>
      <div className="flex h-screen">
        {/* Chat: takes 2/3 on desktop, full on mobile */}
        <div className="flex-1 lg:w-2/3">
          <ChatInterface />
        </div>
        {/* Sidebar: 1/3 on desktop, below on mobile */}
        <div className="hidden lg:block lg:w-1/3 border-l">
          <ComparisonPanel />
        </div>
      </div>
      {/* Mobile: panel below */}
      <div className="lg:hidden border-t">
        <ComparisonPanel />
      </div>
    </MetricsProvider>
  );
}
```

#### 5.5.3 api/chat/route.ts — Proxy

```typescript
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await fetch(`${BACKEND_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
```

#### 5.5.4 ChatInterface.tsx — Key Behavior

```
State: messages[] = [{role: "user"|"agent", content: string}], isLoading: boolean

On submit(text):
  1. Append {role: "user", content: text} to messages
  2. Set isLoading = true
  3. POST /api/chat {message: text}
  4. On success:
     - Append {role: "agent", content: data.response}
     - Call setMetrics(data.metrics) via context
  5. On error:
     - Append {role: "agent", content: "Error: ..."}
  6. Set isLoading = false
  7. Scroll to bottom
```

#### 5.5.5 ComparisonPanel.tsx — Key Behavior

```
Reads metrics from MetricsContext.

If metrics == null:
  Show "Run a query to see comparison metrics"

If metrics available:
  ┌─────────────────────────────┐
  │  📊 Semantic Compression     │
  │                              │
  │  Raw Logs:    45,230 bytes   │
  │  Filtered:       187 bytes   │
  │                              │
  │  ████████████████░  99.6%    │  ← progress bar
  │  Token Reduction             │
  │                              │
  │  💰 $0.0158 saved/query     │
  │  ⏱️  6.2s SLM inference      │
  │  ⏱️  12.4s total response    │
  └─────────────────────────────┘
```


---

## 6. Infrastructure & Setup Scripts Design

### 6.1 scripts/setup_instance.sh

```bash
#!/bin/bash
# Installs all dependencies on a fresh Deep Learning AMI (Ubuntu 22.04)
set -euo pipefail

echo "=== 1. System packages ==="
sudo apt-get update && sudo apt-get install -y cmake build-essential

echo "=== 2. Python environment ==="
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft trl bitsandbytes datasets accelerate
pip install fastapi uvicorn pydantic
pip install mcp strands-agents strands-agents-tools boto3
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

echo "=== 3. Node.js (frontend) ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
cd frontend && npm install && cd ..

echo "=== 4. Build llama.cpp (for quantize tool) ==="
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc) && cd ..

echo "=== Setup complete ==="
```

### 6.2 scripts/download_model.sh

```bash
#!/bin/bash
set -euo pipefail
source venv/bin/activate
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-14B', local_dir='models/Qwen3-14B')
print('Download complete')
"
```

### 6.3 scripts/run_all.sh

```bash
#!/bin/bash
# Starts all three services. Run from project root.
set -euo pipefail

echo "Starting MCP Server..."
source venv/bin/activate
python mcp_server/server.py &
MCP_PID=$!
sleep 65  # wait for model load

echo "Starting Agent Backend..."
python agent/api_server.py &
AGENT_PID=$!
sleep 3

echo "Starting Frontend..."
cd frontend && npm run dev &
FRONT_PID=$!

echo "All services running:"
echo "  MCP Server:    http://localhost:8000 (PID $MCP_PID)"
echo "  Agent Backend: http://localhost:8080 (PID $AGENT_PID)"
echo "  Frontend:      http://localhost:3000 (PID $FRONT_PID)"
echo ""
echo "Press Ctrl+C to stop all"
trap "kill $MCP_PID $AGENT_PID $FRONT_PID 2>/dev/null" EXIT
wait
```

---

## 7. Test Design

### 7.1 Test File Mapping

| Test File | Tests | Component |
|-----------|-------|-----------|
| `tests/test_training.py` | Dataset loading, format validation | C1 |
| `tests/test_post_processing.py` | All 4 filter rules + edge cases | C2 |
| `tests/test_mcp_server.py` | Tool registration, JSON output, grammar | C2 |
| `tests/test_agent.py` | Agent init, Bedrock call, tool binding | C3 |
| `tests/test_e2e.py` | Full pipeline: query → response + metrics | C1+C2+C3 |

### 7.2 test_post_processing.py — Test Cases

```python
def test_rule1_upf_cascade():
    events = [
        {"error_code": "UPF_DEGRADATION", "severity": "high"},
        {"error_code": "RRC_SETUP_FAILURE", "severity": "high"},
        {"error_code": "5GMM_CAUSE_CONGESTION", "severity": "medium"}
    ]
    filtered, rules = apply_filter(events)
    assert len(filtered) == 1
    assert filtered[0]["error_code"] == "UPF_DEGRADATION"
    assert "UPF_CASCADE" in rules

def test_rule2_core_cascade():
    events = [
        {"error_code": "5GMM_CAUSE_NETWORK_FAILURE", "severity": "high"},
        {"error_code": "RRC_SETUP_FAILURE", "severity": "high"},
        {"error_code": "PDU_SESSION_FAILURE", "severity": "high"}
    ]
    filtered, rules = apply_filter(events)
    assert all(not e["error_code"].startswith("RRC_") for e in filtered)
    assert "CORE_CASCADE" in rules

def test_rule3_auth_isolation():
    events = [
        {"error_code": "AUTHENTICATION_FAILURE", "severity": "high"},
        {"error_code": "REGISTRATION_REJECTED", "severity": "high"},
        {"error_code": "RRC_SETUP_FAILURE", "severity": "medium"}
    ]
    filtered, rules = apply_filter(events)
    assert len(filtered) == 2
    assert {e["error_code"] for e in filtered} == {"AUTHENTICATION_FAILURE", "REGISTRATION_REJECTED"}
    assert "AUTH_ISOLATION" in rules

def test_rule4_dedup():
    events = [
        {"error_code": "TCP_RETRANSMISSION", "severity": "high"},
        {"error_code": "TCP_RETRANSMISSION", "severity": "high"}
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
```

### 7.3 test_e2e.py — Integration Test Design

```python
# Requires all 3 services running

def test_youtube_buffering():
    resp = requests.post("http://localhost:8080/api/chat", json={
        "message": "Subscriber 999123 complains YouTube buffers constantly"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["response"]) > 50  # meaningful diagnosis
    assert data["metrics"] is not None
    assert data["metrics"]["token_reduction_pct"] > 90

def test_no_issue():
    resp = requests.post("http://localhost:8080/api/chat", json={
        "message": "Check subscriber 777789 for issues, they called about billing"
    })
    data = resp.json()
    assert "no anomal" in data["response"].lower() or "no issue" in data["response"].lower()

def test_health_endpoint():
    resp = requests.get("http://localhost:8080/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
```

---

## 8. Dependency Matrix

```
                    ┌──────────┐
                    │ train.py │
                    └────┬─────┘
                         │ produces adapter
                         ▼
                  ┌──────────────┐
                  │merge_adapter │
                  └──────┬───────┘
                         │ produces merged model
                         ▼
                  ┌──────────────┐
                  │convert_to_   │
                  │gguf.sh       │
                  └──────┬───────┘
                         │ produces GGUF
                    ┌────▼─────┐
              ┌─────│ GGUF     │─────┐
              │     │ Model    │     │
              │     └──────────┘     │
              ▼                      ▼
      ┌──────────────┐      ┌──────────────┐
      │ validate.py  │      │ MCP Server   │
      └──────────────┘      └──────┬───────┘
                                   │ exposes tool
                                   ▼
                            ┌──────────────┐
                            │ Agent Backend│
                            └──────┬───────┘
                                   │ serves API
                                   ▼
                            ┌──────────────┐
                            │ Frontend     │
                            └──────────────┘
```

---

## 9. File Inventory

| File | Language | Lines (est.) | Purpose |
|------|----------|-------------|---------|
| `fine_tuning/train.py` | Python | ~80 | QLoRA training |
| `fine_tuning/merge_adapter.py` | Python | ~25 | Adapter merge |
| `fine_tuning/convert_to_gguf.sh` | Bash | ~20 | GGUF conversion |
| `fine_tuning/validate.py` | Python | ~90 | Model benchmarking |
| `mcp_server/server.py` | Python | ~45 | MCP server entry point |
| `mcp_server/inference_engine.py` | Python | ~70 | llama.cpp wrapper |
| `mcp_server/post_processing_filter.py` | Python | ~55 | Deterministic filter |
| `mcp_server/log_store.py` | Python | ~30 | Synthetic log loader |
| `mcp_server/telco_grammar.gbnf` | GBNF | ~7 | Grammar definition |
| `agent/agent.py` | Python | ~75 | Strands agent |
| `agent/api_server.py` | Python | ~40 | FastAPI backend |
| `agent/config.py` | Python | ~15 | Configuration |
| `frontend/src/app/page.tsx` | TypeScript | ~25 | Main page |
| `frontend/src/app/api/chat/route.ts` | TypeScript | ~15 | API proxy |
| `frontend/src/context/MetricsContext.tsx` | TypeScript | ~25 | Shared state |
| `frontend/src/components/ChatInterface.tsx` | TypeScript | ~80 | Chat UI |
| `frontend/src/components/MessageBubble.tsx` | TypeScript | ~30 | Message display |
| `frontend/src/components/ComparisonPanel.tsx` | TypeScript | ~60 | Metrics sidebar |
| `scripts/setup_instance.sh` | Bash | ~25 | Instance setup |
| `scripts/download_model.sh` | Bash | ~10 | Model download |
| `scripts/run_all.sh` | Bash | ~25 | Service launcher |
| `tests/test_training.py` | Python | ~40 | Training tests |
| `tests/test_post_processing.py` | Python | ~60 | Filter tests |
| `tests/test_mcp_server.py` | Python | ~50 | MCP tests |
| `tests/test_agent.py` | Python | ~40 | Agent tests |
| `tests/test_e2e.py` | Python | ~50 | E2E tests |
| **Total** | | **~1,150** | |

---

*End of Design Specification*
