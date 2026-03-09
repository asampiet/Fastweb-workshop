# Telco Autonomous Operations Workshop

Build an end-to-end AI-powered Root Cause Analysis system for 5G networks using a fine-tuned Small Language Model at the edge, exposed via MCP, and orchestrated by a frontier model on Amazon Bedrock.

## Architecture

```
Raw 3GPP Logs (MB) → Fine-tuned Qwen3-14B (llama.cpp, GBNF grammar)
    → Post-processing filter → Clean JSON (KB)
    → Frontier Model (Bedrock) → Human-readable diagnosis
```

Three components on a single EC2 GPU instance:
1. **Fine-tuned SLM** — Qwen3-14B with QLoRA, quantized to GGUF Q4_K_M
2. **Smart MCP Server** — Grammar-constrained inference + deterministic noise filter, SSE transport
3. **Web GUI** — React/Next.js chat + Strands agent + Bedrock + comparison panel

## Prerequisites

- EC2 g6.12xlarge (4x L4, 96 GB VRAM) with Deep Learning Base AMI (Ubuntu 22.04)
- AWS account with Bedrock model access (Claude 4.5 Haiku or Nova 2 Pro)
- ~250 GB EBS storage

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> && cd workshop
bash scripts/setup_instance.sh        # ~20 min
bash scripts/download_model.sh        # ~15 min

# 2. Fine-tune
source venv/bin/activate
python fine_tuning/train.py            # ~31 min, loss 1.8 → 0.22

# 3. Convert to GGUF
python fine_tuning/merge_adapter.py    # ~10 min
bash fine_tuning/convert_to_gguf.sh    # ~10 min, produces ~9 GB GGUF

# 4. Validate
python fine_tuning/validate.py \
  --model-path $(pwd)/output/qwen3-14b-telco-Q4_K_M.gguf \
  --with-filter                        # ~90 min, expect F1 ≥ 0.90

# 5. Run everything
bash scripts/run_all.sh               # Starts MCP + Backend + Frontend
# Open http://localhost:3000
```

## Workshop Steps (Detail)

### Step 1: Environment Setup (~20 min)
`scripts/setup_instance.sh` installs Python ML stack, Node.js, llama.cpp with CUDA.

### Step 2: Fine-Tuning (~40 min)
`fine_tuning/train.py` runs QLoRA (r=16, α=32) on 1,300 examples. Targets all linear layers. Expect final loss < 0.30.

### Step 3: Model Conversion (~20 min)
Merge LoRA adapter → convert to GGUF → quantize to Q4_K_M (~9 GB, fits in 12 GB VRAM).

### Step 4: Validation (~15 min quick / ~90 min full)
Run against 1,000-example test set. Use `--limit 50` for quick check, full run for benchmarks.

### Step 5: MCP Server (~5 min)
Starts on port 8000. Loads GGUF model with GBNF grammar constraints. Exposes `analyze_network_telemetry` tool.

### Step 6: Agent Backend (~5 min)
Strands agent on port 8080. Connects to Bedrock + MCP. Configurable model via `BEDROCK_MODEL_ID` env var.

### Step 7: Frontend (~5 min)
Next.js on port 3000. Chat interface + comparison panel showing token reduction metrics.

### Step 8: Demo
Try these queries:
- "Subscriber 999123 complains YouTube buffers constantly despite full signal bars"
- "Subscriber 888456 experiencing frequent call drops in downtown area"
- "Check subscriber 777789 for issues — they called about billing"
- "Subscriber 666012 cannot register on the network after SIM swap"

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `output/qwen3-14b-telco-Q4_K_M.gguf` | GGUF model path (absolute) |
| `MCP_PORT` | `8000` | MCP server port |
| `MCP_SERVER_URL` | `http://localhost:8000/sse` | MCP SSE endpoint |
| `BEDROCK_MODEL_ID` | `anthropic.claude-4-5-haiku-latest-v1:0` | Bedrock model |
| `AWS_REGION` | `us-east-1` | AWS region |
| `API_PORT` | `8080` | Backend API port |
| `BACKEND_URL` | `http://localhost:8080` | Backend URL (for frontend proxy) |

## Testing

```bash
# Unit + component tests (run anywhere)
python -m pytest tests/ -v --ignore=tests/test_e2e.py

# Full e2e tests (requires all services running)
python -m pytest tests/test_e2e.py -v
```

## Project Structure

```
├── data/                    # Training (1,300) + test (1,000) datasets
├── fine_tuning/             # train.py, merge, convert, validate
├── mcp_server/              # server.py, inference engine, filter, grammar
├── agent/                   # Strands agent, FastAPI backend, config
├── frontend/src/            # Next.js React chat + comparison panel
├── tests/                   # Unit, component, and e2e tests
└── scripts/                 # Setup, download, run_all
```

## Production Path

For production, this architecture maps to:
- **Edge:** Smart MCP Server on AWS Outposts / AI Factories
- **Cloud:** Strands agent on Amazon Bedrock AgentCore
- **Connectivity:** AWS Direct Connect with private VIF
- **Privacy:** Deterministic PII hashing before data leaves premises
