# Requirements Document: Telco Autonomous Operations Workshop
# Edge-Deployed Fine-Tuned SLM with Agentic AI for Network Root Cause Analysis

**Document Version:** 1.0
**Date:** 2026-03-09
**Classification:** Workshop Development Requirements
**Author:** Angelo Sampietro

---

## 1. Executive Summary

### 1.1 Purpose
This document defines the complete requirements for a hands-on workshop that teaches participants to build an end-to-end Autonomous Network Operations system. The system uses a fine-tuned Small Language Model (SLM) deployed at the edge as a "Semantic Filter" for raw 3GPP telemetry data, exposed via the Model Context Protocol (MCP), and orchestrated by a frontier Large Language Model (LLM) on Amazon Bedrock through an agentic framework.

### 1.2 Workshop Objective
Participants will build three integrated components on a single EC2 GPU instance:
1. **Fine-Tune Qwen3-14B** using QLoRA on a synthetic 3GPP telco dataset to create a domain-expert SLM capable of filtering noisy network logs into structured diagnostic JSON.
2. **Build a Smart MCP Server** that wraps the fine-tuned SLM with grammar-constrained decoding and a deterministic post-processing filter, exposing a tool via the MCP protocol (SSE transport).
3. **Build a Web GUI** (React/Next.js frontend + Python backend) that implements an AI agent using the AWS Strands SDK. The agent calls a configurable Amazon Bedrock frontier model endpoint and uses the Smart MCP Server tool to retrieve semantically filtered logs for Root Cause Analysis.

### 1.3 Key Architectural Principle — Semantic Compression
The core innovation is **Semantic Compression**: instead of streaming gigabytes of raw telemetry to a frontier LLM (causing token explosion and unsustainable costs), a locally-hosted fine-tuned SLM reads the raw data, understands the 3GPP context, and outputs only the causal chain as lightweight structured JSON. This reduces input to the frontier model by approximately 90%, enabling economically viable Autonomous Operations at scale.

### 1.4 End-to-End Demo Flow
The completed workshop deliverable demonstrates:
1. User opens the web GUI and types a natural language query (e.g., "Subscriber 999123 complains YouTube is buffering constantly despite full signal bars").
2. The Strands-based agent sends the query to Amazon Bedrock (configurable frontier model).
3. The frontier model reasons about the query and invokes the `analyze_network_telemetry` MCP tool.
4. The Smart MCP Server receives the tool call, fetches raw logs locally, runs them through the fine-tuned Qwen3-14B (via llama.cpp with GBNF grammar constraints), applies the deterministic post-processing filter, and returns a clean JSON diagnostic summary.
5. The frontier model synthesizes the filtered JSON into a human-readable diagnosis with actionable recommendations.
6. The GUI displays the agent's response in a chat interface.
7. A **comparison panel** shows: raw log volume (bytes/tokens), filtered output volume, token reduction percentage, and estimated cost savings.

---

## 2. Glossary

| Term | Definition |
|------|-----------|
| **SLM** | Small Language Model — a model with fewer than 20B parameters optimized for edge deployment |
| **3GPP** | 3rd Generation Partnership Project — standards body for mobile telecommunications |
| **QLoRA** | Quantized Low-Rank Adaptation — parameter-efficient fine-tuning method that applies LoRA adapters to a 4-bit quantized base model |
| **LoRA** | Low-Rank Adaptation — technique that trains small adapter matrices instead of full model weights |
| **GGUF** | GPT-Generated Unified Format — binary format for quantized LLM weights used by llama.cpp |
| **Q4_K_M** | A specific quantization profile using mixed-precision: 6-bit for attention heads, 4-bit for feed-forward layers |
| **GBNF** | GGML BNF — grammar format used by llama.cpp to constrain token generation to valid structures |
| **MCP** | Model Context Protocol — open protocol standardizing how AI agents interact with external tools and data sources |
| **SSE** | Server-Sent Events — HTTP-based protocol for server-to-client streaming |
| **NAS** | Non-Access Stratum — 3GPP signaling layer between UE and core network (AMF) |
| **NGAP** | Next Generation Application Protocol — interface between gNodeB and AMF |
| **RRC** | Radio Resource Control — protocol managing radio connections between UE and gNodeB |
| **RCA** | Root Cause Analysis — process of identifying the primary failure in a chain of symptoms |
| **Sympathetic Noise** | Consequential alarms triggered by a primary failure that are not themselves root causes |
| **ChatML** | Chat Markup Language — template format using `<\|im_start\|>` and `<\|im_end\|>` delimiters for multi-turn conversations |
| **Strands** | AWS Strands Agents SDK — open-source Python framework for building agentic AI workflows |
| **AgentCore** | Amazon Bedrock AgentCore — managed runtime for deploying production-grade AI agents |

---

## 3. System Architecture Overview

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SINGLE EC2 GPU INSTANCE                       │
│                                                                       │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐   │
│  │   Web GUI     │    │         Smart MCP Server                  │   │
│  │  (React/Next) │    │                                            │   │
│  │               │    │  ┌────────────┐  ┌─────────────────────┐ │   │
│  │  Chat UI      │    │  │ MCP Tool   │  │  llama.cpp Engine   │ │   │
│  │  + Comparison │    │  │ (SSE)      │──│  Qwen3-14B GGUF     │ │   │
│  │    Panel      │    │  │            │  │  + GBNF Grammar     │ │   │
│  │               │    │  └────────────┘  │  + LoRA Adapter     │ │   │
│  └──────┬───────┘    │                   └─────────────────────┘ │   │
│         │             │  ┌────────────────────────────────────┐   │   │
│         │             │  │  Post-Processing Filter             │   │   │
│         │             │  │  (Deterministic Sympathetic Noise   │   │   │
│         │             │  │   Removal)                          │   │   │
│         │             │  └────────────────────────────────────┘   │   │
│  ┌──────▼───────┐    └──────────────────────────────────────────┘   │
│  │ Python Backend│                                                    │
│  │ (Strands Agent│◄──── MCP SSE Connection ────►                     │
│  │  + Bedrock)   │                                                    │
│  └──────┬───────┘                                                    │
│         │                                                             │
└─────────┼─────────────────────────────────────────────────────────┘
          │ HTTPS (Bedrock API)
          ▼
┌──────────────────┐
│  Amazon Bedrock   │
│  (Frontier Model) │
│  Configurable:    │
│  - Claude 4.5     │
│  - Nova 2 Pro     │
│  - Others         │
└──────────────────┘
```

### 3.2 Data Flow

```
Raw 3GPP Logs (MB)
       │
       ▼
┌──────────────────┐
│ Qwen3-14B SLM    │  ← Grammar-constrained decoding (GBNF)
│ (llama.cpp, Q4)  │  ← Fine-tuned with QLoRA adapter
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Post-Processing   │  ← Deterministic sympathetic noise filter
│ Filter            │
└────────┬─────────┘
         │
         ▼
   Clean JSON (KB)  ← ~90% token reduction
         │
         ▼
┌──────────────────┐
│ Frontier Model    │  ← Reasoning + synthesis
│ (Bedrock)         │
└────────┬─────────┘
         │
         ▼
  Human-Readable Diagnosis + Recommendations
```

### 3.3 Production Deployment Note
For production deployment, the architecture maps to:
- **Edge (AWS Outposts / AWS AI Factories):** Smart MCP Server + fine-tuned SLM
- **Cloud (AWS Region):** Strands Agent deployed on Amazon Bedrock AgentCore for managed runtime, memory management, identity propagation, and secure networking via PrivateLink
- **Connectivity:** AWS Direct Connect with private VIF

The workshop simulates this hybrid architecture on a single EC2 instance. The Strands agent code is written to be directly deployable to AgentCore without modification.

---

## 4. EC2 Instance Selection Guide

### 4.1 Hardware Requirements Analysis

The workshop runs two GPU-intensive workloads on the same instance:

**Phase 1 — Fine-Tuning (QLoRA 4-bit):**
- Qwen3-14B base model in 4-bit NF4 quantization: ~8 GB VRAM
- LoRA adapter weights + optimizer states: ~4 GB VRAM
- Activation memory (batch_size=1, seq_len=2048, gradient checkpointing): ~6 GB VRAM
- CUDA kernels + overhead: ~2 GB VRAM
- **Total fine-tuning VRAM requirement: ~20 GB minimum**

**Phase 2 — Inference (llama.cpp Q4_K_M):**
- Qwen3-14B Q4_K_M GGUF model: ~9 GB VRAM
- KV cache (16K context window): ~2 GB VRAM
- CUDA kernel workspace: ~1 GB VRAM
- **Total inference VRAM requirement: ~12 GB minimum**

**Additional requirements:**
- System RAM: ≥64 GB (model loading, data preprocessing)
- Storage: ≥250 GB (base model download ~28 GB, GGUF conversion, datasets, adapters)
- CPU: ≥8 vCPUs (data preprocessing, GGUF conversion, web server)

### 4.2 Instance Comparison Matrix

| Instance Type | GPU | VRAM | vCPUs | RAM | Storage | Fine-Tune | Inference | Cost (us-east-1) | Verdict |
|--------------|-----|------|-------|-----|---------|-----------|-----------|-------------------|---------|
| **g5.4xlarge** | 1x A10G | 24 GB | 16 | 64 GB | EBS | ✅ Tight (24 GB, single GPU) | ✅ Comfortable | ~$1.62/hr | ⚠️ Feasible but tight for fine-tuning |
| **g5.12xlarge** | 4x A10G | 96 GB | 48 | 192 GB | EBS | ✅ Comfortable (multi-GPU) | ✅ Comfortable | ~$5.67/hr | ✅ Safe choice, fast training |
| **g6.4xlarge** | 1x L4 | 24 GB | 16 | 64 GB | EBS | ✅ Tight (24 GB, single GPU) | ✅ Comfortable | ~$1.32/hr | ⚠️ Feasible but tight |
| **g6.12xlarge** | 4x L4 | 96 GB | 48 | 192 GB | EBS | ✅ Comfortable (multi-GPU) | ✅ Comfortable | ~$4.60/hr | ✅ **RECOMMENDED** |
| **p4d.24xlarge** | 8x A100 | 320 GB | 96 | 1152 GB | 8x1TB NVMe | ✅ Overkill | ✅ Overkill | ~$32.77/hr | ❌ Excessive for workshop |
| **p5.48xlarge** | 8x H100 | 640 GB | 192 | 2048 GB | 8x3.84TB | ✅ Overkill | ✅ Overkill | ~$98.32/hr | ❌ Excessive for workshop |

### 4.3 Recommended Instance: g6.12xlarge

**Justification:**
- **4x NVIDIA L4 GPUs (24 GB VRAM each, 96 GB total):** Provides comfortable headroom for QLoRA fine-tuning of Qwen3-14B in 4-bit. The model shards across GPUs automatically via `device_map="auto"`.
- **48 vCPUs, 192 GB RAM:** Sufficient for concurrent data preprocessing, GGUF conversion (CPU-intensive), web server, and MCP server.
- **Cost-effective:** At ~$4.60/hr, a complete workshop session (fine-tuning ~30 min + inference setup ~15 min + demo ~30 min) costs approximately $5.75.
- **L4 GPU architecture:** Ada Lovelace generation with excellent INT8/FP8 support, optimized for inference workloads. The L4's Tensor Cores accelerate both the QLoRA training and the llama.cpp inference.

**Alternative for budget-constrained workshops:** g5.4xlarge or g6.4xlarge (single GPU, 24 GB). Fine-tuning will work but requires careful memory management (gradient checkpointing mandatory, batch_size=1 only, may need to reduce max_seq_length to 1024).

### 4.4 Storage Configuration
- **Root volume:** 250 GB gp3 EBS
  - Base model download (Qwen3-14B HuggingFace weights): ~28 GB
  - GGUF converted model: ~9 GB
  - Training dataset: ~1 GB
  - LoRA adapter checkpoints: ~500 MB
  - Python environment + dependencies: ~15 GB
  - Working space: ~50 GB
- **IOPS:** 3000 (gp3 default) — sufficient for model loading
- **Throughput:** 125 MB/s (gp3 default) — sufficient

### 4.5 AMI Selection
- **Recommended:** AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
- **Rationale:** Pre-installed NVIDIA drivers and CUDA toolkit, eliminating GPU driver compatibility issues. Participants install Python dependencies on top.

---

## 5. Component 1: Fine-Tuning Qwen3-14B with QLoRA

### 5.1 Objective
Fine-tune the open-weight Qwen3-14B model to become a domain expert in 3GPP protocol Root Cause Analysis. The fine-tuned model must be able to:
- Read raw, noisy telco signaling logs (NAS, NGAP, RRC, Transport layers)
- Distinguish primary causal failures from sympathetic noise (consequential alarms)
- Output a structured JSON diagnostic summary containing only the root cause events
- Correctly return an empty events array (`{"events": []}`) for logs that contain only routine operations (heartbeats, keepalives, successful sessions)

### 5.2 Base Model Specification

| Property | Value |
|----------|-------|
| **Model** | Qwen/Qwen3-14B |
| **Architecture** | Qwen2 (decoder-only transformer) |
| **Parameters** | 14.2 billion |
| **Context Window** | 128K tokens (training uses 2048 max) |
| **License** | Apache 2.0 |
| **Source** | Hugging Face Hub |

### 5.3 Fine-Tuning Method: QLoRA

#### 5.3.1 Why QLoRA
Full-parameter fine-tuning of a 14B model requires >100 GB VRAM (FP16) — impractical for workshop hardware. QLoRA enables fine-tuning by:
1. Quantizing the base model to 4-bit NF4 precision (~8 GB VRAM)
2. Attaching small trainable LoRA adapter matrices to specific layers
3. Computing gradients only for the adapter parameters (~0.1% of total weights)

#### 5.3.2 Default QLoRA Configuration (Proven Baseline)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Quantization** | 4-bit NF4 via bitsandbytes | Optimal compression with minimal perplexity loss |
| **Double Quantization** | Enabled | Further reduces memory by quantizing the quantization constants |
| **Compute dtype** | float16 | Balance of precision and speed |
| **LoRA Rank (r)** | 16 | Constrained rank prevents catastrophic forgetting while allowing sufficient adaptation |
| **LoRA Alpha (α)** | 32 | Scaling factor; α/r = 2 is a standard effective learning rate multiplier |
| **LoRA Dropout** | 0.1 | Regularization to prevent overfitting on repetitive log patterns |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All linear layers — necessary for learning rigid 3GPP syntax (per QLoRA paper findings) |
| **Task Type** | CAUSAL_LM | Standard for decoder-only text generation |

**Experimentation note:** Participants may experiment with rank values (8, 16, 32) and alpha values (16, 32, 64) to observe the impact on model quality. The default configuration is proven to achieve F1 ≥ 0.91 on the test set.

#### 5.3.3 Training Hyperparameters (Proven Baseline)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 1 | Telco logs are highly repetitive; >1 epoch causes overfitting (model memorizes specific logs instead of learning patterns) |
| **Batch Size** | 1 | Memory constraint on single/multi-GPU setup |
| **Gradient Accumulation Steps** | 4 | Effective batch size = 4, smooths gradient updates |
| **Learning Rate** | 1e-4 | Standard for QLoRA fine-tuning |
| **LR Scheduler** | Cosine | Gradual decay prevents training instability |
| **Warmup Steps** | 20 | Prevents early gradient explosion |
| **Max Sequence Length** | 2048 | Sufficient for log analysis prompts; keeps VRAM manageable |
| **Gradient Checkpointing** | Enabled | Trades compute for memory — critical for fitting on available VRAM |
| **Optimizer** | paged_adamw_32bit | Memory-efficient optimizer from bitsandbytes |
| **BF16 Training** | Enabled | Mixed precision for speed |
| **Trainer** | SFTTrainer (from TRL library) | Supervised Fine-Tuning trainer optimized for instruction tuning |

**Experimentation note:** Participants may experiment with learning rates (5e-5, 1e-4, 2e-4) and epoch fractions (0.25, 0.5, 1.0). The default configuration achieves training loss convergence from ~1.8 to ~0.22 in approximately 325 steps (~31 minutes on g6.12xlarge).

### 5.4 Training Dataset Specification

#### 5.4.1 Dataset Structure
The training dataset is provided as a pre-generated JSON file. It is NOT generated during the workshop.

| Property | Value |
|----------|-------|
| **Total Examples** | 1,300 |
| **Positive Examples** | 1,100 (logs containing failures that should be detected) |
| **Negative Examples** | 200 (logs containing only routine operations — model must output empty events) |
| **Format** | JSON array of conversation objects |
| **File Size** | ~1 MB |

#### 5.4.2 Chat Template Format (Qwen ChatML)
Each training example follows the Qwen ChatML template with three turns:

```
<|im_start|>system
You are a 3GPP Protocol Expert specializing in Root Cause Analysis.
Analyze the logs provided.
1. Ignore routine Heartbeats and KeepAlives.
2. Filter out sympathetic noise (consequential failures).
3. Identify the primary causal chain (NAS, NGAP, or RRC).
4. Output ONLY valid JSON according to the schema.<|im_end|>
<|im_start|>user
[RAW 3GPP LOG LINES - timestamps, protocol layers, error codes, cell IDs]<|im_end|>
<|im_start|>assistant
{"events": [{"error_code": "ERROR_CODE", "severity": "high|medium|low"}]}<|im_end|>
```

#### 5.4.3 Scenario Categories in the Dataset

| Scenario | Count | Description | Expected Output |
|----------|-------|-------------|-----------------|
| core_network_failure | ~180 | AMF/UPF failures (5GMM_CAUSE_NETWORK_FAILURE, PDU_SESSION_FAILURE) | Events with core error codes |
| authentication_failure | ~100 | AUTHENTICATION_FAILURE + REGISTRATION_REJECTED | Auth-specific events only |
| radio_failure | ~180 | RRC_SETUP_FAILURE, signal degradation, cell-level issues | Radio error codes |
| handover_failure | ~140 | Inter-cell handover failures, X2/Xn interface issues | Handover-specific codes |
| transport_jitter | ~180 | TCP retransmission, backhaul jitter, N3 interface degradation | Transport layer codes |
| congestion | ~100 | Network congestion, 5GMM_CAUSE_CONGESTION, overload | Congestion codes |
| qos_violation | ~50 | QoS policy violations, GBR not met | QoS-specific codes |
| normal (negative) | ~200 | Heartbeats, keepalives, successful sessions, routine operations | `{"events": []}` |

#### 5.4.4 Importance of Negative Examples
The 200 negative examples are critical for precision. Without them, the model develops a bias toward always reporting errors (high recall, low precision). Negative examples teach the model to recognize when logs are healthy and output an empty events array, reducing false positives by >90%.

### 5.5 Training Output Artifacts

| Artifact | Description | Approximate Size |
|----------|-------------|-----------------|
| LoRA Adapter | Trained adapter weights (adapter_model.safetensors + adapter_config.json) | ~500 MB |
| Training Log | Loss curve, token accuracy per step | ~30 KB |
| Merged Model | Base model + adapter merged into full HuggingFace format | ~28 GB |
| GGUF Model | Merged model converted to Q4_K_M GGUF format for llama.cpp | ~9 GB |

### 5.6 Model Conversion Pipeline

After fine-tuning, the LoRA adapter must be converted to a format usable by llama.cpp:

**Step 1: Merge LoRA Adapter into Base Model**
- Load the 4-bit base model and the trained LoRA adapter
- Merge adapter weights into the base model using PEFT's `merge_and_unload()`
- Save the merged model in HuggingFace format (full precision FP16)

**Step 2: Convert HuggingFace Model to GGUF**
- Use `llama.cpp/convert_hf_to_gguf.py` to convert the merged HuggingFace model to GGUF format
- Output: FP16 GGUF file (~28 GB)

**Step 3: Quantize GGUF to Q4_K_M**
- Use `llama.cpp/tools/quantize/quantize` to apply Q4_K_M quantization
- Q4_K_M uses intelligent mixed-precision: sensitive attention heads at 6-bit, feed-forward layers at 4-bit
- Output: Q4_K_M GGUF file (~9 GB)
- This quantized model retains >97% of the full-precision model's reasoning capabilities while fitting in ~9 GB VRAM

### 5.7 Expected Training Metrics

| Metric | Expected Range |
|--------|---------------|
| Initial Training Loss | 1.5 – 2.0 |
| Final Training Loss | 0.15 – 0.30 |
| Token Accuracy | >90% |
| Training Duration (g6.12xlarge) | 25 – 35 minutes |
| Training Steps | ~325 (1,300 examples / effective_batch_size 4) |

### 5.8 Software Dependencies for Fine-Tuning

| Package | Purpose |
|---------|---------|
| torch (≥2.0) | Deep learning framework |
| transformers (≥4.40) | Model loading, tokenization |
| peft (≥0.10) | QLoRA/LoRA adapter management |
| trl | SFTTrainer for supervised fine-tuning |
| bitsandbytes | 4-bit NF4 quantization |
| datasets | Dataset loading and processing |
| accelerate | Multi-GPU training orchestration |

---

## 6. Component 2: Smart MCP Server

### 6.1 Objective
Build an MCP-compliant server that exposes a single tool (`analyze_network_telemetry`) via SSE transport. When invoked, the tool:
1. Fetches raw 3GPP signaling logs for a given subscriber (IMSI)
2. Passes the raw logs through the fine-tuned Qwen3-14B model running on llama.cpp with GBNF grammar-constrained decoding
3. Applies a deterministic post-processing filter to remove known sympathetic noise patterns
4. Returns a clean, validated JSON diagnostic summary

### 6.2 MCP Server Specification

| Property | Value |
|----------|-------|
| **Framework** | Anthropic MCP Python SDK (FastMCP) |
| **Transport** | SSE (Server-Sent Events) over HTTP |
| **Port** | 8000 (configurable) |
| **Tools Exposed** | 1: `analyze_network_telemetry` |
| **Protocol Version** | MCP 1.0 |

### 6.3 Tool Definition: `analyze_network_telemetry`

#### 6.3.1 Interface

| Property | Value |
|----------|-------|
| **Name** | `analyze_network_telemetry` |
| **Description** | "Retrieves network signaling logs for a subscriber IMSI, filters noise using a local fine-tuned SLM, and returns a clean JSON summary of failure events for Root Cause Analysis." |
| **Parameters** | `imsi` (string, required): Subscriber IMSI identifier. `context` (string, optional, default="general_failure"): Description of the reported symptom (e.g., "YouTube Buffering", "Call Drop"). |
| **Returns** | JSON string conforming to the output schema |

#### 6.3.2 Output JSON Schema

```json
{
  "type": "object",
  "properties": {
    "events": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "error_code": { "type": "string" },
          "severity": { "type": "string", "enum": ["high", "medium", "low"] }
        },
        "required": ["error_code", "severity"]
      }
    }
  },
  "required": ["events"]
}
```

Example output:
```json
{
  "events": [
    {"error_code": "TCP_RETRANSMISSION", "severity": "high"},
    {"error_code": "TRANSPORT_JITTER", "severity": "high"}
  ]
}
```

### 6.4 Inference Engine: llama.cpp with Grammar Constraints

#### 6.4.1 Runtime Specification

| Property | Value |
|----------|-------|
| **Engine** | llama-cpp-python (Python bindings for llama.cpp) |
| **Model Format** | GGUF (Q4_K_M quantization) |
| **Context Window** | 16,384 tokens |
| **GPU Acceleration** | CUDA (uses custom CUDA kernels for on-the-fly 4-bit dequantization) |
| **VRAM Usage** | ~9 GB model + ~2 GB KV cache + ~1 GB overhead = ~12 GB total |

#### 6.4.2 GBNF Grammar for Constrained Decoding

The inference engine MUST use a GBNF grammar to guarantee that every model output is structurally valid JSON conforming to the output schema. This eliminates the need for JSON parsing error handling and ensures 100% structural validity.

The GBNF grammar definition:

```
root     ::= object
object   ::= "{" ws "\"events\"" ws ":" ws array ws "}"
array    ::= "[" ws (event-obj (ws "," ws event-obj)*)? ws "]"
event-obj ::= "{" ws "\"error_code\"" ws ":" ws string ws "," ws "\"severity\"" ws ":" ws string ws "}"
string   ::= "\"" ([^"\\] | "\\" .)* "\""
ws       ::= [ \t\n]*
```

**How grammar-constrained decoding works:**
At each token generation step, llama.cpp intersects the model's probability distribution with the set of tokens allowed by the grammar's current state. Only tokens that would produce a valid continuation of the grammar are considered. This means:
- The output always starts with `{"events":`
- Each event object always contains both `error_code` and `severity` fields
- The JSON is always syntactically valid — no unclosed braces, no missing quotes
- The model cannot produce free-text explanations or markdown — only the constrained JSON structure

**Trade-off:** Grammar constraints limit the model's output space, which can slightly increase inference latency (the model may need more tokens to express the same information within the grammar's bounds). However, for our structured output use case, this overhead is negligible (~5-10% latency increase) and the guarantee of valid JSON is critical for production reliability.

#### 6.4.3 System Prompt

The system prompt instructs the model on its analysis rules:

```
You are a 3GPP Protocol Expert specializing in Root Cause Analysis.
Analyze the logs provided.
1. Ignore routine Heartbeats and KeepAlives.
2. Filter out sympathetic noise (consequential failures).
3. Identify the primary causal chain (NAS, NGAP, or RRC).
4. Output ONLY valid JSON according to the schema.
```

#### 6.4.4 Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **max_tokens** | 512 | Sufficient for up to ~20 events; prevents runaway generation |
| **temperature** | 0.1 | Near-deterministic output for consistent diagnostics |
| **repeat_penalty** | 1.2 | Prevents the model from repeating the same error code multiple times |
| **top_p** | 0.9 | Nucleus sampling for slight diversity |

### 6.5 Deterministic Post-Processing Filter

#### 6.5.1 Purpose
Even with fine-tuning and grammar constraints, the SLM can produce false positives — particularly in scenarios involving cascading failures where sympathetic alarms are difficult to distinguish from root causes. The post-processing filter applies deterministic, domain-expert rules to remove known sympathetic noise patterns from the model's output.

#### 6.5.2 Filter Rules

The filter applies the following rules sequentially to the model's JSON output:

**Rule 1: UPF Degradation Cascade**
- **Trigger:** If the events array contains `UPF_DEGRADATION`
- **Action:** Remove all downstream RRC errors (`RRC_SETUP_FAILURE`, `RRC_REESTABLISHMENT_FAILURE`) and NAS errors (`5GMM_CAUSE_CONGESTION`) from the events array
- **Rationale:** When a User Plane Function degrades, it triggers a storm of thousands of sympathetic alarms across neighboring gNodeBs and AMFs. These are consequences, not causes.

**Rule 2: Core Network Failure Cascade**
- **Trigger:** If the events array contains `5GMM_CAUSE_NETWORK_FAILURE` or `PDU_SESSION_FAILURE`
- **Action:** Remove all RRC-layer errors from the events array
- **Rationale:** Core network failures cause radio-layer symptoms. The RRC errors are effects of the core failure, not independent problems.

**Rule 3: Authentication Failure Isolation**
- **Trigger:** If the events array contains `AUTHENTICATION_FAILURE`
- **Action:** Keep only `AUTHENTICATION_FAILURE` and `REGISTRATION_REJECTED` in the events array; remove all other error codes
- **Rationale:** Authentication failures have a unique causal chain. Any other errors present are either unrelated or consequential.

**Rule 4: Deduplication**
- **Trigger:** Always applied
- **Action:** Remove duplicate error codes from the events array (keep first occurrence)
- **Rationale:** The model may generate the same error code multiple times due to repetitive log patterns.

#### 6.5.3 Filter Implementation Requirements
- The filter MUST be implemented as a pure Python function with no ML dependencies
- The filter MUST operate on the parsed JSON object (not raw text)
- The filter MUST be deterministic — same input always produces same output
- The filter MUST preserve the original event order (after removals)
- The filter MUST log which rules were applied and which events were removed (for debugging/transparency)

#### 6.5.4 Expected Impact
Based on validation testing:
- Precision improvement: +0.05 to +0.10 (fewer false positives)
- Recall impact: -0.02 to -0.04 (occasional true positive removed)
- Net F1 improvement: +0.02 to +0.04
- Primary benefit: Eliminates systematic false positives in `radio_failure`, `qos_violation`, and `congestion` scenarios

### 6.6 Log Data Source

For the workshop, the MCP Server uses synthetic log data (hardcoded or loaded from a local file). The tool function:
1. Receives the IMSI and context parameters
2. Fetches synthetic raw logs (simulating a query to an on-premises data lake)
3. Passes logs through the inference engine + post-processing filter
4. Returns the filtered JSON

The synthetic logs must cover the same scenario categories as the training dataset (core_network_failure, authentication_failure, radio_failure, handover_failure, transport_jitter, congestion, qos_violation, normal).

### 6.7 Software Dependencies for MCP Server

| Package | Purpose |
|---------|---------|
| mcp (Anthropic MCP SDK) | MCP protocol implementation, FastMCP server |
| llama-cpp-python | Python bindings for llama.cpp inference engine |
| uvicorn | ASGI server for SSE transport |

### 6.8 llama.cpp Build Requirements

llama.cpp must be compiled from source with CUDA support to enable GPU-accelerated inference:
- Clone llama.cpp repository
- Build with `cmake -DGGML_CUDA=ON`
- The `llama-cpp-python` package must be installed with CUDA support: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`
- Verify GPU detection: the inference engine should report CUDA device usage on startup

---

## 7. Component 3: Web GUI with Strands Agent

### 7.1 Objective
Build a web application with a React/Next.js frontend and a Python backend. The backend implements an AI agent using the AWS Strands SDK that:
1. Connects to a configurable Amazon Bedrock frontier model endpoint for reasoning
2. Connects to the Smart MCP Server (via SSE) to use the `analyze_network_telemetry` tool
3. Orchestrates the diagnostic conversation: receives user queries, invokes the frontier model, which decides when to call the MCP tool, and synthesizes the final diagnosis

### 7.2 Frontend Specification (React/Next.js)

#### 7.2.1 Technology Stack

| Technology | Purpose |
|-----------|---------|
| Next.js (≥14) | React framework with API routes |
| React (≥18) | UI component library |
| TypeScript | Type safety |
| Tailwind CSS | Styling |

#### 7.2.2 User Interface: Chat Interface

The GUI presents a single-page chat interface:

**Chat Area:**
- Message input field at the bottom where the user types natural language queries
- Scrollable message history showing the conversation between user and agent
- User messages displayed on the right, agent responses on the left
- Agent responses rendered as formatted text (markdown support for code blocks, bold, lists)
- Loading indicator while the agent is processing
- Error display for failed requests

**Example interaction:**
- User: "Subscriber 999123 is complaining that YouTube buffers constantly, even though they have full signal bars."
- Agent: "I'll analyze the network telemetry for this subscriber. [calls analyze_network_telemetry tool] Based on the analysis, the subscriber is experiencing **Packet Loss on the Transport Backhaul**. While the radio signal is strong, the N3 interface shows 15% TCP retransmission with transport jitter. **Recommendation:** Dispatch Field Tech to inspect the microwave backhaul link at the affected site."

#### 7.2.3 Comparison Panel

Below or beside the chat area, a comparison panel displays metrics for the most recent tool invocation:

| Metric | Description | Example Value |
|--------|-------------|---------------|
| **Raw Log Size** | Size of the raw logs fetched by the MCP server (bytes) | 45,230 bytes |
| **Raw Log Tokens** | Estimated token count of raw logs (chars / 4 approximation) | ~11,308 tokens |
| **Filtered Output Size** | Size of the JSON returned by the MCP tool (bytes) | 187 bytes |
| **Filtered Output Tokens** | Token count of the filtered JSON | ~47 tokens |
| **Token Reduction** | Percentage reduction: (1 - filtered/raw) × 100 | 99.6% |
| **Estimated Cost Savings** | Approximate cost difference based on Bedrock input token pricing | "$0.0158 saved per query" |
| **Inference Latency** | Time taken by the local SLM to process the logs | 6.2 seconds |

The comparison panel visually demonstrates the economic value of the Semantic Compression architecture.

#### 7.2.4 Frontend-Backend Communication
- The frontend communicates with the Python backend via REST API (Next.js API routes proxying to the Python backend, or direct fetch calls)
- Endpoint: `POST /api/chat` with body `{"message": "user query text"}`
- Response: JSON with `{"response": "agent response text", "metrics": {...comparison data...}}`
- The frontend should support streaming responses if the backend provides them (nice-to-have, not mandatory)

### 7.3 Backend Specification (Python + Strands)

#### 7.3.1 Technology Stack

| Technology | Purpose |
|-----------|---------|
| Python ≥3.11 | Backend runtime |
| AWS Strands SDK | Agent framework for tool orchestration |
| FastAPI or Flask | HTTP server for the REST API |
| boto3 | AWS SDK for Bedrock API calls |

#### 7.3.2 Agent Architecture

The backend implements a Strands agent with the following configuration:

**Agent Persona (System Prompt):**
```
You are a Telco Network Operations AI Assistant specializing in Root Cause Analysis
for 5G networks. When a user reports a network issue:
1. Extract the subscriber identifier (IMSI) and symptom description from the query.
2. Use the analyze_network_telemetry tool to retrieve filtered diagnostic data.
3. Interpret the diagnostic JSON and provide a clear, actionable diagnosis.
4. Include a specific recommendation (e.g., dispatch field tech, escalate to core team).
Always explain your reasoning in plain language suitable for a NOC operator.
```

**Model Configuration:**
- The Bedrock model ID must be configurable via environment variable (e.g., `BEDROCK_MODEL_ID`)
- Default: Claude 4.5 Haiku (`anthropic.claude-4-5-haiku-latest-v1:0` or equivalent)
- Alternative: Amazon Nova 2 Pro, Claude 4.5 Sonnet, or any Bedrock-supported model
- The agent code MUST NOT be hardcoded to a specific model — switching models should require only changing the environment variable

**Tool Binding:**
- The Strands agent connects to the Smart MCP Server via SSE transport at `http://localhost:8000`
- The MCP connection URL must be configurable via environment variable (e.g., `MCP_SERVER_URL`)
- The agent automatically discovers the `analyze_network_telemetry` tool from the MCP server's tool listing

#### 7.3.3 Metrics Collection
The backend must collect and return the following metrics with each response:
- Raw log size (bytes) — obtained from the MCP server's internal processing
- Filtered output size (bytes) — size of the JSON returned by the MCP tool
- Token estimates — calculated using character count / 4 approximation
- Inference latency — time between MCP tool call and response
- Total response time — end-to-end time from user query to agent response

These metrics are returned alongside the agent's response for the comparison panel.

#### 7.3.4 Production Deployment Note
The Strands agent code should be structured so it can be deployed to Amazon Bedrock AgentCore without modification. Specifically:
- Agent logic should be encapsulated in a single module/class
- Configuration should be externalized (environment variables)
- No hardcoded file paths or localhost URLs
- The MCP server URL should be parameterized for production (where it would point to an Outpost endpoint via PrivateLink)

AgentCore provides managed runtime with memory management (context retention across turns), identity propagation, and secure networking. The workshop code should mention this as the production deployment target in code comments.

### 7.4 Amazon Bedrock Configuration

#### 7.4.1 Prerequisites
- AWS account with Amazon Bedrock access enabled
- Model access granted for the chosen frontier model(s) in the target region
- IAM role/credentials with `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions
- AWS credentials configured on the EC2 instance (instance profile or environment variables)

#### 7.4.2 Supported Models
The system must work with any Bedrock model that supports the Converse API. Recommended models for the workshop:

| Model | Model ID | Strengths | Latency |
|-------|----------|-----------|---------|
| Claude 4.5 Haiku | anthropic.claude-4-5-haiku-* | Fast, good tool use | Low |
| Amazon Nova 2 Pro | amazon.nova-2-pro-* | Highest reasoning, best for complex RCA | Medium |
| Claude 4.5 Sonnet | anthropic.claude-4-5-sonnet-* | Balanced reasoning + speed | Medium |

---

## 8. Validation and Testing Requirements

### 8.1 Automated Test Suite

#### 8.1.1 Unit Tests

**Fine-Tuning Component:**
- Test that the training dataset loads correctly and has the expected number of examples (1,300)
- Test that the dataset contains both positive and negative examples
- Test that each example follows the Qwen ChatML template format
- Test that the LoRA adapter saves and loads correctly
- Test that the merged model produces valid output on a sample input

**MCP Server Component:**
- Test that the MCP server starts and responds to health checks
- Test that the `analyze_network_telemetry` tool is discoverable via MCP tool listing
- Test that the tool returns valid JSON conforming to the output schema for a sample input
- Test that the GBNF grammar produces structurally valid JSON (no parse errors)
- Test that the post-processing filter correctly applies each rule:
  - Rule 1: UPF cascade removal
  - Rule 2: Core network cascade removal
  - Rule 3: Authentication isolation
  - Rule 4: Deduplication
- Test that the post-processing filter returns empty events for normal logs

**GUI/Agent Component:**
- Test that the backend API endpoint (`POST /api/chat`) accepts requests and returns responses
- Test that the Strands agent correctly binds to the MCP server tools
- Test that the metrics collection returns valid numeric values
- Test that the frontend renders without errors

#### 8.1.2 Integration Tests

- Test the full pipeline: send a query to the backend → agent calls Bedrock → Bedrock invokes MCP tool → MCP server runs inference → post-processing → response returned to frontend
- Test with multiple scenario types (at least one from each category: core_network_failure, radio_failure, transport_jitter, normal)
- Test that the comparison metrics are populated correctly

#### 8.1.3 Model Quality Benchmarks (F1/Precision/Recall)

Run the fine-tuned model against the full 1,000-example sophisticated test set:

| Metric | Minimum Acceptable | Target |
|--------|-------------------|--------|
| **F1 Score** | ≥ 0.85 | ≥ 0.91 |
| **Precision** | ≥ 0.85 | ≥ 0.87 |
| **Recall** | ≥ 0.85 | ≥ 0.95 |
| **Exact Match** | ≥ 50% | ≥ 62% |
| **Average Inference Latency** | ≤ 15 seconds | ≤ 7 seconds |

**With post-processing filter applied:**

| Metric | Minimum Acceptable | Target |
|--------|-------------------|--------|
| **F1 Score** | ≥ 0.90 | ≥ 0.94 |
| **Precision** | ≥ 0.93 | ≥ 0.97 |
| **Recall** | ≥ 0.85 | ≥ 0.91 |
| **Exact Match** | ≥ 65% | ≥ 77% |

The validation script must:
1. Load the test dataset (1,000 examples)
2. Run each example through the inference engine + post-processing filter
3. Compare predicted events against ground truth events
4. Calculate True Positives, False Positives, False Negatives
5. Compute F1, Precision, Recall, and Exact Match
6. Generate a per-scenario breakdown table
7. Report average inference latency

### 8.2 Manual End-to-End Demo

The workshop concludes with a live demonstration through the GUI:

**Demo Scenario 1: YouTube Buffering (Transport Issue)**
- Input: "Subscriber 999123 complains YouTube buffers constantly despite full signal bars"
- Expected: Agent identifies TCP retransmission / transport jitter on N3 interface
- Comparison panel shows ~99% token reduction

**Demo Scenario 2: Call Drop (Radio Issue)**
- Input: "Subscriber 888456 experiencing frequent call drops in downtown area"
- Expected: Agent identifies RRC failures, signal degradation
- Comparison panel shows significant token reduction

**Demo Scenario 3: No Issue (Negative Case)**
- Input: "Check if subscriber 777789 has any network issues — they called to ask about their bill"
- Expected: Agent reports no network anomalies detected (empty events from MCP tool)
- Comparison panel shows filtering correctly identified healthy logs

**Demo Scenario 4: Authentication Failure**
- Input: "Subscriber 666012 cannot register on the network after SIM swap"
- Expected: Agent identifies authentication failure + registration rejected
- Post-processing filter correctly isolates auth-specific codes

---

## 9. Non-Functional Requirements

### 9.1 Performance

| Requirement | Target |
|-------------|--------|
| Local SLM inference latency | ≤ 10 seconds per query |
| End-to-end response time (user query → GUI response) | ≤ 20 seconds |
| MCP server startup time | ≤ 60 seconds (model loading) |
| GUI page load time | ≤ 3 seconds |

### 9.2 Reliability
- The MCP server must handle inference errors gracefully and return a valid error JSON: `{"events": [], "status": "error", "message": "..."}`
- The Strands agent must handle MCP connection failures and report them to the user
- The GUI must display error messages for failed requests without crashing

### 9.3 Security
- AWS credentials must NOT be hardcoded in source code — use instance profiles or environment variables
- The Bedrock API calls must use HTTPS (default)
- The MCP server runs on localhost only (not exposed to the internet)
- No real PII is used in the workshop — all data is synthetic

### 9.4 Portability
- All Python code must run on Python ≥3.11
- The frontend must work in modern browsers (Chrome, Firefox, Safari, Edge)
- The system must run on Ubuntu 22.04 (AWS Deep Learning AMI)

---

## 10. Project Structure

```
workshop/
├── README.md                          # Workshop overview and setup instructions
├── requirements.txt                   # Python dependencies
├── package.json                       # Node.js dependencies (frontend)
│
├── data/
│   ├── training_data.json             # 1,300 training examples (pre-generated, copied)
│   └── test_data.json                 # 1,000 test examples (pre-generated, copied)
│
├── fine_tuning/
│   ├── train.py                       # QLoRA fine-tuning script
│   ├── merge_adapter.py               # Merge LoRA adapter into base model
│   ├── convert_to_gguf.sh             # Convert merged model to GGUF Q4_K_M
│   └── validate.py                    # Run F1/Precision/Recall benchmarks
│
├── mcp_server/
│   ├── server.py                      # FastMCP server with SSE transport
│   ├── inference_engine.py            # llama.cpp inference with GBNF grammar
│   ├── post_processing_filter.py      # Deterministic sympathetic noise filter
│   └── telco_grammar.gbnf             # GBNF grammar file for JSON output
│
├── agent/
│   ├── agent.py                       # Strands agent with Bedrock + MCP tool binding
│   ├── api_server.py                  # FastAPI/Flask REST API backend
│   └── config.py                      # Configuration (env vars, model IDs, URLs)
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Main chat page
│   │   │   └── api/chat/route.ts      # API route (proxy to Python backend)
│   │   └── components/
│   │       ├── ChatInterface.tsx       # Chat UI component
│   │       ├── MessageBubble.tsx       # Individual message display
│   │       └── ComparisonPanel.tsx     # Token reduction metrics display
│   ├── package.json
│   └── tailwind.config.js
│
├── tests/
│   ├── test_training.py               # Fine-tuning unit tests
│   ├── test_mcp_server.py             # MCP server unit tests
│   ├── test_post_processing.py        # Post-processing filter tests
│   ├── test_agent.py                  # Agent integration tests
│   └── test_e2e.py                    # End-to-end pipeline tests
│
└── scripts/
    ├── setup_instance.sh              # EC2 instance setup (CUDA, Python, Node.js)
    ├── download_model.sh              # Download Qwen3-14B from HuggingFace
    ├── build_llamacpp.sh              # Build llama.cpp with CUDA support
    └── run_all.sh                     # Start all services (MCP server + backend + frontend)
```

---

## 11. Workshop Execution Order

The workshop follows this sequence:

1. **Environment Setup** (~20 min): Launch EC2 instance, install dependencies, download base model
2. **Fine-Tuning** (~40 min): Run QLoRA training, monitor loss convergence, save adapter
3. **Model Conversion** (~20 min): Merge adapter, convert to GGUF, quantize to Q4_K_M
4. **Validation** (~15 min): Run automated benchmarks on test set, verify F1/Precision/Recall
5. **MCP Server** (~30 min): Build inference engine with GBNF grammar, implement post-processing filter, start MCP server
6. **Agent Backend** (~30 min): Configure Strands agent with Bedrock + MCP tool, implement REST API
7. **Frontend** (~30 min): Build React/Next.js chat interface with comparison panel
8. **Integration & Demo** (~15 min): Connect all components, run end-to-end demo scenarios

**Total estimated time: ~3.5 hours**

---

## 12. Constraints and Assumptions

### 12.1 Constraints
- The workshop runs entirely on a single EC2 GPU instance (no multi-instance deployment)
- The training dataset is pre-generated and provided — no dataset generation during the workshop
- All telco log data is synthetic — no real subscriber data or PII
- Internet access is required for Hugging Face model download and Bedrock API calls
- AWS account must have Bedrock model access pre-approved before the workshop

### 12.2 Assumptions
- Participants have basic familiarity with Python, React, and AWS services
- The EC2 instance has been pre-provisioned with the recommended AMI and instance type
- AWS credentials (instance profile) are pre-configured with Bedrock access
- The Hugging Face model (Qwen3-14B) remains available and accessible
- llama.cpp CUDA compilation succeeds on the chosen AMI (tested with Ubuntu 22.04 + CUDA 12.x)

### 12.3 Out of Scope
- Real PII handling, data sovereignty, or GDPR compliance implementation
- AWS Outposts or AI Factories deployment
- Amazon Bedrock AgentCore deployment (mentioned as production path only)
- AWS Direct Connect or PrivateLink configuration
- Multi-tenant or multi-user support
- Authentication/authorization for the web GUI
- CI/CD pipeline
- Monitoring and observability (CloudWatch, X-Ray)

---

## 13. Acceptance Criteria

The workshop deliverable is considered complete when:

1. ✅ The fine-tuned Qwen3-14B model achieves F1 ≥ 0.85 on the 1,000-example test set (≥ 0.90 with post-processing filter)
2. ✅ The GGUF Q4_K_M model loads successfully in llama.cpp and produces grammar-constrained JSON output
3. ✅ The MCP server starts, exposes the `analyze_network_telemetry` tool via SSE, and returns valid JSON for all test scenarios
4. ✅ The post-processing filter correctly applies all four rules (verified by unit tests)
5. ✅ The Strands agent successfully connects to Bedrock and the MCP server, orchestrating a complete diagnostic conversation
6. ✅ The web GUI displays the chat interface and comparison panel with accurate metrics
7. ✅ The end-to-end demo completes successfully for all four demo scenarios (YouTube buffering, call drop, no issue, authentication failure)
8. ✅ All automated tests pass (unit + integration + model benchmarks)
9. ✅ The comparison panel correctly shows ≥90% token reduction for failure scenarios

---

*End of Requirements Document*
