#!/bin/bash
# T-103: Convert merged HuggingFace model to GGUF Q4_K_M for llama.cpp
set -euo pipefail

MERGED_DIR="${1:-output/merged_model}"
OUTPUT_DIR="${2:-output}"
LLAMACPP_DIR="${3:-llama.cpp}"

QUANTIZE_BIN="$LLAMACPP_DIR/build/bin/llama-quantize"
CONVERT_SCRIPT="$LLAMACPP_DIR/convert_hf_to_gguf.py"

# Validate prerequisites
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "ERROR: $CONVERT_SCRIPT not found. Run scripts/setup_instance.sh first."
    exit 1
fi
if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "ERROR: $QUANTIZE_BIN not found. Build llama.cpp with CUDA first."
    exit 1
fi
if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "ERROR: Merged model not found at $MERGED_DIR. Run merge_adapter.py first."
    exit 1
fi

echo "Step 1/3: Converting HF → GGUF (FP16)..."
python3 "$CONVERT_SCRIPT" "$MERGED_DIR" \
    --outfile "$OUTPUT_DIR/model-fp16.gguf" --outtype f16
echo "FP16 GGUF: $(du -h "$OUTPUT_DIR/model-fp16.gguf" | cut -f1)"

echo "Step 2/3: Quantizing to Q4_K_M..."
"$QUANTIZE_BIN" "$OUTPUT_DIR/model-fp16.gguf" \
    "$OUTPUT_DIR/qwen3-14b-telco-Q4_K_M.gguf" Q4_K_M

echo "Step 3/3: Cleaning up intermediate FP16 GGUF..."
rm "$OUTPUT_DIR/model-fp16.gguf"

echo ""
echo "Done ✓"
ls -lh "$OUTPUT_DIR/qwen3-14b-telco-Q4_K_M.gguf"
