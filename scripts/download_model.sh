#!/bin/bash
# T-003: Download Qwen3-14B base model from HuggingFace Hub
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source venv/bin/activate

MODEL_DIR="models/Qwen3-14B"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model already downloaded at $MODEL_DIR"
    exit 0
fi

echo "Downloading Qwen/Qwen3-14B to $MODEL_DIR (~28 GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-14B', local_dir='$MODEL_DIR')
print('Download complete')
"

echo "Verifying tokenizer loads..."
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_DIR')
print(f'Tokenizer loaded: vocab_size={tok.vocab_size}')
"

echo "Model ready at $MODEL_DIR"
du -sh "$MODEL_DIR"
