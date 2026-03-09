#!/bin/bash
# T-002: EC2 Instance Setup Script
# Target: AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# Instance: g6.12xlarge (4x L4, 96GB VRAM) recommended
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Telco Workshop - Instance Setup"
echo "  Project dir: $PROJECT_DIR"
echo "============================================"

# 1. System packages (cmake needed for llama.cpp build)
echo ""
echo "=== Step 1/5: System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq cmake build-essential git python3-venv

# 2. Python virtual environment + ML dependencies
echo ""
echo "=== Step 2/5: Python environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

echo "Installing ML packages (torch, transformers, peft, trl, bitsandbytes)..."
pip install -q torch transformers peft trl bitsandbytes datasets accelerate

echo "Installing server packages (fastapi, uvicorn, mcp, strands)..."
pip install -q fastapi uvicorn pydantic
pip install -q "mcp[cli]"
pip install -q strands-agents strands-agents-tools boto3

echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python -q

# 3. Node.js 20 for frontend
echo ""
echo "=== Step 3/5: Node.js ==="
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y -qq nodejs
fi
echo "Node.js $(node --version), npm $(npm --version)"

# 4. Build llama.cpp with CUDA (needed for quantize tool)
echo ""
echo "=== Step 4/5: Build llama.cpp ==="
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
cmake --build build --config Release -j$(nproc) 2>&1 | tail -5
cd "$PROJECT_DIR"
echo "llama.cpp quantize binary: $(ls llama.cpp/build/bin/llama-quantize 2>/dev/null || echo 'NOT FOUND')"

# 5. Verify GPU
echo ""
echo "=== Step 5/5: GPU Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem // 1024**3} GB)')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate env: source venv/bin/activate"
echo "============================================"
