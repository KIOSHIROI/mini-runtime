#!/bin/bash
# ============================================================================
# A800 环境一键安装脚本
# 用法: bash setup_a800.sh
# ============================================================================
set -e

echo "========================================"
echo "  mini-runtime A800 环境安装"
echo "========================================"

# ── 1. 检查 Python ──
PYTHON=$(command -v python3.11 || command -v python3 || command -v python)
echo "[1/4] Using Python: $PYTHON"
$PYTHON --version

# ── 2. 创建虚拟环境 ──
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating virtual environment ..."
    $PYTHON -m venv $VENV_DIR
else
    echo "[2/4] Virtual environment already exists, skip."
fi

source $VENV_DIR/bin/activate

# ── 3. 安装依赖 ──
echo "[3/4] Installing PyTorch CUDA + dependencies ..."
pip install --upgrade pip -q

# 安装 CUDA 版 PyTorch（A800 需要 CUDA 11.8+，这里用 cu130）
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130

# 安装其他依赖
pip install -q transformers>=4.40 safetensors>=0.4 huggingface-hub>=0.20

echo ""
echo "Installed packages:"
pip list | grep -E "torch|transformers|safetensors|huggingface"

# ── 4. 验证 CUDA ──
echo ""
echo "[4/4] Verifying CUDA ..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
else:
    print('WARNING: CUDA not detected! Check NVIDIA driver.')
"

echo ""
echo "========================================"
echo "  Installation complete!"
echo ""
echo "  Quick start:"
echo "    source $VENV_DIR/bin/activate"
echo "    python bench_a800.py --small      # 冒烟测试"
echo "    python bench_a800.py              # 中等扫描"
echo "    python bench_a800.py --large      # 完整扫描"
echo "========================================"
