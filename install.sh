#!/usr/bin/env bash
# ================================================================
# Auto_Fine_Tuning 安装脚本（Linux / macOS）
# 用法: bash install.sh
# ================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo "  Auto_Fine_Tuning — SFT · DPO · Auto-ORPO - 安装脚本"
echo "================================================================"

# ── 检测 Python ──────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "❌ 未找到 Python，请先安装 Python 3.9+"
    exit 1
fi

PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python 版本: $PY_VER"

# ── 检测平台与 GPU ───────────────────────────────────────────────
PLATFORM=$($PYTHON -c "
import sys, platform
p = platform.system().lower()
try:
    import torch
    if torch.cuda.is_available():
        print('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('mps')
    else:
        print('cpu')
except ImportError:
    if p == 'darwin':
        print('mps')
    else:
        print('cpu')
" 2>/dev/null || echo "cpu")

echo "检测到平台: $PLATFORM"

# ── 安装依赖 ─────────────────────────────────────────────────────
case "$PLATFORM" in
    cuda)
        echo "安装 CUDA 依赖..."
        pip install -r requirements-cuda.txt
        echo ""
        echo "提示：如需安装 Unsloth 最优化版本，请运行："
        echo '  pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"'
        echo "（根据实际 CUDA 版本选择 cu118 / cu121 / cu124）"
        ;;
    mps)
        echo "安装 macOS MPS 依赖..."
        pip install -r requirements-mps.txt
        echo ""
        echo "提示：macOS 不支持 4bit 量化，训练将使用 bfloat16 精度。"
        ;;
    cpu)
        echo "安装 CPU 依赖（注意：CPU 训练极慢）..."
        pip install -r requirements-cpu.txt
        ;;
esac

echo ""
echo "================================================================"
echo "✅ 安装完成！启动方式："
echo "   python app.py"
echo "================================================================"
