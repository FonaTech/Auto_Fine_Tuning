@echo off
REM ================================================================
REM Auto_Fine_Tuning 安装脚本（Windows）
REM 用法: 双击运行 install.bat 或在 CMD 中执行
REM ================================================================

setlocal

echo ================================================================
echo   Auto_Fine_Tuning — SFT · DPO · Auto-ORPO - 安装脚本 (Windows)
echo ================================================================

REM ── 检测 Python ─────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.9+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%v
echo Python 版本: %PY_VER%

REM ── 检测 CUDA ────────────────────────────────────────────────────
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" > "%TEMP%\platform.txt" 2>nul
set /p PLATFORM=<"%TEMP%\platform.txt"
if "%PLATFORM%"=="" set PLATFORM=cpu

echo 检测到平台: %PLATFORM%

REM ── 安装依赖 ─────────────────────────────────────────────────────
if "%PLATFORM%"=="cuda" (
    echo 安装 CUDA 依赖...
    pip install -r requirements-cuda.txt
    echo.
    echo 提示：安装 Unsloth 最优化版本请运行：
    echo pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
) else (
    echo 未检测到 CUDA，安装 CPU 依赖...
    pip install -r requirements-cpu.txt
    echo.
    echo 警告：CPU 训练极慢，建议使用 NVIDIA GPU。
)

echo.
echo ================================================================
echo  安装完成！启动方式：
echo    python app.py
echo ================================================================
pause
