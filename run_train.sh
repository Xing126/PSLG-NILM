#!/bin/bash
#SBATCH -J PSLG-CLSTER
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.err
#SBATCH --time=24:00:00

# --- 1. 彻底隔离环境 ---
module purge              # 清除所有系统模块
export PYTHONNOUSERSITE=1 # 禁止使用用户目录下的 Python 包
export PYTHONPATH=""      # 清空 Python 路径
export PYTHONUNBUFFERED=1 # 实时打印日志

# --- 2. 激活 Conda 环境 (使用正确的自动获取路径) ---
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/bin/activate"
conda activate PSLG-NILM

# --- 3. 修复动态库搜索路径 (核心修复) ---
# 强制程序优先去 Conda 环境的 nvidia 库目录找 .so 文件
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# --- 4. 诊断信息 ---
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- 5. 执行 ---
cd /home/scnu2023024258/data/code/PSLG-NILM
python main.py --config "${1:-config/config.yaml}"
