#!/bin/bash

#SBATCH -J PSLG-NILM-FLUSS
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.err
#SBATCH --time=24:00:00

# 默认使用 fluss 专用配置文件
CONFIG_PATH=${1:-config/config_fluss.yaml}

# --- 1. 作业开始，打印基本信息 ---
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Using config: $CONFIG_PATH"

# --- 2. 创建日志和输出目录 ---
mkdir -p /home/scnu2023024258/data/code/PSLG-NILM/slurm_log
mkdir -p /home/scnu2023024258/data/code/PSLG-NILM/output
mkdir -p /home/scnu2023024258/data/code/PSLG-NILM/log
echo "Created necessary directories"

# --- 3. 设置软件环境 ---
module purge
echo "Environment purged."
module load miniconda3/latest cuda-toolkit/12.1
echo "Modules loaded."

# --- 4. 激活 Conda 环境 ---
source $(conda info --base)/bin/activate
conda activate PSLG-NILM
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- 5. 针对 FLUSS 段错误的优化设置 ---
# FLUSS 依赖 stumpy/numba，多线程并行有时会导致段错误
# 限制线程数为 1 可以显著提高鲁棒性，虽然会降低单文件处理速度
echo "Applying thread limits to prevent Segmentation Fault in FLUSS..."
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- 6. 执行脚本 ---
echo "Starting main.py with FLUSS configuration..."
cd /home/scnu2023024258/data/code/PSLG-NILM

python main.py --config "$CONFIG_PATH"

if [ $? -eq 0 ]; then
    echo "FLUSS Workflow executed successfully!"
else
    echo "FLUSS Workflow execution failed!"
fi

# --- 7. 作业结束 ---
echo "Job finished on: $(date)"
