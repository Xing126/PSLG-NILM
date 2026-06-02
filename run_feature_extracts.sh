#!/bin/bash

#SBATCH -J PSLG-FEATURE_EXTRACTS
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.err
#SBATCH --time=24:00:00

# --- 1. 环境准备 ---
# 不使用 module purge，确保驱动相关的基础环境不被清理
module load cuda-toolkit/12.1 

export PYTHONNOUSERSITE=1 # 禁止使用用户目录下的 Python 包
export PYTHONPATH=""      # 清空 Python 路径
export PYTHONUNBUFFERED=1 # 实时打印日志

# --- 2. 激活 Conda 环境 ---
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/bin/activate"
conda activate PSLG-NILM

# --- 3. 修复动态库搜索路径 ---
# 优先级：Conda NVIDIA库 > Conda基础库 > 系统路径(含驱动)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- 4. 诊断信息 ---
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# --- 5. 执行循环运行逻辑 ---
cd /home/scnu2023024258/data/code/PSLG-NILM

MODELS=("detsec" "bilstm_ae" "autoencoder" "dtw")
ORIGINAL_CONFIG="config/config.yaml"

for MODEL in "${MODELS[@]}"; do
    echo "================================================================"
    echo "Starting workflow with model_name: $MODEL"
    echo "================================================================"
    
    # 创建临时配置文件
    TEMP_CONFIG="config/config_${MODEL}.yaml"
    if ! cp "$ORIGINAL_CONFIG" "$TEMP_CONFIG"; then
        echo "Error: Failed to copy config for $MODEL. Skipping..."
        continue
    fi
    
    # 使用 sed 修改 model_name
    if ! sed -i "s/model_name: \".*\"/model_name: \"$MODEL\"/" "$TEMP_CONFIG"; then
        echo "Error: Failed to update config for $MODEL. Skipping..."
        continue
    fi
    
    echo "Using temporary config: $TEMP_CONFIG"
    
    # 运行 main.py
    python main.py --config "$TEMP_CONFIG"
    
    if [ $? -eq 0 ]; then
        echo "Workflow for $MODEL executed successfully!"
    else
        echo "Workflow for $MODEL failed! Skipping this model."
        continue
    fi
    
    echo "Finished $MODEL run."
    echo ""
done

# --- 6. 作业结束 ---
echo "All model tasks finished on: $(date)"
