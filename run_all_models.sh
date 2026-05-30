#!/bin/bash

#SBATCH -J PSLG-NILM-ALL
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.err
#SBATCH --time=24:00:00

# --- 1. 作业开始，打印基本信息 ---
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

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

# --- 5. 执行循环运行逻辑 ---
cd /home/scnu2023024258/data/code/PSLG-NILM

METHODS=("clasp" "fluss" "espresso" "clasp-origin")
ORIGINAL_CONFIG="config/config.yaml"

for METHOD in "${METHODS[@]}"; do
    echo "================================================================"
    echo "Starting workflow with segment_method: $METHOD"
    echo "================================================================"
    
    # 创建临时配置文件
    TEMP_CONFIG="config/config_${METHOD}.yaml"
    if ! cp "$ORIGINAL_CONFIG" "$TEMP_CONFIG"; then
        echo "Error: Failed to copy config for $METHOD. Skipping..."
        continue
    fi
    
    # 使用 sed 修改 segment_method (匹配 line 26 附近的格式)
    # 替换 segment_method: "..." 为当前方法
    if ! sed -i "s/segment_method: \".*\"/segment_method: \"$METHOD\"/" "$TEMP_CONFIG"; then
        echo "Error: Failed to update config for $METHOD. Skipping..."
        continue
    fi
    
    echo "Using temporary config: $TEMP_CONFIG"
    
    # 运行 main.py
    python main.py --config "$TEMP_CONFIG"
    
    if [ $? -eq 0 ]; then
        echo "Workflow for $METHOD executed successfully!"
    else
        echo "Workflow for $METHOD failed! Skipping this model."
        continue
    fi
    
    # 清理临时配置文件 (可选，若需保留请注释掉)
    # rm "$TEMP_CONFIG"
    
    echo "Finished $METHOD run."
    echo ""
done

# --- 6. 作业结束 ---
echo "All tasks finished on: $(date)"
