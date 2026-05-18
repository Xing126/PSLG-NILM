import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Add src to path to import ApplianceDataSegmenter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.steps.extract_active_data_step import ApplianceDataSegmenter

def visualize_clustering(input_file, month_str):
    sequence_id = "20260518_washing"
    step_name = "ExtractActiveData"
    output_dir = f"/home/scnu2023024258/data/code/PSLG-NILM/log/{sequence_id}/{step_name}/figures"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data for clustering visualization from {input_file}...")
    df = pd.read_csv(input_file)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)
    
    # Filter for the specific month
    df_month = df[df.index.to_period("M") == month_str]
    valid_data = df_month["power"].dropna().values
    
    if len(valid_data) < 100:
        print(f"Not enough data for month {month_str}")
        return

    # 1. 提取底噪基准
    bg_baseline = np.percentile(valid_data, 25)

    # 2. 采样加速聚类
    sample_rate = 10
    sampled_data = valid_data[::sample_rate].reshape(-1, 1)

    # 3. 用极速聚类区分“背景”与“工作”
    kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
    kmeans.fit(sampled_data)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    bg_center, work_center = centers[0], centers[1]

    # 4. 根据两级中心计算阈值
    p_low = bg_center + (work_center - bg_center) * 0.15
    p_high = bg_center + (work_center - bg_center) * 0.40

    # 安全兜底
    p_low_final = max(p_low, bg_baseline + 5.0)
    p_high_final = max(p_high, p_low_final + 10.0)

    # Visualization
    plt.figure(figsize=(12, 7))
    
    # Plot histogram of the power data
    # Use log scale for y-axis because background data usually dominates
    plt.hist(valid_data, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Power Distribution')
    plt.yscale('log')
    
    # Plot cluster centers
    plt.axvline(bg_center, color='green', linestyle='--', linewidth=2, label=f'Background Center: {bg_center:.2f}W')
    plt.axvline(work_center, color='red', linestyle='--', linewidth=2, label=f'Working Center: {work_center:.2f}W')
    
    # Plot thresholds
    plt.axvline(p_low_final, color='orange', linestyle='-', linewidth=2, label=f'p_low (Threshold): {p_low_final:.2f}W')
    plt.axvline(p_high_final, color='purple', linestyle='-', linewidth=2, label=f'p_high (Threshold): {p_high_final:.2f}W')
    
    plt.title(f"Clustering Analysis for Washing Machine ({month_str})\nK-Means (k=2) on Sampled Power Data")
    plt.xlabel("Power (W)")
    plt.ylabel("Frequency (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"clustering_analysis_{month_str}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Saved clustering visualization to {filepath}")
    print(f"Results for {month_str}:")
    print(f"  Background Center: {bg_center:.2f}W")
    print(f"  Working Center: {work_center:.2f}W")
    print(f"  p_low: {p_low_final:.2f}W")
    print(f"  p_high: {p_high_final:.2f}W")

if __name__ == "__main__":
    input_file = "/home/scnu2023024258/data/code/PSLG-NILM/input/washing_machine.csv"
    visualize_clustering(input_file, "2012-11")
