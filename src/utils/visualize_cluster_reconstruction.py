import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

def load_config():
    """Loads configuration from config/config.yaml."""
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(arg=None):
    # 1. Load configuration
    config = load_config()
    
    # 2. Determine Run ID (GUIDELINES.md Section 2)
    if arg is None:
        workflow_cfg = config.get('workflow', {})
        appliance_name = workflow_cfg.get('appliance_name', 'unknown_appliance')
        sequence_id = workflow_cfg.get('sequence_id')
        
        if not sequence_id:
            sequence_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Warning: sequence_id not found in config, using current time: {sequence_id}")
            
        run_id = f"{appliance_name}_{sequence_id}"
        print(f"Using Run ID from config: {run_id}")
    else:
        if os.path.isdir(arg):
            parts = arg.rstrip(os.sep).split(os.sep)
            if "log" in parts:
                idx = parts.index("log")
                run_id = parts[idx+1]
            else:
                run_id = os.path.basename(arg.rstrip(os.sep))
        else:
            run_id = arg
        print(f"Using provided Run ID: {run_id}")
    
    # Define paths
    log_root = os.path.join(project_root, "log", run_id)
    output_root = os.path.join(project_root, "output", run_id)
    
    print(f"Starting cluster reconstruction visualization for Run ID: {run_id}")
    
    # Check if TimeClustering and TimeSegmentation data exists
    seg_cfg = config.get('steps', {}).get('time_segmentation', {})
    seg_method = seg_cfg.get('segment_method', 'clasp').lower()
    time_seg_dir = os.path.join(log_root, f"TimeSegmentation_{seg_method}")
    
    cluster_cfg = config.get('steps', {}).get('time_clustering', {})
    cluster_method = cluster_cfg.get('cluster_method', 'dbscan').lower()
    time_cluster_dir = os.path.join(log_root, f"TimeClustering_{cluster_method}")
    
    # Try the 3-column indices.npy from clustering first
    indices_path = os.path.join(time_cluster_dir, "indices.npy")
    if not os.path.exists(indices_path):
        # Fallback to output directory
        indices_path = os.path.join(output_root, "output", "indices.npy")
        
    lengths_path = os.path.join(time_seg_dir, "lengths.npy")
    
    if not (os.path.exists(indices_path) and os.path.exists(lengths_path)):
        print(f"Error: Required files not found.")
        print(f"Expected indices.npy at {indices_path}")
        print(f"Expected lengths.npy at {lengths_path}")
        return

    visualize_reconstruction(log_root, output_root, config, indices_path, lengths_path)

def visualize_reconstruction(log_dir, output_root, config, indices_path, lengths_path):
    """
    Reconstructs signal with cluster labels on the original signals.
    """
    figure_dir = os.path.join(output_root, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    
    # Find input CSVs
    extract_cfg = config.get('steps', {}).get('extract_active_data', {})
    extract_method = extract_cfg.get('method', 'simple').lower()
    extract_folder = f"ExtractActiveData_{extract_method}"
    
    possible_dataloader_dirs = [
        os.path.join(log_dir, "DataLoader"),
        os.path.join(log_dir, extract_folder, "segments"),
        os.path.join(log_dir, extract_folder)
    ]
    
    dataloader_dir = None
    for d in possible_dataloader_dirs:
        if os.path.exists(d) and any(f.lower().endswith('.csv') for f in os.listdir(d)):
            dataloader_dir = d
            break
            
    if dataloader_dir is None:
        print(f"Error: Could not find CSV files.")
        return
    
    print(f"Found input CSVs in: {dataloader_dir}")

    # Load data
    indices = np.load(indices_path)
    lengths = np.load(lengths_path).flatten()
    
    if indices.shape[1] < 3:
        print(f"Error: indices.npy does not have 3 columns. Found {indices.shape[1]} columns.")
        return

    csv_files = sorted([f for f in os.listdir(dataloader_dir) if f.lower().endswith('.csv')])
    
    # Color mapping for clusters
    unique_clusters = np.unique(indices[:, 2])
    n_clusters = len(unique_clusters)
    cmap = plt.cm.get_cmap('tab10', 10) # Max 10 colors, will wrap if more
    
    # Noise color
    noise_color = (0.7, 0.7, 0.7, 0.3) # Gray with alpha
    
    for i, csv_name in enumerate(csv_files):
        csv_path = os.path.join(dataloader_dir, csv_name)
        df = pd.read_csv(csv_path)
        signal = df['power'].values
        csv_len = len(signal)
        
        # Filter segments for this CSV
        file_mask = (indices[:, 0] == i)
        if not np.any(file_mask):
            continue
            
        file_indices = indices[file_mask]
        
        plt.figure(figsize=(15, 6))
        plt.plot(signal, label='Original Signal', color='black', linewidth=1, alpha=0.8)
        
        # Plot segments with colors
        for idx_row in range(len(file_indices)):
            start_idx = file_indices[idx_row, 1]
            cluster_id = file_indices[idx_row, 2]
            
            # Find global index to get length
            # We need to find the global row index in 'indices' to match with 'lengths'
            # But wait, indices and lengths should be matched 1:1.
            # However, if some samples were removed, we need to be careful.
            # In TimeClusteringStep, I filtered indices by valid_mask.
            # So I should use the global index from the filtered set.
            
            # Actually, I can calculate end_idx if I have lengths for these specific indices.
            # Wait, the lengths array should match the filtered indices array.
            # Let's assume they match (as per my modification in TimeClusteringStep).
            
            # Find the index in the global indices array
            global_idx = np.where(file_mask)[0][idx_row]
            seg_len = lengths[global_idx]
            end_idx = start_idx + seg_len
            
            if cluster_id == -1:
                color = noise_color
                label = "Noise"
            else:
                # Use tab10 for clusters, cycle if needed
                color = cmap(cluster_id % 10)
                # Set alpha
                color = (color[0], color[1], color[2], 0.4)
                label = f"Cluster {cluster_id}"
            
            plt.axvspan(start_idx, end_idx, color=color, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.axvline(x=start_idx, color='red', linestyle='--', alpha=0.3)
            
            # Add text label for cluster ID
            plt.text(start_idx + seg_len/2, np.max(signal)*0.9, str(cluster_id), 
                     horizontalalignment='center', fontsize=8, color='darkred')

        plt.title(f'Cluster Reconstruction - {csv_name}')
        plt.xlabel('Time Index')
        plt.ylabel('Power')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        plot_path = os.path.join(figure_dir, f'cluster_reconstruction_{csv_name.replace(".csv", "")}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"[{i+1}/{len(csv_files)}] Saved cluster reconstruction plot: {plot_path}")
        
        if i >= 19: # Limit to first 20 files
            break
            
    print(f"\nDone! Cluster-based visualizations saved to: {figure_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
