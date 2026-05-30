import os
import gc
import sys
import yaml
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import medfilt

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
    
    # 2. Determine Run ID and Paths (GUIDELINES.md Section 2)
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
        # Use provided arg (could be full path or sequence_id)
        if os.path.isdir(arg):
            parts = arg.rstrip(os.sep).split(os.sep)
            # Check if any part starts with ExtractActiveData or TimeSegmentation
            extract_part = next((p for p in parts if p.startswith("ExtractActiveData")), None)
            seg_part = next((p for p in parts if p.startswith("TimeSegmentation")), None)
            
            if extract_part:
                idx = parts.index(extract_part)
                run_id = parts[idx-1]
            elif seg_part:
                idx = parts.index(seg_part)
                run_id = parts[idx-1]
            elif "log" in parts:
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
    
    print(f"Starting reconstruction visualization for Run ID: {run_id}")
    
    # Check if TimeSegmentation data exists
    # Get segment_method from config to build the correct folder name with suffix
    seg_cfg = config.get('steps', {}).get('time_segmentation', {})
    seg_method = seg_cfg.get('segment_method', 'clasp').lower()
    time_seg_folder = f"TimeSegmentation_{seg_method}"
    
    time_seg_dir = os.path.join(log_root, time_seg_folder)
    if not os.path.exists(time_seg_dir) or not os.path.exists(os.path.join(time_seg_dir, "X.npy")):
        print(f"Error: TimeSegmentation logs not found in {time_seg_dir}")
        print("This tool requires output from TimeSegmentationStep (X.npy, lengths.npy, indices.npy).")
        return

    visualize_from_logs(log_root, output_root, config)

def visualize_from_logs(log_dir, output_root, config):
    """
    Reconstructs changepoints from TimeSegmentation output files (X.npy, lengths.npy, indices.npy)
    and visualizes them on the original signals.
    """
    # Get segment_method from config
    seg_cfg = config.get('steps', {}).get('time_segmentation', {})
    seg_method = seg_cfg.get('segment_method', 'clasp').lower()
    time_seg_folder = f"TimeSegmentation_{seg_method}"
    time_seg_dir = os.path.join(log_dir, time_seg_folder)
    
    figure_dir = os.path.join(output_root, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    
    # Get extract method from config
    extract_cfg = config.get('steps', {}).get('extract_active_data', {})
    extract_method = extract_cfg.get('method', 'simple').lower()
    extract_folder = f"ExtractActiveData_{extract_method}"
    
    # Try different possible locations for input CSVs (DataLoader or ExtractActiveData)
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
        print(f"Error: Could not find CSV files in any of: {possible_dataloader_dirs}")
        return
    
    print(f"Found input CSVs in: {dataloader_dir}")

    x_path = os.path.join(time_seg_dir, "X.npy")
    l_path = os.path.join(time_seg_dir, "lengths.npy")
    i_path = os.path.join(time_seg_dir, "indices.npy")
    
    if not (os.path.exists(x_path) and os.path.exists(l_path)):
        print(f"Error: Required files (X.npy, lengths.npy) not found in {time_seg_dir}")
        return

    print(f"\n--- Reconstructing Changepoints from Logs: {log_dir} ---")
    
    # Load segmentation data
    X = np.load(x_path)
    lengths = np.load(l_path).flatten()
    
    # Try to load indices, fallback to length-based reconstruction if missing
    indices = None
    if os.path.exists(i_path):
        indices = np.load(i_path)
        print("Using indices.npy for precise changepoint reconstruction.")
    else:
        print("Warning: indices.npy not found. Using length-based reconstruction (less robust).")
    
    # Load original active segments (sorted to match TimeSegmentationStep logic)
    csv_files = sorted([f for f in os.listdir(dataloader_dir) if f.lower().endswith('.csv')])
    
    current_segment_idx = 0
    total_segments = len(lengths)
    
    # Color definitions
    blue = (74/255, 75/255, 157/255)
    red = (200/255, 22/255, 29/255)
    green = (90/255, 164/255, 174/255)
    yellow = (250/255, 192/255, 61/255)
    
    for i, csv_name in enumerate(csv_files):
        csv_path = os.path.join(dataloader_dir, csv_name)
        df = pd.read_csv(csv_path)
        signal = df['power'].values
        csv_len = len(signal)
        
        # Determine segments belonging to this CSV
        file_segment_indices = []
        if indices is not None:
            # Column 0 is CSV index, Column 1 is start index in the CSV
            file_mask = (indices[:, 0] == i)
            file_segment_indices = np.where(file_mask)[0]
            # Get start points as changepoints
            changepoints = indices[file_mask, 1]
            # Internal changepoints (start > 0)
            internal_cps = changepoints[changepoints > 0]
        else:
            # Fallback length-based matching
            acc_len = 0
            while acc_len < csv_len and current_segment_idx < total_segments:
                seg_len = lengths[current_segment_idx]
                acc_len += seg_len
                file_segment_indices.append(current_segment_idx)
                current_segment_idx += 1
            
            seg_lengths = [lengths[idx] for idx in file_segment_indices]
            internal_cps = np.cumsum(seg_lengths[:-1])
            
        if len(file_segment_indices) == 0:
            continue
            
        # Visualization: 4 subplots matching the separation steps
        plt.figure(figsize=(15, 12))
        
        # 1. Original Signal (from CSV) with reconstructed segment boundaries
        plt.subplot(4, 1, 1)
        plt.plot(signal, label='Original Signal', color='gray', alpha=0.7)
        for cp in internal_cps:
            plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8, label='Segment Boundary' if cp == internal_cps[0] else "")
        plt.title(f'Reconstructed Analysis - {csv_name}')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 2. Cleaned Signal (from X.npy Channel 1)
        plt.subplot(4, 1, 2)
        offset = 0
        for idx in file_segment_indices:
            seg_len = lengths[idx]
            cleaned_seg = X[idx, :seg_len, 1]
            plt.plot(np.arange(offset, offset + seg_len), cleaned_seg, color=blue)
            offset += seg_len
        plt.title('Cleaned Signal (Channel 1)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 3. Low Frequency (from X.npy Channel 2)
        plt.subplot(4, 1, 3)
        offset = 0
        for idx in file_segment_indices:
            seg_len = lengths[idx]
            low_seg = X[idx, :seg_len, 2]
            plt.plot(np.arange(offset, offset + seg_len), low_seg, color=green)
            offset += seg_len
        plt.title('Low Frequency Component (Channel 2)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 4. High Frequency (from X.npy Channel 3)
        plt.subplot(4, 1, 4)
        offset = 0
        for idx in file_segment_indices:
            seg_len = lengths[idx]
            high_seg = X[idx, :seg_len, 3]
            plt.plot(np.arange(offset, offset + seg_len), high_seg, color=yellow)
            offset += seg_len
        plt.title('High Frequency Component (Channel 3)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(figure_dir, f'reconstruction_{csv_name.replace(".csv", "")}.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"[{i+1}/{len(csv_files)}] Saved reconstruction plot: {plot_path}")
        
        if i >= nums: # Limit to first 10 for demonstration
            print("... (limited to first 10 files)")
            break
            
    print(f"\nDone! Log-based visualizations saved to: {figure_dir}")

if __name__ == "__main__":
    nums = 20
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
