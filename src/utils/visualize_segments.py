import os
import gc
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Define project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

def load_config():
    """Loads configuration from config/config.yaml."""
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def plot_final_segments(signal, output_path, title):
    """
    Plots the power signal of a single segment.
    """
    plt.figure(figsize=(12, 6))
    
    # Standard color palette
    signal_color = (74/255, 75/255, 157/255)  # Blue
    
    plt.plot(signal, label='Power Signal', color=signal_color, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Power (W)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def main(arg=None):
    # 1. Load configuration
    config = load_config()
    
    # 2. Determine Run ID and Paths (GUIDELINES.md Section 2)
    # Priority 1: Argument (Run ID or Directory)
    # Priority 2: Config (appliance_name + sequence_id)
    
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
            # If it's a directory like log/run_id/ExtractActiveData/segments
            # We try to extract run_id from the path
            parts = arg.rstrip(os.sep).split(os.sep)
            if "ExtractActiveData" in parts:
                idx = parts.index("ExtractActiveData")
                run_id = parts[idx-1]
            else:
                run_id = os.path.basename(arg.rstrip(os.sep))
        else:
            run_id = arg
        print(f"Using provided Run ID: {run_id}")
    
    # Define output paths
    output_root = os.path.join(project_root, "output", run_id)
    figure_dir = os.path.join(output_root, "figure")
    
    # Ensure directories exist
    os.makedirs(figure_dir, exist_ok=True)
    
    print(f"Starting visualization for Run ID: {run_id}")
    
    # 3. Read segments (Step 1 of the logic)
    print("\nStep 1: Reading Active Segments...")
    
    # Priority for segments directory:
    # 1. If arg is a directory, use it directly
    # 2. Otherwise, use standard log path: log/{run_id}/ExtractActiveData/segments
    if arg and os.path.isdir(arg):
        segments_dir = arg
    else:
        segments_dir = os.path.join(project_root, "log", run_id, "ExtractActiveData", "segments")
    
    print(f"Source segments directory: {segments_dir}")
    
    if not os.path.exists(segments_dir):
        print(f"Error: Segments directory not found: {segments_dir}")
        return

    segment_files = [
        os.path.join(segments_dir, f)
        for f in os.listdir(segments_dir)
        if f.lower().endswith(".csv")
    ]
    segment_files.sort()
    
    if not segment_files:
        print(f"Error: No segment files (.csv) found in {segments_dir}")
        return
        
    print(f"Successfully found {len(segment_files)} segments.")
    
    # 5. Process and Visualize (Step 2 of the logic)
    # We limit to 10 segments for visualization to avoid excessive resource usage
    max_plots = 30
    target_segments = segment_files[:max_plots]
    
    print(f"\nStep 2: Visualizing Segments (Top {len(target_segments)})...")
    for i, file_path in enumerate(target_segments):
        file_name = os.path.basename(file_path)
        print(f"Processing segment {i+1}/{len(target_segments)}: {file_name}")
        
        # Load power data
        df = pd.read_csv(file_path)
        if 'power' not in df.columns:
            print(f"Warning: 'power' column missing in {file_name}. Skipping.")
            continue
            
        signal = df['power'].values
        
        # Generate the plot
        output_file = os.path.join(figure_dir, f"segment_{file_name.replace('.csv', '.png')}")
        plot_final_segments(signal, output_file, f"Segment Visualization - {file_name}")
        
        # Explicit memory cleanup (GUIDELINES.md Section 3)
        del df, signal
        gc.collect()

    print(f"\nExecution Complete! Results saved to: {figure_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
