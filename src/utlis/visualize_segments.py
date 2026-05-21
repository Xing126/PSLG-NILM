import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from src.steps.extract_active_data_step import ApplianceDataSegmenter

def main():
    input_file = "/home/scnu2023024258/data/code/PSLG-NILM/input/washing_machine.csv"
    sequence_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"visualize_segments_{sequence_id}"
    
    # According to GUIDELINES.md: output/{run_id}/
    output_dir = os.path.join(project_root, "output", run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {input_file}...")
    segmenter = ApplianceDataSegmenter(
        appliance_name="washing_machine",
        power_threshold=1.0,
        min_duration_seconds=30,
        context_seconds=40
    )
    
    # Read data using segmenter's method
    data = segmenter._read_data(input_file)
    print(f"Read {len(data)} points.")

    # Detect segments
    segments = segmenter._detect_working_segments_from_data(data)
    print(f"Detected {len(segments)} segments.")

    # Take first 20
    top_20_segments = segments[:30]
    
    df = pd.DataFrame(data, columns=["timestamp", "power"])

    for i, (s_idx, e_idx, s_time, e_time, duration) in enumerate(top_20_segments):
        print(f"Processing segment {i+1}/{len(top_20_segments)}: duration {duration}s")
        
        # Add 50 points buffer
        start_plot_idx = max(0, s_idx - 50)
        end_plot_idx = min(len(df) - 1, e_idx + 50)
        
        plot_data = df.iloc[start_plot_idx : end_plot_idx + 1]
        
        plt.figure(figsize=(12, 6))
        # Use index as x-axis for clarity in point-based buffer
        plt.plot(plot_data.index, plot_data["power"], marker='.', linestyle='-', markersize=2, label="Power")
        
        # Highlight the detected segment
        plt.axvspan(s_idx, e_idx, color='red', alpha=0.2, label='Working Segment')
        
        # Mark the start and end of the segment
        plt.axvline(x=s_idx, color='green', linestyle='--', alpha=0.5)
        plt.axvline(x=e_idx, color='orange', linestyle='--', alpha=0.5)

        plt.title(f"Washing Machine Segment {i+1} (Duration: {duration}s, Points: {e_idx-s_idx+1})")
        plt.xlabel("Data Point Index")
        plt.ylabel("Power (W)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"segment_{i+1:02d}_{s_time}_{duration}s.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved plot to {filepath}")

    print(f"\nSuccessfully generated {len(top_20_segments)} plots in: {output_dir}")

if __name__ == "__main__":
    main()
