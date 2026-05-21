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

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dependencies
from models.time_segmentation.claspy.segmentation import BinaryClaSPSegmentation
from src.steps.extract_active_data_step import ApplianceDataSegmenter

def load_config():
    """Loads configuration from config/config.yaml."""
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_segmentation_points(time_series, distance="znormed_euclidean_distance"):
    """Segmentation logic using BinaryClaSPSegmentation."""
    try:
        clasp = BinaryClaSPSegmentation(
            n_segments="learn",
            window_size="suss",
            validation="score_threshold",
            threshold=0.001,
            distance=distance,
            n_jobs=1,
        )
        clasp.fit_predict(time_series)
        return clasp.change_points
    except Exception as e:
        print(f"Segmentation error: {e}")
        return []

def synthesize_changepoints(low_cp, high_cp):
    """Synthesizes changepoints from low and high frequency components."""
    if len(low_cp) == 0 and len(high_cp) == 0:
        return []

    if len(low_cp) >= len(high_cp):
        ref_cp = np.sort(low_cp)
    else:
        ref_cp = np.sort(high_cp)

    if len(ref_cp) == 0:
        return []

    # Merging logic consistent with project standards
    others = [np.sort(high_cp) if len(low_cp) >= len(high_cp) else np.sort(low_cp)]
    groups = {i: [ref_val] for i, ref_val in enumerate(ref_cp)}
    for other_list in others:
        for p in other_list:
            closest_idx = np.argmin(np.abs(ref_cp - p))
            groups[closest_idx].append(p)
            
    synthesized_cp = []
    for i in sorted(groups.keys()):
        group_mean = np.mean(groups[i])
        synthesized_cp.append(group_mean)
        
    return sorted(synthesized_cp)

def medfilt_outlier_removal(series):
    """Simple median filter for noise reduction."""
    return medfilt(np.asarray(series), kernel_size=5)

def run_segmentation(signal, wavelet='db4'):
    """Core logic to get final synthesized changepoints using wavelet separation."""
    level = 2
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA2, cD2, cD1 = coeffs
    
    zeros_cD2 = np.zeros_like(cD2)
    zeros_cD1 = np.zeros_like(cD1)
    zeros_cA2 = np.zeros_like(cA2)
    
    low_freq_signal = pywt.waverec([cA2, zeros_cD2, zeros_cD1], wavelet)[:len(signal)]
    high_freq_combined = pywt.waverec([zeros_cA2, cD2, cD1], wavelet)[:len(signal)]
    
    low_cp = get_segmentation_points(low_freq_signal)
    high_cp = get_segmentation_points(high_freq_combined)
    
    return synthesize_changepoints(low_cp, high_cp)

def plot_final_segments(signal, synth_cp, output_path, title):
    """
    Simplified plot showing only the final segmentation result.
    Equivalent to the total effect plot (plt.subplot(4, 1, 4)) without comparisons.
    """
    plt.figure(figsize=(12, 6))
    
    # Standard color palette from GUIDELINES.md context
    signal_color = (74/255, 75/255, 157/255)  # Blue
    synth_color = (166/255, 85/255, 157/255)   # Magenta/Purple
    
    plt.plot(signal, label='Power Signal', color=signal_color, alpha=0.8)
    
    # Plot final synthesized changepoints
    for i, cp in enumerate(synth_cp):
        plt.axvline(x=cp, color=synth_color, linestyle='-', linewidth=2, alpha=1.0, 
                    label='Final Changepoints' if i == 0 else "")
        
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Power (W)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    # 1. Load configuration
    config = load_config()
    appliance_name = config['workflow'].get('appliance_name', 'unknown_appliance')
    sequence_id = config['workflow'].get('sequence_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # 2. Define Run ID and Paths (GUIDELINES.md Section 2)
    # Format: {script_name}_{sequence_id}
    run_id = f"visualize_segments_{sequence_id}"
    output_root = os.path.join(project_root, "output", run_id)
    figure_dir = os.path.join(output_root, "figure")
    segments_dir = os.path.join(output_root, "segments")
    
    # Ensure directories exist
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)
    
    print(f"Starting visualization with Run ID: {run_id}")
    
    # 3. Resolve input file path
    input_file_rel = config['steps']['extract_active_data']['input_file']
    input_file = os.path.join(project_root, input_file_rel)
    
    # 4. Extract segments (Step 1 of the logic)
    print("\nStep 1: Extracting Active Segments...")
    segmenter = ApplianceDataSegmenter(
        appliance_name=appliance_name,
        power_threshold=config['steps']['extract_active_data'].get('power_threshold', 1.0),
        min_duration_seconds=config['steps']['extract_active_data'].get('min_duration_seconds', 30),
        context_seconds=config['steps']['extract_active_data'].get('context_seconds', 60)
    )
    
    segment_files = segmenter.process_dataset(input_file, segments_dir)
    print(f"Successfully extracted {len(segment_files)} segments.")
    
    # 5. Process and Visualize (Step 2 of the logic)
    # We limit to 10 segments for visualization to avoid excessive resource usage
    max_plots = 10
    target_segments = segment_files[:max_plots]
    
    print(f"\nStep 2: Generating Final Segmentation Visualizations (Top {len(target_segments)})...")
    for i, file_path in enumerate(target_segments):
        file_name = os.path.basename(file_path)
        print(f"Processing segment {i+1}/{len(target_segments)}: {file_name}")
        
        # Load power data
        df = pd.read_csv(file_path)
        if 'power' not in df.columns:
            print(f"Warning: 'power' column missing in {file_name}. Skipping.")
            continue
            
        signal = df['power'].values
        # Apply noise reduction before segmentation
        signal_cleaned = medfilt_outlier_removal(signal)
        
        # Get final synthesized changepoints using model logic
        synth_cp = run_segmentation(signal_cleaned)
        
        # Generate the plot
        output_file = os.path.join(figure_dir, f"final_segment_{file_name.replace('.csv', '.png')}")
        plot_final_segments(signal, synth_cp, output_file, f"Final Segmentation Effect - {file_name}")
        
        # Explicit memory cleanup (GUIDELINES.md Section 3)
        del df, signal, signal_cleaned, synth_cp
        gc.collect()

    print(f"\nExecution Complete! Results saved to: {figure_dir}")

if __name__ == "__main__":
    main()
