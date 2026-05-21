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

# Add project root and models/time_segmentation to sys.path for claspy imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "models", "time_segmentation"))

# Import vendored claspy
from models.time_segmentation.claspy.segmentation import BinaryClaSPSegmentation
from src.steps.extract_active_data_step import ApplianceDataSegmenter

def get_sequence_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def medfilt_outlier_removal(series):
    """Perform outlier removal using median filter."""
    ts = np.asarray(series)
    cleaned_series = medfilt(ts, kernel_size=5)
    outlier_mask = np.zeros_like(ts, dtype=bool)
    return cleaned_series, outlier_mask

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
        return [], "None"

    if len(low_cp) >= len(high_cp):
        ref_cp = np.sort(low_cp)
        others = [np.sort(high_cp)]
        ref_name = "Low-Freq"
    else:
        ref_cp = np.sort(high_cp)
        others = [np.sort(low_cp)]
        ref_name = "High-Freq"

    if len(ref_cp) == 0:
        return [], "None"

    groups = {i: [ref_val] for i, ref_val in enumerate(ref_cp)}
    for other_list in others:
        for p in other_list:
            closest_idx = np.argmin(np.abs(ref_cp - p))
            groups[closest_idx].append(p)
            
    synthesized_cp = []
    for i in sorted(groups.keys()):
        group_mean = np.mean(groups[i])
        synthesized_cp.append(group_mean)
        
    return sorted(synthesized_cp), ref_name

def run_wavelet_analysis(signal, wavelet, is_shape_dtw=False):
    """Performs wavelet separation and segmentation logic from TimeSegmentationStep."""
    level = 2
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA2, cD2, cD1 = coeffs
    
    zeros_cD2 = np.zeros_like(cD2)
    zeros_cD1 = np.zeros_like(cD1)
    zeros_cA2 = np.zeros_like(cA2)
    
    low_freq_signal = pywt.waverec([cA2, zeros_cD2, zeros_cD1], wavelet)
    high_freq_combined = pywt.waverec([zeros_cA2, cD2, cD1], wavelet)
    
    low_freq_signal = low_freq_signal[:len(signal)]
    high_freq_combined = high_freq_combined[:len(signal)]
    
    distance = "shape_dtw" if is_shape_dtw else "znormed_euclidean_distance"
    low_cp = get_segmentation_points(low_freq_signal, distance=distance)
    high_cp = get_segmentation_points(high_freq_combined, distance="znormed_euclidean_distance")
    
    # We also need initial segmentation for original signal
    orig_cp = get_segmentation_points(signal, distance=distance)
    
    synthesized_cp, ref_name = synthesize_changepoints(low_cp, high_cp)
    
    return {
        'wavelet': wavelet,
        'low_freq_signal': low_freq_signal,
        'high_freq_combined': high_freq_combined,
        'low_cp': low_cp,
        'high_cp': high_cp,
        'orig_cp': orig_cp,
        'synthesized_cp': synthesized_cp,
        'ref_name': ref_name,
        'cleaned_signal': signal
    }

def plot_results(signal, signal_cleaned, orig_cp, results, output_dir, file_id):
    """Visual style from wavelet_transform.py."""
    wavelet = results['wavelet']
    low_freq_signal = results['low_freq_signal']
    high_freq_combined = results['high_freq_combined']
    low_cp = results['low_cp']
    high_cp = results['high_cp']
    synth_cp = results['synthesized_cp']
    ref_name = results['ref_name']
    
    plt.figure(figsize=(15, 12))
    
    # Color definitions with specified RGB values
    blue = (74/255, 75/255, 157/255)
    red = (200/255, 22/255, 29/255)
    green = (90/255, 164/255, 174/255)
    yellow = (250/255, 192/255, 61/255)
    synth_color = (166/255, 85/255, 157/255)
    cleaned_color = (204/255, 93/255, 32/255)
    
    # 1. Original Signal with Cleaned Signal
    plt.subplot(4, 1, 1)
    plt.plot(signal, label='Original Signal', color='gray', alpha=0.6)
    plt.plot(signal_cleaned, label='Cleaned Signal', color=cleaned_color, alpha=0.8)
    for cp in orig_cp:
        plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
    plt.title(f'Signal Comparison - {file_id}')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 2. Low Frequency Component
    plt.subplot(4, 1, 2)
    plt.plot(low_freq_signal, label=f'Low Frequency ({wavelet})', color=blue)
    for cp in low_cp:
        plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 3. High Frequency Component
    plt.subplot(4, 1, 3)
    plt.plot(high_freq_combined, label=f'High Frequency ({wavelet})', color=green, alpha=0.8)
    for cp in high_cp:
        plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 4. Comparison with Synthesized Changepoints
    plt.subplot(4, 1, 4)
    plt.plot(signal, label='Original Signal', color=blue, alpha=0.8)
    plt.plot(low_freq_signal, label=f'Separated Low Freq ({wavelet})', color='gray', alpha=0.5, linewidth=1.5)
    
    for i, cp in enumerate(orig_cp):
        plt.axvline(x=cp, color=green, linestyle='--', linewidth=1, alpha=1.0, label='Orig CP' if i == 0 else "")
    for i, cp in enumerate(synth_cp):
        plt.axvline(x=cp, color=synth_color, linestyle='-', linewidth=2, alpha=1.0, label=f'Synth CP ({ref_name})' if i == 0 else "")
        
    plt.title(f'Comparison: Components vs Synthesized (Wavelet: {wavelet})')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'separation_{file_id}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved separation plot: {plot_path}")

    # Heatmap
    scales = np.arange(1, 128)
    cwt_wavelet = 'cmor1.5-1.0'
    try:
        cwtmatr, freqs = pywt.cwt(low_freq_signal, scales, cwt_wavelet)
    except:
        cwt_wavelet = 'mexh'
        cwtmatr, freqs = pywt.cwt(low_freq_signal, scales, cwt_wavelet)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(low_freq_signal), scales[0], scales[-1]], 
               cmap='jet', aspect='auto', interpolation='nearest', origin='lower')
    plt.colorbar(label='Energy')
    plt.title(f'Scalogram - {file_id}')
    heatmap_path = os.path.join(output_dir, f'heatmap_{file_id}.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap plot: {heatmap_path}")

def main():
    input_file = "/home/scnu2023024258/data/code/PSLG-NILM/input/washing_machine.csv"
    appliance_name = "washing_machine"
    sequence_id = get_sequence_id()
    run_id = f"visualize_separation_{sequence_id}"
    
    # According to GUIDELINES.md: output/{run_id}/
    output_root = os.path.join(project_root, "output", run_id)
    figure_dir = os.path.join(output_root, "figure")
    segments_dir = os.path.join(output_root, "segments")
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)
    
    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_root}")
    
    # 1. Extract segments using ApplianceDataSegmenter (matching main.py logic)
    print("\n--- Step 1: Extracting Active Segments ---")
    segmenter = ApplianceDataSegmenter(
        appliance_name=appliance_name,
        power_threshold=1.0,        # Default from main.py
        min_duration_seconds=30,   # Default from main.py
        context_seconds=60        # Default from main.py
    )
    
    segment_files = segmenter.process_dataset(input_file, segments_dir)
    print(f"Extracted {len(segment_files)} segments to {segments_dir}")
    
    # 2. Process segments (limit to first 10 for visualization)
    max_plots = 10
    target_files = segment_files[:max_plots]
    
    print(f"\n--- Step 2: Running Wavelet Analysis & Visualization (Top {len(target_files)}) ---")
    for i, file_path in enumerate(target_files):
        file_name = os.path.basename(file_path)
        print(f"Processing segment {i+1}/{len(target_files)}: {file_name}")
        
        # Load the segment data
        df = pd.read_csv(file_path)
        if 'power' not in df.columns:
            print(f"Skipping {file_name}: 'power' column not found.")
            continue
            
        signal = df['power'].values
        
        # 3. Run analysis (WaveletSeparationStep logic)
        signal_cleaned, _ = medfilt_outlier_removal(signal)
        results = run_wavelet_analysis(signal_cleaned, 'db4')
        
        # 4. Plot (wavelet_transform.py style)
        plot_results(signal, signal_cleaned, results['orig_cp'], results, figure_dir, f"seg_{i+1}")
        
        del df, signal, signal_cleaned, results
        gc.collect()

    print(f"\nDone! Visualizations saved to: {figure_dir}")

def visualize_from_logs(log_dir):
    """
    Reconstructs changepoints from TimeSegmentation output files (X.npy, lengths.npy, indices.npy)
    and visualizes them on the original signals.
    """
    time_seg_dir = os.path.join(log_dir, "TimeSegmentation")
    
    # Try different possible locations for input CSVs (DataLoader or ExtractActiveData)
    possible_dataloader_dirs = [
        os.path.join(log_dir, "DataLoader"),
        os.path.join(log_dir, "ExtractActiveData", "segments"),
        os.path.join(log_dir, "ExtractActiveData")
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
    
    # Output directory for log-based visualization
    output_dir = os.path.join(log_dir, "visualize_reconstruction")
    os.makedirs(output_dir, exist_ok=True)
    
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
            # Column 0 is CSV index, Column 1 is start index
            file_mask = (indices[:, 0] == i)
            file_segment_indices = np.where(file_mask)[0]
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
            
        if not file_segment_indices:
            continue
            
        # Visualization: 4 subplots matching the separation steps
        plt.figure(figsize=(15, 12))
        
        # 1. Original Signal (from CSV)
        plt.subplot(4, 1, 1)
        plt.plot(signal, label='Original Signal', color='gray', alpha=0.7)
        for cp in internal_cps:
            plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
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
        plt.title('Cleaned Signal (Reconstructed from X.npy)')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 3. Low Frequency (from X.npy Channel 2)
        plt.subplot(4, 1, 3)
        offset = 0
        for idx in file_segment_indices:
            seg_len = lengths[idx]
            low_seg = X[idx, :seg_len, 2]
            plt.plot(np.arange(offset, offset + seg_len), low_seg, color=green)
            offset += seg_len
        plt.title('Low Frequency Component')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 4. High Frequency (from X.npy Channel 3)
        plt.subplot(4, 1, 4)
        offset = 0
        for idx in file_segment_indices:
            seg_len = lengths[idx]
            high_seg = X[idx, :seg_len, 3]
            plt.plot(np.arange(offset, offset + seg_len), high_seg, color=yellow)
            offset += seg_len
        plt.title('High Frequency Component')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'reconstruction_{i+1}.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"[{i+1}/{len(csv_files)}] Saved reconstruction plot for {csv_name}")
        
        if i >= 9: # Limit to first 10 for demonstration
            print("... (limited to first 10 files)")
            break
            
    print(f"\nDone! Log-based visualizations saved to: {output_dir}")

if __name__ == "__main__":
    # Check if a log directory or sequence_id is provided as argument
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        # Try to read sequence_id and appliance_name from config/config.yaml
        config_path = os.path.join(project_root, "config", "config.yaml")
        arg = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    if config:
                        workflow_cfg = config.get('workflow', {})
                        seq_id = workflow_cfg.get('sequence_id')
                        appliance = workflow_cfg.get('appliance_name')
                        
                        if seq_id and appliance:
                            arg = f"{appliance}_{seq_id}"
                            print(f"DEBUG: Found in config - appliance: {appliance}, seq_id: {seq_id}")
                            print(f"Using combined ID from config: {arg}")
                        elif seq_id:
                            arg = seq_id
                            print(f"DEBUG: Found only seq_id in config: {seq_id}")
                            print(f"Using sequence_id from config: {arg}")
                        else:
                            print("DEBUG: workflow.sequence_id not found in config.yaml")
                    else:
                        print("DEBUG: config.yaml is empty")
            except Exception as e:
                print(f"Error reading config: {e}")
        
        if not arg:
            print("No argument provided and no valid sequence_id found in config. Running default main().")
            main()
            sys.exit(0)

    # Process the argument (either from sys.argv or config)
    # 1. Check if it's already a valid directory path
    if os.path.isdir(arg):
        log_dir = arg
    else:
        # 2. Check if it's a sequence_id in the project's log folder
        log_dir = os.path.join(project_root, "log", arg)
        
        # 3. If not found, try to search for a folder containing the sequence_id in log/
        if not os.path.isdir(log_dir):
            log_base = os.path.join(project_root, "log")
            found = False
            if os.path.exists(log_base):
                for d in os.listdir(log_base):
                    if arg in d and os.path.isdir(os.path.join(log_base, d)):
                        log_dir = os.path.join(log_base, d)
                        found = True
                        print(f"Matched sequence_id '{arg}' to directory: {d}")
                        break
            
            if not found:
                print(f"Error: Could not find log directory for '{arg}'")
                sys.exit(1)
    
    visualize_from_logs(log_dir)
