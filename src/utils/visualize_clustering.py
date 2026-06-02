import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import clustering_utils

def main():
    """
    Standalone visualization script for clustering results.
    Follows GUIDELINES.md for Run ID and path association.
    """
    # 1. Determine Run ID
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Priority 1: CLI argument
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        # Priority 2: config.yaml
        appliance_name = config['workflow'].get('appliance_name', 'unknown')
        sequence_id = config['workflow'].get('sequence_id', 'default')
        run_id = f"{appliance_name}_{sequence_id}"

    print(f"[VisualizeClustering] Using Run ID: {run_id}")

    # 2. Locate paths
    log_root = os.path.join(project_root, 'log', run_id)
    output_root = os.path.join(project_root, 'output', run_id)
    
    # Get model/method info from config
    segment_method = config['steps'].get('time_segmentation', {}).get('segment_method', 'clasp')
    feature_model = config['steps'].get('feature_extract', {}).get('model_name', 'detsec')
    cluster_config = config['steps'].get('time_clustering', {})
    cluster_method = str(cluster_config.get('cluster_method', 'dbscan')).lower()
    
    cluster_dir = os.path.join(log_root, f"TimeClustering_{cluster_method}")
    if not os.path.exists(cluster_dir):
        print(f"Error: Cluster directory not found: {cluster_dir}")
        return

    # 3. Load Metrics (naming convention: {segment_method}_{feature_model}.npy)
    metrics_file = f"{segment_method}_{feature_model}.npy"
    metrics_path = os.path.join(cluster_dir, metrics_file)
    if os.path.exists(metrics_path):
        metrics_array = np.load(metrics_path)
        print(f"[VisualizeClustering] Loaded metrics from {metrics_file}:")
        print(f"  Silhouette Score: {metrics_array[0]:.4f}")
        print(f"  Davies-Bouldin Score: {metrics_array[1]:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics_array[2]:.4f}")
    else:
        print(f"Warning: Metrics file {metrics_file} not found in {cluster_dir}")

    # 4. Load Data for Visualization
    data_loaded = False
    try:
        labels = np.load(os.path.join(cluster_dir, 'cluster_labels.npy'))
        feature_matrix = np.load(os.path.join(cluster_dir, 'feature_matrix.npy'))
        org_data = np.load(os.path.join(cluster_dir, 'org_data.npy'))
        seq_len = np.load(os.path.join(cluster_dir, 'seq_len.npy'))
        data_loaded = True
    except FileNotFoundError as e:
        print(f"[VisualizeClustering] Warning: Missing required .npy files in {cluster_dir}. "
              "Detailed cluster visualization will be skipped.")
        # Do not return, proceed to check for scan artifacts

    # 5. Perform Visualization
    figure_dir = os.path.join(output_root, 'figure')
    os.makedirs(figure_dir, exist_ok=True)
    
    if data_loaded:
        # Extract visualization params from config
        viz_config = cluster_config.get('visualization_specific', {})
        language = viz_config.get('language', 'en')
        visualize_noise = viz_config.get('visualize_noise', True)
        cluster_stack_count = viz_config.get('cluster_stack_count', 50)
        col_index = cluster_config.get('col_index', 2)
        
        # Unified distance method logic (dtw if metric is dtw)
        metric = cluster_config.get('method_specific', {}).get(cluster_method, {}).get('metric', 'euclidean')
        dist_method = 'dtw' if metric in ('dtw', 'fastdtw') else 'euclidean'

        print(f"[VisualizeClustering] Generating plots in {figure_dir}...")
        
        # Call existing utils functions
        clustering_utils.cluster_result_pic_save(
            data_array=org_data,
            seq_length=seq_len,
            cluster_result=labels,
            save_dir=figure_dir,
            threshold=200,
            col_index=col_index,
            language=language
        )
        
        valid_idx = labels != -1
        valid_labels = labels[valid_idx]
        valid_org_data = org_data[valid_idx]
        valid_feature_matrix = feature_matrix[valid_idx]
        
        clustering_utils.visualize_cluster_results(
            cluster_labels=labels,
            valid_labels=valid_labels,
            valid_org_data=valid_org_data,
            feature_matrix=feature_matrix,
            org_data=org_data,
            seq_length=seq_len,
            save_dir=figure_dir,
            dist_method=dist_method,
            col_index=col_index,
            sampling_threshold=200,
            cluster_stack_count=cluster_stack_count,
            visualize_noise=visualize_noise,
            language=language,
            show=False,
            cluster_method=cluster_method,
            feature_model=feature_model,
            segment_method=segment_method
        )

    else:
        print(f"[VisualizeClustering] Skipping detailed visualization as data files are missing.")
    
    # 6. Optional: Generate scan plots if JSON exists
    # Naming convention: {cluster_method}_{feature_model}_{segment_method}.json
    kmeans_scan_json = os.path.join(cluster_dir, f'kmeans-scan_{feature_model}_{segment_method}.json')
    if os.path.exists(kmeans_scan_json):
        import json
        with open(kmeans_scan_json, 'r') as f:
            scan_payload = json.load(f)
        clustering_utils.save_kmeans_scan_artifacts(
            scan_payload['records'],
            scan_payload['best_n_clusters'],
            cluster_dir,
            figure_dir=figure_dir,
            feature_model=feature_model,
            segment_method=segment_method
        )
    
    dbscan_scan_json = os.path.join(cluster_dir, f'dbscan-scan_{feature_model}_{segment_method}.json')
    if os.path.exists(dbscan_scan_json):
        import json
        with open(dbscan_scan_json, 'r') as f:
            scan_payload = json.load(f)
        clustering_utils.save_dbscan_scan_artifacts(
            scan_payload['records'],
            scan_payload['best_eps'],
            cluster_dir,
            figure_dir=figure_dir,
            feature_model=feature_model,
            segment_method=segment_method
        )
    
    # Also check for standard evaluation metrics JSON
    eval_metrics_json = os.path.join(cluster_dir, f'{cluster_method}_{feature_model}_{segment_method}.json')
    if os.path.exists(eval_metrics_json):
        print(f"[VisualizeClustering] Found evaluation metrics: {eval_metrics_json}")


    print(f"[VisualizeClustering] Visualization complete. Results in {figure_dir}")

if __name__ == "__main__":
    main()
