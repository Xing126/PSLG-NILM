import yaml
import os
import argparse
import sys
import numpy as np

# Add project root and models to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from src.framework.workflow import Workflow
from src.steps.data_loader import DataLoaderStep
from src.steps.wavelet_separation import WaveletSeparationStep
from src.steps.feature_extract_step import FeatureExtractStep
from src.steps.time_clustering_step import TimeClusteringStep


def run_workflow(config_path: str):
    """
    Main function to run the workflow based on config file.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize workflow with name from config
    workflow_name = config['workflow'].get('name', 'ML_Workflow')
    appliance_name = config['workflow'].get('appliance_name', '')
    wf = Workflow(workflow_name, appliance_name=appliance_name)

    # Add steps to the workflow sequentially based on enabled flag in config
    if config['steps']['data_loader'].get('enabled', True):
        wf.add_step(DataLoaderStep("DataLoader", appliance_name=appliance_name))

    if config['steps'].get('wavelet_separation', {}).get('enabled', True):
        is_shape_dtw = config['steps']['wavelet_separation'].get('is_shape_dtw', False)
        plot_count = config['steps']['wavelet_separation'].get('plot_count', 0)
        wf.add_step(WaveletSeparationStep("WaveletSeparation", is_shape_dtw=is_shape_dtw, plot_count=plot_count, appliance_name=appliance_name))

    if config['steps'].get('feature_extract', {}).get('enabled', True):
        extract_config = config['steps']['feature_extract']
        wf.add_step(FeatureExtractStep(
            name="FeatureExtract",
            model_name=extract_config.get('model_name', 'bilstm_ae'),
            latent_dim=extract_config.get('latent_dim', 64),
            epochs=extract_config.get('epochs', 50),
            batch_size=extract_config.get('batch_size', 32),
            learning_rate=extract_config.get('learning_rate', 0.001),
            patience=extract_config.get('patience', 5),
            attention_size=extract_config.get('attention_size', 32),
            data_path=extract_config.get('data_path', ''),
            seq_len_path=extract_config.get('seq_len_path', ''),
            appliance_name=appliance_name
        ))
    
    if config['steps'].get('time_clustering', {}).get('enabled', True):
        cluster_config = config['steps']['time_clustering']
        cluster_method = str(cluster_config.get('cluster_method', 'dbscan')).lower()
        method_specific = cluster_config.get('method_specific', {}) or {}
        method_config = method_specific.get(cluster_method, {}) if isinstance(method_specific, dict) else {}
        visualization_specific = cluster_config.get('visualization_specific', {}) or {}

        # Backward compatibility: keep supporting old flat keys if method_specific is absent.
        dbscan_eps = method_config.get('eps', cluster_config.get('eps', 1.25))
        dbscan_min_eps = method_config.get('min_eps', cluster_config.get('min_eps'))
        dbscan_max_eps = method_config.get('max_eps', cluster_config.get('max_eps'))
        dbscan_eps_gap = method_config.get('eps_gap', cluster_config.get('eps_gap', 0.1))
        dbscan_min_pts = method_config.get('min_pts', cluster_config.get('min_pts', 20))
        metric = method_config.get('metric', cluster_config.get('metric', 'euclidean'))
        kmeans_n_clusters = method_config.get('n_clusters', cluster_config.get('kmeans_n_clusters', 8))
        min_cluster = method_config.get('min_cluster', cluster_config.get('min_cluster', 2))
        max_cluster = method_config.get('max_cluster', cluster_config.get('max_cluster', 10))
        kmeans_random_state = method_config.get('random_state', cluster_config.get('kmeans_random_state', 42))
        kmeans_n_init = method_config.get('n_init', cluster_config.get('kmeans_n_init', 10))
        kmeans_max_iter = method_config.get('max_iter', cluster_config.get('kmeans_max_iter', 300))

        enable_visualization = visualization_specific.get(
            'enabled',
            cluster_config.get('enable_visualization', cluster_config.get('enable_heatmap', True))
        )
        visualize_noise = visualization_specific.get(
            'visualize_noise',
            cluster_config.get('visualize_noise', cluster_config.get('visualize_noise_minus1', True))
        )
        visualization_language = visualization_specific.get(
            'language',
            cluster_config.get('visualization_language', 'en')
        )
        cluster_stack_count = visualization_specific.get('cluster_stack_count', 50)

        if cluster_method in ('dbscan-scan', 'dbscan_scan'):
            if dbscan_min_eps is None or dbscan_max_eps is None:
                raise ValueError(
                    "[Config] For cluster_method='dbscan-scan', both min_eps and max_eps must be provided."
                )

        wf.add_step(TimeClusteringStep(
            name="TimeClustering",
            data_path=cluster_config.get('data_path'),
            feature_path=cluster_config.get('feature_path'),
            seq_len_path=cluster_config.get('seq_len_path'),
            cluster_method=cluster_method,
            eps=dbscan_eps,
            min_eps=dbscan_min_eps,
            max_eps=dbscan_max_eps,
            eps_gap=dbscan_eps_gap,
            min_pts=dbscan_min_pts,
            kmeans_n_clusters=kmeans_n_clusters,
            min_cluster=min_cluster,
            max_cluster=max_cluster,
            kmeans_random_state=kmeans_random_state,
            kmeans_n_init=kmeans_n_init,
            kmeans_max_iter=kmeans_max_iter,
            metric=metric,
            normalization_method=cluster_config.get('normalization_method', 'zscore'),
            col_index=cluster_config.get('col_index', 2),
            enable_visualization=enable_visualization,
            visualize_noise=visualize_noise,
            visualization_language=visualization_language,
            cluster_stack_count=cluster_stack_count,
            appliance_name=appliance_name
        ))

    # Run the workflow
    wf.run()

def create_sample_data():
    """Helper to create some sample data in the input folder."""
    input_dir = 'input'
    os.makedirs(input_dir, exist_ok=True)
    
    # Create sample .csv file for wavelet separation
    import pandas as pd
    sample_csv_data = {
        'timestamp': range(100),
        'power': np.random.randn(100).cumsum() # Random walk to simulate power signal
    }
    pd.DataFrame(sample_csv_data).to_csv(os.path.join(input_dir, 'Kettle_sample.csv'), index=False)
    
    print(f"Sample data created in {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML Workflow framework.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file.")
    parser.add_argument("--sample", action="store_true", help="Create sample data in input folder.")
    args = parser.parse_args()

    # Create sample data if requested
    if args.sample:
        create_sample_data()

    # Run workflow
    run_workflow(args.config)
