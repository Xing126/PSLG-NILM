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
from src.steps.extract_active_data_step import ExtractActiveDataStep
from src.steps.wavelet_separation import WaveletSeparationStep
from src.steps.feature_extract_step import FeatureExtractStep
from src.steps.time_clustering_step import TimeClusteringStep


def run_workflow(config_path: str, resume: bool = False, sequence_id: str | None = None):
    """
    Main function to run the workflow based on config file.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize workflow with name from config
    workflow_name = config['workflow'].get('name', 'ML_Workflow')
    appliance_name = config['workflow'].get('appliance_name', '')
    resume_cfg = bool(config.get('workflow', {}).get('resume', False))
    sequence_id_cfg = config.get('workflow', {}).get('sequence_id', None)
    wf = Workflow(
        workflow_name,
        sequence_id=sequence_id or sequence_id_cfg,
        resume=resume or resume_cfg,
    )

    # Add steps to the workflow sequentially based on enabled flag in config
    extract_active_cfg = config["steps"].get("extract_active_data", {})
    if extract_active_cfg.get("enabled", False):
        wf.add_step(
            ExtractActiveDataStep(
                name="ExtractActiveData",
                appliance_name=appliance_name,
                input_file=extract_active_cfg.get("input_file", ""),
                power_threshold=extract_active_cfg.get("power_threshold", 1.0),
                min_duration_seconds=extract_active_cfg.get("min_duration_seconds", 30),
                context_seconds=extract_active_cfg.get("context_seconds", 120),
                set_input_root=extract_active_cfg.get("set_input_root", True),
            )
        )

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
        wf.add_step(TimeClusteringStep(
            name="TimeClustering",
            data_path=cluster_config.get('data_path'),
            feature_path=cluster_config.get('feature_path'),
            seq_len_path=cluster_config.get('seq_len_path'),
            eps=cluster_config.get('eps', 1.25),
            min_pts=cluster_config.get('min_pts', 20),
            metric=cluster_config.get('metric', 'euclidean'),
            normalization_method=cluster_config.get('normalization_method', 'zscore'),
            col_index=cluster_config.get('col_index', 2),
            enable_visualization=cluster_config.get('enable_visualization', cluster_config.get('enable_heatmap', True)),
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
    parser.add_argument("--resume", action="store_true", help="Resume from cached step outputs if available.")
    parser.add_argument("--sequence-id", type=str, default=None, help="Specify existing sequence_id for resume.")
    args = parser.parse_args()

    # Create sample data if requested
    if args.sample:
        create_sample_data()

    # Run workflow
    run_workflow(args.config, resume=args.resume, sequence_id=args.sequence_id)
