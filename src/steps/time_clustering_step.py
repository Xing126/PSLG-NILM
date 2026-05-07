import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from scipy.signal import medfilt

from src.framework.step import Step

sys.stdout.flush()
sys.stderr.flush()

from src.utlis import clustering_utils


class TimeClusteringStep(Step):
    """
    Step for time series clustering using DBSCAN.
    Migrated from src/time_clustering/dbscan.py
    """
    def __init__(
        self,
        name="TimeClustering",
        data_path=None,
        feature_path=None,
        seq_len_path=None,
        save_dir=None,
        col_index=2,
        eps=1.25,
        min_pts=20,
        metric='euclidean',
        normalization_method='zscore',
        cluster_method='dbscan',
        enable_visualization=True,
        appliance_name=""
    ):
        super().__init__(name)
        self.data_path = data_path
        self.feature_path = feature_path
        self.seq_len_path = seq_len_path
        self.save_dir = save_dir or os.path.join('cluster_result', 'dbscan_result')
        self.col_index = col_index
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        self.normalization_method = normalization_method
        self.cluster_method = cluster_method
        self.enable_visualization = enable_visualization
        self.appliance_name = appliance_name

    def load_data(self, context: dict) -> tuple:
        """Load feature data and sequence length from context or files."""
        data_np = None
        feature_matrix = None
        seq_len = None

        if self.data_path and os.path.exists(self.data_path):
            data_np = np.load(self.data_path)
        elif 'data' in context and context['data'] is not None:
            context_data = context['data']
            if isinstance(context_data, dict):
                # Preferred contract from WaveletSeparation: context['data']['X']
                data_np = context_data.get('X', None)
            elif isinstance(context_data, np.ndarray):
                # Backward compatibility for direct ndarray payload
                data_np = context_data

        if self.feature_path and os.path.exists(self.feature_path):
            feature_matrix = np.load(self.feature_path)
        elif 'features' in context and context['features'] is not None:
            context_features = context['features']
            if isinstance(context_features, np.ndarray):
                # New contract from FeatureExtractStep
                feature_matrix = context_features
            else:
                raise ValueError(
                    f"[TimeClustering] Invalid features type: {type(context_features).__name__}. "
                    "Expected context['features'] as numpy.ndarray."
                )

        if self.seq_len_path and os.path.exists(self.seq_len_path):
            seq_len = np.load(self.seq_len_path)
        elif 'seq_len' in context and context['seq_len'] is not None:
            seq_len = context['seq_len']
        elif 'data' in context and isinstance(context['data'], dict):
            # Preferred contract from WaveletSeparation: context['data']['lengths']
            seq_len = context['data'].get('lengths', None)

        # Normalize types
        if data_np is not None and not isinstance(data_np, np.ndarray):
            data_np = np.asarray(data_np)
        if feature_matrix is not None and not isinstance(feature_matrix, np.ndarray):
            feature_matrix = np.asarray(feature_matrix)
        if seq_len is not None and not isinstance(seq_len, np.ndarray):
            seq_len = np.asarray(seq_len)

        if data_np is not None:
            print(f"[TimeClustering] Loaded data shape: {data_np.shape}")
        if feature_matrix is not None and feature_matrix.size > 0:
            print(f"[TimeClustering] Loaded feature matrix shape: {feature_matrix.shape}")
        if seq_len is not None:
            print(f"[TimeClustering] Loaded seq_len shape: {seq_len.shape}")

        return data_np, feature_matrix, seq_len

    def _get_seq_length(self, seq_len: np.ndarray, idx: int, fallback_len: int) -> int:
        """Safely read sequence length value from seq_len with support for (n,) and (n,1)."""
        if seq_len is None or idx >= len(seq_len):
            return fallback_len

        raw_len = seq_len[idx]
        try:
            actual_len = int(np.asarray(raw_len).reshape(-1)[0])
        except Exception:
            return fallback_len

        if actual_len <= 0:
            return fallback_len
        return min(actual_len, fallback_len)

    def normalize_features(self, feature_matrix: np.ndarray) -> list:
        """Normalize feature matrix using specified method."""
        if feature_matrix is None or feature_matrix.size == 0:
            print("[TimeClustering] Warning: Empty feature matrix, skipping normalization")
            return []

        data = feature_matrix

        if self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
            print("[TimeClustering] Using Min-Max normalization")
        elif self.normalization_method == 'zscore':
            scaler = StandardScaler()
            print("[TimeClustering] Using Z-Score normalization")
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

        if data.ndim == 2:
            normalized_features = scaler.fit_transform(data)
            normalized_feature_list = [normalized_features[i] for i in range(len(normalized_features))]
        elif data.ndim == 1:
            normalized_features = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            normalized_feature_list = [[normalized_features[i]] for i in range(len(normalized_features))]
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")

        print(f"[TimeClustering] Normalization complete, valid samples: {len(normalized_feature_list)}")
        return normalized_feature_list

    def compute_distance_matrix(self, ts_list: list, metric: str = 'euclidean') -> np.ndarray:
        """Compute distance matrix for time series."""
        print(f"[TimeClustering] Computing {metric} distance matrix...")

        if metric == 'dtw' or metric == 'fastdtw':
            return self._compute_dtw_matrix(ts_list)
        else:
            distance_matrix = cdist(ts_list, ts_list, metric=metric)
            print(f"[TimeClustering] Distance matrix shape: {distance_matrix.shape}")
            return distance_matrix

    def _compute_dtw_matrix(self, ts_list: list) -> np.ndarray:
        """Compute DTW distance matrix using tslearn."""
        from tslearn.utils import to_time_series_dataset
        from tslearn.metrics import cdist_dtw

        start_time = time.time()
        X = to_time_series_dataset(ts_list)
        dist_matrix = cdist_dtw(X, n_jobs=-1)

        elapsed_time = time.time() - start_time
        print(f"[TimeClustering] DTW matrix computed in {elapsed_time:.2f}s, shape: {dist_matrix.shape}")
        return dist_matrix

    def get_distance_matrix(self, ts_list: list, metric: str = 'euclidean', cache_dir: str = None) -> np.ndarray:
        """Get distance matrix with caching support for DTW."""
        if metric in ['dtw', 'fastdtw'] and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            n_samples = len(ts_list)
            seq_len = len(ts_list[0]) if ts_list else 0
            cache_filename = f"dtw_dist_matrix_{metric}_{n_samples}_{seq_len}.npy"
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                print(f"[TimeClustering] Loading cached distance matrix from {cache_path}")
                return np.load(cache_path)

        dist_matrix = self.compute_distance_matrix(ts_list, metric)

        if metric in ['dtw', 'fastdtw'] and cache_dir:
            np.save(cache_path, dist_matrix)
            print(f"[TimeClustering] Saved distance matrix to cache: {cache_path}")

        return dist_matrix

    def run_dbscan(self, dist_matrix: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
        """Execute DBSCAN clustering."""
        print(f"[TimeClustering] Running DBSCAN with eps={eps}, min_samples={min_pts}")

        dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_pts,
            metric="precomputed"
        )
        labels = dbscan_model.fit_predict(dist_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"[TimeClustering] Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = "noise" if label == -1 else f"cluster_{label}"
            print(f"  {label_name}: {count} samples")

        return labels

    def evaluate_clustering(self, labels: np.ndarray, dist_matrix: np.ndarray,
                           org_data: np.ndarray, feature_matrix: np.ndarray,
                           seq_len: np.ndarray, save_dir: str,
                           enable_visualization: bool = False) -> dict:
        """Thin wrapper: forward to clustering_utils and assemble a lightweight metrics dict."""
        metrics = {}

        try:
            sil_score, db_score, ch_score = clustering_utils.cluster_result_quantification(
                cluster_labels=labels,
                dist_matrix=dist_matrix,
                org_data=org_data,
                feature_matrix=feature_matrix,
                save_dir=save_dir,
                seq_length=seq_len,
                col_index=self.col_index,
                visualize=enable_visualization,
                sampling_threshold=200,
            )

            if sil_score is not None:
                metrics['silhouette_score'] = float(sil_score)
                print(f"[TimeClustering] Silhouette Score: {sil_score:.4f}")
            if db_score is not None:
                metrics['davies_bouldin_score'] = float(db_score)
                print(f"[TimeClustering] Davies-Bouldin Score: {db_score:.4f}")
            if ch_score is not None:
                metrics['calinski_harabasz_score'] = float(ch_score)
                print(f"[TimeClustering] Calinski-Harabasz Score: {ch_score:.4f}")
        except Exception as e:
            print(f"[TimeClustering] Quantification warning: {e}")

        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_distribution'] = {
            "noise" if label == -1 else f"cluster_{label}": int(count)
            for label, count in zip(unique_labels, counts)
        }
        metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        metrics['n_noise'] = int(np.sum(labels == -1))
        return metrics

    def save_clustering_results(self, data_np: np.ndarray, seq_len: np.ndarray,
                                 labels: np.ndarray, save_dir: str) -> str:
        """Save clustering results including labels and visualizations using clustering_utils."""
        labels_save_path = os.path.join(save_dir, 'cluster_labels.npy')
        np.save(labels_save_path, labels)
        print(f"[TimeClustering] Labels saved to {labels_save_path}")

        # Save cluster arrays
        self._save_cluster_arrays(data_np, seq_len, labels, save_dir)

        print(f"[TimeClustering] Results saved to {save_dir}")
        return save_dir

    def _save_cluster_arrays(self, data_np: np.ndarray, seq_len: np.ndarray,
                             labels: np.ndarray, save_dir: str):
        """Save cluster data as numpy arrays."""
        cluster_groups = {}
        for i in range(len(labels)):
            cluster_id = labels[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            base_data = data_np[i]
            actual_len = self._get_seq_length(seq_len, i, len(base_data))
            data = base_data[:actual_len]
            cluster_groups[cluster_id].append(data)

        for cluster_id, cluster_data in cluster_groups.items():
            max_length = max(len(seq) for seq in cluster_data)
            sample_data = cluster_data[0]

            if len(sample_data.shape) == 1:
                feature_dim = 1
                cluster_data = [seq.reshape(-1, 1) if len(seq.shape) == 1 else seq for seq in cluster_data]
            else:
                feature_dim = sample_data.shape[1]

            n_samples = len(cluster_data)
            padded_data = np.zeros((n_samples, max_length, feature_dim))

            for i, seq in enumerate(cluster_data):
                actual_len = len(seq)
                padded_data[i, :actual_len, :] = seq

            cluster_file_path = os.path.join(save_dir, f'Cluster_{cluster_id}.npy')
            np.save(cluster_file_path, padded_data)
            print(f"[TimeClustering] Cluster_{cluster_id}.npy saved with shape {padded_data.shape}")

    def run(self, context: dict) -> dict:
        """Main execution logic for time series clustering."""
        log_dir = self.get_log_dir(context)
        save_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)

        print(f"[TimeClustering] Starting clustering workflow")
        print(f"[TimeClustering] Parameters: eps={self.eps}, min_pts={self.min_pts}, metric={self.metric}")

        data_np, feature_matrix, seq_len = self.load_data(context)

        if data_np is None or feature_matrix is None:
            print("[TimeClustering] Error: Required data not found. Please provide feature data.")
            return context

        if feature_matrix.size == 0:
            print("[TimeClustering] Error: Feature matrix is empty.")
            return context

        normalized_feature_list = self.normalize_features(feature_matrix)

        if not normalized_feature_list:
            print("[TimeClustering] Error: No valid features after normalization.")
            return context

        dist_matrix = self.get_distance_matrix(
            normalized_feature_list,
            metric=self.metric,
            cache_dir=save_dir
        )

        labels = self.run_dbscan(dist_matrix, self.eps, self.min_pts)

        self.evaluate_clustering(
            labels,
            dist_matrix,
            data_np,
            feature_matrix,
            seq_len,
            save_dir,
            enable_visualization=self.enable_visualization
        )

        self.save_clustering_results(data_np, seq_len, labels, save_dir)

        context['cluster_labels'] = labels
        context['cluster_save_dir'] = save_dir
        context['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        context['n_noise'] = int(np.sum(labels == -1))

        print(f"[TimeClustering] Clustering workflow complete")
        return context