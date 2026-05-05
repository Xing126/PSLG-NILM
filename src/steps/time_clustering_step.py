import os
import sys
import time
import json
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from scipy.signal import medfilt

from src.framework.step import Step

sys.stdout.flush()
sys.stderr.flush()


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
        enable_heatmap=False
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
        self.enable_heatmap = enable_heatmap

    def load_data(self, context: dict) -> tuple:
        """Load feature data and sequence length from context or files."""
        data_np = None
        feature_matrix = None
        seq_len = None

        if self.data_path and os.path.exists(self.data_path):
            data_np = np.load(self.data_path)
        elif 'data' in context and context['data'] is not None:
            data_np = context['data']

        if self.feature_path and os.path.exists(self.feature_path):
            feature_matrix = np.load(self.feature_path)
        elif 'features' in context and context['features'] is not None:
            feature_matrix = context['features']

        if self.seq_len_path and os.path.exists(self.seq_len_path):
            seq_len = np.load(self.seq_len_path)
        elif 'seq_len' in context and context['seq_len'] is not None:
            seq_len = context['seq_len']

        if data_np is not None:
            print(f"[TimeClustering] Loaded data shape: {data_np.shape}")
        if feature_matrix is not None and feature_matrix.size > 0:
            print(f"[TimeClustering] Loaded feature matrix shape: {feature_matrix.shape}")
        if seq_len is not None:
            print(f"[TimeClustering] Loaded seq_len shape: {seq_len.shape}")

        return data_np, feature_matrix, seq_len

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
                           save_dir: str) -> dict:
        """Evaluate clustering quality with multiple metrics."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        metrics = {}
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters > 1 and n_clusters < len(labels) - 1:
            valid_mask = labels != -1
            if valid_mask.sum() > 0 and self.metric == 'precomputed':
                try:
                    sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')
                    metrics['silhouette_score'] = sil_score
                    print(f"[TimeClustering] Silhouette Score: {sil_score:.4f}")
                except Exception as e:
                    print(f"[TimeClustering] Silhouette score error: {e}")

            try:
                db_score = davies_bouldin_score(feature_matrix, labels)
                metrics['davies_bouldin_score'] = db_score
                print(f"[TimeClustering] Davies-Bouldin Score: {db_score:.4f}")
            except Exception as e:
                print(f"[TimeClustering] DB score error: {e}")

            try:
                ch_score = calinski_harabasz_score(feature_matrix, labels)
                metrics['calinski_harabasz_score'] = ch_score
                print(f"[TimeClustering] Calinski-Harabasz Score: {ch_score:.4f}")
            except Exception as e:
                print(f"[TimeClustering] CH score error: {e}")

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_distribution = {
            "noise" if label == -1 else f"cluster_{label}": int(count)
            for label, count in zip(unique_labels, counts)
        }
        metrics['cluster_distribution'] = cluster_distribution
        metrics['n_clusters'] = n_clusters
        metrics['n_noise'] = int(np.sum(labels == -1))

        json_save_path = os.path.join(save_dir, "evaluation_metrics.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[TimeClustering] Metrics saved to {json_save_path}")

        return metrics

    def save_clustering_results(self, data_np: np.ndarray, seq_len: np.ndarray,
                                 labels: np.ndarray, save_dir: str) -> str:
        """Save clustering results including labels and visualizations."""
        labels_save_path = os.path.join(save_dir, 'cluster_labels.npy')
        np.save(labels_save_path, labels)
        print(f"[TimeClustering] Labels saved to {labels_save_path}")

        data_list = []
        for i in range(min(len(labels), len(data_np))):
            data_list.append(data_np[i])

        self._save_cluster_visualization(data_list, seq_len, labels, save_dir)
        self._save_cluster_arrays(data_np, seq_len, labels, save_dir)

        print(f"[TimeClustering] Results saved to {save_dir}")
        return save_dir

    def _save_cluster_visualization(self, data_list: list, seq_len: np.ndarray,
                                    labels: np.ndarray, save_dir: str):
        """Save cluster visualization plots."""
        try:
            import matplotlib.pyplot as plt
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_cols = min(3, n_clusters + 1)
            n_rows = (n_clusters + 2) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

            colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters + 1)))

            for idx, label in enumerate(sorted(unique_labels)):
                cluster_data = [data_list[i] for i in range(len(labels)) if labels[i] == label]
                if not cluster_data:
                    continue

                ax = axes[idx] if idx < len(axes) else axes[0]
                for seq in cluster_data[:10]:
                    seq_to_plot = seq[:min(len(seq), 100)] if len(seq) > 100 else seq
                    ax.plot(seq_to_plot, alpha=0.7, label=f'Cluster {label}')

                ax.set_title(f"Cluster {label} ({len(cluster_data)} samples)")
                ax.legend()
                ax.grid(True, alpha=0.3)

            for idx in range(len(unique_labels), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cluster_visualization.png'), dpi=150)
            plt.close()
            print(f"[TimeClustering] Visualization saved")
        except Exception as e:
            print(f"[TimeClustering] Visualization save warning: {e}")

    def _save_cluster_arrays(self, data_np: np.ndarray, seq_len: np.ndarray,
                             labels: np.ndarray, save_dir: str):
        """Save cluster data as numpy arrays."""
        cluster_groups = {}
        for i in range(len(labels)):
            cluster_id = labels[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            data = data_np[i][:seq_len[i]] if seq_len is not None and i < len(seq_len) else data_np[i]
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
        save_dir = os.path.join(log_dir, 'cluster_result')
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
            save_dir
        )

        self.save_clustering_results(data_np, seq_len, labels, save_dir)

        context['cluster_labels'] = labels
        context['cluster_save_dir'] = save_dir
        context['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        context['n_noise'] = int(np.sum(labels == -1))

        print(f"[TimeClustering] Clustering workflow complete")
        return context