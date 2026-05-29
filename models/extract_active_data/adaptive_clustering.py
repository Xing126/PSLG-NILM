import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import gc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import List, Dict, Any
from .base import BaseActiveDetector

class AdaptiveClusteringDetector(BaseActiveDetector):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.threshold = None
        self.t_drop = None
        self.t_min_work = None

    def _moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def train(self, raw_power_series: np.ndarray):
        print(f"[{self.name}] 开始自适应参数学习...")
        smoothed = self._moving_average(raw_power_series)
        
        # 1. Amplitude threshold
        X = smoothed.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X)
        centers = np.sort(kmeans.cluster_centers_.flatten())
        self.threshold = float((centers[0] + centers[1]) / 2)
        
        # 2. Time parameters
        binary = (smoothed > self.threshold).astype(int)
        
        # Zero segments (potential drops)
        labeled_zeros, num_zeros = ndimage.label(1 - binary)
        zero_durations = np.array([np.sum(labeled_zeros == i) / self.fs for i in range(1, num_zeros + 1)])
        
        if len(zero_durations) >= 2:
            gmm = GaussianMixture(n_components=2, random_state=42).fit(np.log1p(zero_durations).reshape(-1, 1))
            labels = gmm.predict(np.log1p(zero_durations).reshape(-1, 1))
            short_idx = np.argmin(gmm.means_.flatten())
            short_durations = zero_durations[labels == short_idx]
            self.t_drop = float(np.max(short_durations)) if len(short_durations) > 0 else 5.0
        else:
            self.t_drop = 10.0

        # One segments (potential work)
        labeled_ones, num_ones = ndimage.label(binary)
        one_durations = np.array([np.sum(labeled_ones == i) / self.fs for i in range(1, num_ones + 1)])
        
        if len(one_durations) >= 2:
            gmm = GaussianMixture(n_components=2, random_state=42).fit(np.log1p(one_durations).reshape(-1, 1))
            labels = gmm.predict(np.log1p(one_durations).reshape(-1, 1))
            noise_idx = np.argmin(gmm.means_.flatten())
            noise_durations = one_durations[labels == noise_idx]
            self.t_min_work = float(np.max(noise_durations)) if len(noise_durations) > 0 else 5.0
        else:
            self.t_min_work = 15.0

        print(f"  学习结果: threshold={self.threshold:.2f}, t_drop={self.t_drop:.2f}, t_min_work={self.t_min_work:.2f}")
        gc.collect()

    def detect(self, raw_power_series: np.ndarray, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        if self.threshold is None:
            self.train(raw_power_series)
            
        smoothed = self._moving_average(raw_power_series)
        binary = (smoothed > self.threshold).astype(int)
        
        # Closing operation
        pts_drop = int(self.t_drop * self.fs)
        struct = np.ones(max(1, pts_drop))
        closed = ndimage.binary_closing(binary, structure=struct).astype(int)
        
        # Length cleaning
        labeled, num = ndimage.label(closed)
        pts_min = int(self.t_min_work * self.fs)
        context_pts = int(self.context_seconds * self.fs)
        
        work_intervals = []
        for i in range(1, num + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) >= pts_min:
                start_idx, end_idx = indices[0], indices[-1]
                work_intervals.append(self._create_interval_dict(
                    raw_power_series, timestamps, start_idx, end_idx, context_pts
                ))
        
        gc.collect()
        return work_intervals

    def _create_interval_dict(self, powers, timestamps, start_idx, end_idx, context_pts) -> Dict[str, Any]:
        c_start = max(0, start_idx - context_pts)
        c_end = min(len(powers) - 1, end_idx + context_pts)
        return {
            "start_time": int(timestamps[start_idx]),
            "end_time": int(timestamps[end_idx]),
            "duration_sec": float(len(np.where(np.arange(len(powers)) >= start_idx)[0]) / self.fs), # Simplified
            "duration_sec": float((end_idx - start_idx + 1) / self.fs),
            "data": pd.DataFrame({
                "timestamp": timestamps[c_start:c_end + 1],
                "power": powers[c_start:c_end + 1]
            })
        }
