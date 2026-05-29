import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .base import BaseActiveDetector

class SimpleThresholdDetector(BaseActiveDetector):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.power_threshold = self.config.get("power_threshold", 1.0)
        self.min_duration_seconds = self.config.get("min_duration_seconds", 30)

    def train(self, raw_power_series: np.ndarray):
        """简单阈值模型无需训练，参数已在 config 中给出"""
        print(f"[{self.name}] 使用固定阈值: {self.power_threshold} W")
        pass

    def detect(self, raw_power_series: np.ndarray, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        """基于硬阈值的区间提取"""
        work_intervals = []
        current_start_idx = None
        consecutive_count = 0
        
        fs = self.fs
        min_pts = int(self.min_duration_seconds * fs)
        context_pts = int(self.context_seconds * fs)

        for i, power in enumerate(raw_power_series):
            if power >= self.power_threshold:
                if current_start_idx is None:
                    current_start_idx = i
                consecutive_count += 1
            else:
                if current_start_idx is not None and consecutive_count >= min_pts:
                    end_idx = i - 1
                    work_intervals.append(self._create_interval_dict(
                        raw_power_series, timestamps, current_start_idx, end_idx, context_pts
                    ))
                current_start_idx = None
                consecutive_count = 0
        
        # 处理末尾
        if current_start_idx is not None and consecutive_count >= min_pts:
            end_idx = len(raw_power_series) - 1
            work_intervals.append(self._create_interval_dict(
                raw_power_series, timestamps, current_start_idx, end_idx, context_pts
            ))
            
        return work_intervals

    def _create_interval_dict(self, powers, timestamps, start_idx, end_idx, context_pts) -> Dict[str, Any]:
        c_start = max(0, start_idx - context_pts)
        c_end = min(len(powers) - 1, end_idx + context_pts)
        
        return {
            "start_time": int(timestamps[start_idx]),
            "end_time": int(timestamps[end_idx]),
            "duration_sec": float((end_idx - start_idx + 1) / self.fs),
            "data": pd.DataFrame({
                "timestamp": timestamps[c_start:c_end + 1],
                "power": powers[c_start:c_end + 1]
            })
        }
