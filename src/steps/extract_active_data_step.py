import os
import gc
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from src.framework.step import Step
from models.extract_active_data.simple_threshold import SimpleThresholdDetector
from models.extract_active_data.adaptive_clustering import AdaptiveClusteringDetector


class ExtractActiveDataStep(Step):
    def __init__(
        self,
        name: str = "ExtractActiveData",
        method: str = "simple",  # simple / adaptive
        appliance_name: str = "",
        input_file: str = "",
        **method_kwargs
    ):
        super().__init__(name, suffix=method.lower())
        self.method = method.lower()
        self.appliance_name = appliance_name
        self.input_file = input_file
        self.set_input_root = True  # 强制设为 True，不再从外部配置
        self.method_kwargs = method_kwargs

    def _get_detector(self, context: dict):
        app_name = self.appliance_name or context.get("appliance_name", "appliance")
        config = {
            "appliance_name": app_name,
            **self.method_kwargs
        }
        
        if self.method == "adaptive":
            return AdaptiveClusteringDetector("AdaptiveDetector", config)
        else:
            return SimpleThresholdDetector("SimpleDetector", config)

    def _read_data(self, input_file: str) -> Tuple[np.ndarray, np.ndarray]:
        file_ext = os.path.splitext(input_file)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(input_file)
            # 优先顺序: datetime (字符串/对象) > timestamp (数值)
            if 'datetime' in df.columns and 'power' in df.columns:
                return df['datetime'].values, df['power'].values
            elif 'timestamp' in df.columns and 'power' in df.columns:
                return df['timestamp'].values, df['power'].values
            else:
                # 兜底：取前两列
                return df.iloc[:, 0].values, df.iloc[:, 1].values
        elif file_ext == '.npy':
            data = np.load(input_file)
            return data[:, 0], data[:, 1]
        else:
            data = np.loadtxt(input_file)
            return data[:, 0], data[:, 1]

    def restore(self, context: dict) -> dict:
        log_dir = self.get_log_dir(context)
        segments_dir = os.path.join(log_dir, "segments")
        if not os.path.exists(segments_dir):
            return context

        segment_files = [
            os.path.join(segments_dir, f)
            for f in os.listdir(segments_dir)
            if f.lower().endswith(".csv")
        ]
        segment_files.sort()

        if "data" not in context:
            context["data"] = {}
        context["data"]["extract_active_data"] = {
            "segments_dir": segments_dir,
            "segment_files": segment_files,
        }

        if self.set_input_root and segment_files:
            context["input_root"] = segments_dir

        return context

    def run(self, context: dict) -> dict:
        if not self.input_file:
            print(f"[{self.name}] 未设置 input_file，跳过。")
            return context

        log_dir = self.get_log_dir(context)
        segments_dir = os.path.join(log_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        # 1. 加载数据
        print(f"[{self.name}] 正在加载数据: {self.input_file}")
        timestamps, powers = self._read_data(self.input_file)
        
        # 2. 获取并训练探测器
        detector = self._get_detector(context)
        detector.train(powers, timestamps)
        
        # 3. 执行检测
        print(f"[{self.name}] 使用方法 '{self.method}' 执行提取...")
        work_intervals = detector.detect(powers, timestamps)
        
        # 4. 保存结果
        output_files = []
        app_name = self.appliance_name or context.get("appliance_name", "appliance")
        
        for idx, interval in enumerate(work_intervals):
            start_dt = datetime.fromtimestamp(interval["start_time"])
            end_dt = datetime.fromtimestamp(interval["end_time"])
            filename = f"{app_name}_{start_dt.strftime('%Y%m%d_%H%M%S')}_{end_dt.strftime('%Y%m%d_%H%M%S')}_{int(interval['duration_sec'])}s.csv"
            filepath = os.path.join(segments_dir, filename)
            
            df = interval["data"]
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.to_csv(filepath, index=False)
            output_files.append(filepath)
            
            # Intermediate saving mechanism (log message for already saved files)
            if self.should_save_intermediate(idx + 1, context):
                print(f"[{self.name}] Intermediate progress: {idx + 1} segments processed and saved.")
                # Save current progress list
                checkpoint_path = os.path.join(log_dir, f'segments_checkpoint_{idx+1}.json')
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(output_files, f, indent=4)

            if (idx + 1) % 50 == 0:
                print(f"  已保存 {idx + 1}/{len(work_intervals)} 个区间")

        print(f"[{self.name}] 处理完成，共提取 {len(output_files)} 个工作区间。")

        # 5. 更新上下文
        if "data" not in context:
            context["data"] = {}
        context["data"]["extract_active_data"] = {
            "segments_dir": segments_dir,
            "segment_files": output_files,
            "method": self.method
        }

        if self.set_input_root and output_files:
            context["input_root"] = segments_dir

        # 显式回收内存
        del timestamps, powers, work_intervals, detector
        gc.collect()
        
        return context
