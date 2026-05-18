import os
import bisect
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.framework.step import Step


class ApplianceDataSegmenter:
    def __init__(
        self,
        appliance_name: str,
        power_threshold: float = 1.0,
        min_duration_seconds: int = 30,
        context_seconds: int = 120,
        alpha: float = 0.2,
        min_stop_time: int = 5,
        sample_rate: int = 10,
    ):
        self.appliance_name = appliance_name
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_seconds = context_seconds
        self.alpha = alpha
        self.min_stop_time = min_stop_time
        self.sample_rate = sample_rate

    def process_dataset(self, input_file: str, output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理电器数据: {self.appliance_name}")
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
        print(f"功率阈值: {self.power_threshold}")
        print(f"最小持续时间: {self.min_duration_seconds}秒")
        print(f"上下文: ±{self.context_seconds}秒")

        data = self._read_data(input_file)
        print(f"成功读取 {len(data)} 个数据点")

        segments = self._detect_working_segments_from_data(data)
        print(f"检测到 {len(segments)} 个潜在工作区间")

        output_files = self._extract_all_segments_from_data(data, segments, output_dir)
        print(f"处理完成！共提取 {len(output_files)} 个工作区间")
        return output_files

    def _read_data(self, input_file: str) -> List[Tuple[int, float]]:
        file_ext = os.path.splitext(input_file)[1].lower()
        print(f"检测到文件格式: {file_ext}")

        if file_ext == ".dat":
            return self._read_dat_file(input_file)
        if file_ext == ".csv":
            return self._read_csv_file(input_file)
        if file_ext == ".npy":
            return self._read_npy_file(input_file)
        raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .dat、.csv 和 .npy 格式")

    def _read_dat_file(self, input_file: str) -> List[Tuple[int, float]]:
        data: List[Tuple[int, float]] = []
        with open(input_file, "r") as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 1000000 == 0:
                    print(f"已读取 {line_count} 行数据...")

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                try:
                    timestamp = int(parts[0])
                    power = float(parts[1])
                    data.append((timestamp, power))
                except (ValueError, IndexError):
                    continue

        return data

    def _read_csv_file(self, input_file: str) -> List[Tuple[int, float]]:
        df = pd.read_csv(input_file)
        if "timestamp" in df.columns and "power" in df.columns:
            df = df[["timestamp", "power"]]
        else:
            df = df.iloc[:, [0, 1]]
            df.columns = ["timestamp", "power"]

        df = df.dropna(subset=["timestamp", "power"])
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df["power"] = df["power"].astype(np.float64)
        df = df.sort_values("timestamp")
        return list(zip(df["timestamp"].tolist(), df["power"].tolist()))

    def _read_npy_file(self, input_file: str) -> List[Tuple[int, float]]:
        data_array = np.load(input_file)
        if data_array.ndim == 2 and data_array.shape[1] >= 2:
            return [(int(row[0]), float(row[1])) for row in data_array]
        raise ValueError("NPY 文件格式不正确，需要至少两列数据")

    def _estimate_monthly_thresholds(self, monthly_series: pd.Series) -> Tuple[float, float]:
        """核心一步：动态计算单月的自适应阈值"""
        # 过滤空值
        valid_data = monthly_series.dropna().values
        if len(valid_data) < 100:  # 数据量过少，返回兜底默认值
            return self.power_threshold, self.power_threshold * 3

        # 1. 提取底噪基准（25%分位数，基本就是纯待机状态）
        bg_baseline = np.percentile(valid_data, 25)

        # 2. 采样加速聚类：每隔 sample_rate 个点抽一个，应付几百万数据轻轻松松
        sampled_data = valid_data[:: self.sample_rate].reshape(-1, 1)

        # 3. 用极速聚类区分“背景”与“工作”
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
        kmeans.fit(sampled_data)
        centers = np.sort(kmeans.cluster_centers_.flatten())

        # centers[0] 是背景中心，centers[1] 是工作中心
        bg_center, work_center = centers[0], centers[1]

        # 4. 根据两级中心，按比例动态计算阈值（可根据实际效果微调比例）
        p_low = bg_center + (work_center - bg_center) * 0.15
        p_high = bg_center + (work_center - bg_center) * 0.40

        # 安全兜底：防止某些月份洗碗机一次没开过导致聚类失效
        p_low = max(p_low, bg_baseline + 5.0)
        p_high = max(p_high, p_low + 10.0)

        return p_low, p_high

    def _extract_periods_with_thresholds(self, df_month: pd.DataFrame, p_low: float, p_high: float) -> List[Tuple[int, int, int, int, int]]:
        """利用当前月阈值执行单向状态机扫描"""
        timestamps = df_month["timestamp"].values
        powers = df_month["power"].values
        orig_indices = df_month["orig_idx"].values

        periods = []
        status = "BACKGROUND"
        ema = powers[0]
        start_idx = None
        cooldown_counter = 0

        for i in range(1, len(powers)):
            # 更新 EMA
            ema = self.alpha * powers[i] + (1 - self.alpha) * ema

            if status == "BACKGROUND":
                if ema > p_high:
                    status = "WORKING"
                    start_idx = i
                    cooldown_counter = 0
            elif status == "WORKING":
                if ema < p_low:
                    cooldown_counter += 1
                    if cooldown_counter >= self.min_stop_time:
                        status = "BACKGROUND"
                        # 回溯真正结束的时间
                        end_idx_in_month = max(0, i - self.min_stop_time)
                        
                        s_idx = orig_indices[start_idx]
                        e_idx = orig_indices[end_idx_in_month]
                        s_time = timestamps[start_idx]
                        e_time = timestamps[end_idx_in_month]
                        duration = max(0, e_time - s_time)
                        
                        if duration >= self.min_duration_seconds:
                            periods.append((int(s_idx), int(e_idx), int(s_time), int(e_time), int(duration)))
                else:
                    cooldown_counter = 0

        # 闭合月末仍在运行的区间
        if status == "WORKING":
            end_idx_in_month = len(powers) - 1
            s_idx = orig_indices[start_idx]
            e_idx = orig_indices[end_idx_in_month]
            s_time = timestamps[start_idx]
            e_time = timestamps[end_idx_in_month]
            duration = max(0, e_time - s_time)
            if duration >= self.min_duration_seconds:
                periods.append((int(s_idx), int(e_idx), int(s_time), int(e_time), int(duration)))

        return periods

    def _detect_working_segments_from_data(
        self, data: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int, int, int]]:
        if not data:
            return []

        # Convert to DataFrame for processing
        df = pd.DataFrame(data, columns=["timestamp", "power"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["orig_idx"] = np.arange(len(df))
        df.set_index("datetime", inplace=True)

        all_periods = []
        # 按月分组遍历
        grouped = df.groupby(df.index.to_period("M"))

        for month, df_month in grouped:
            if len(df_month) < 2:
                continue

            # 1. 动态计算当月阈值
            p_low, p_high = self._estimate_monthly_thresholds(df_month["power"])
            print(f"月份 {month}: 动态阈值 p_low={p_low:.2f}, p_high={p_high:.2f}")

            # 2. 用当月阈值跑状态机
            month_periods = self._extract_periods_with_thresholds(df_month, p_low, p_high)
            all_periods.extend(month_periods)

        return all_periods

    def _extract_all_segments_from_data(
        self,
        data: List[Tuple[int, float]],
        segments: List[Tuple[int, int, int, int, int]],
        output_dir: str,
    ) -> List[str]:
        output_files: List[str] = []
        for seg_idx, (start_idx, end_idx, start_time, end_time, duration) in enumerate(segments):
            output_file = self._extract_single_segment_from_data(
                data=data,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                output_dir=output_dir,
                segment_id=seg_idx,
            )
            if output_file:
                output_files.append(output_file)

            if (seg_idx + 1) % 10 == 0:
                print(f"已提取 {seg_idx + 1}/{len(segments)} 个工作区间")

        return output_files

    def _extract_single_segment_from_data(
        self,
        data: List[Tuple[int, float]],
        start_idx: int,
        end_idx: int,
        start_time: int,
        end_time: int,
        duration: int,
        output_dir: str,
        segment_id: int,
    ) -> Optional[str]:
        timestamps = [t for t, _ in data]
        context_start_time = start_time - self.context_seconds
        context_end_time = end_time + self.context_seconds
        context_start_idx = bisect.bisect_left(timestamps, context_start_time)
        context_end_idx = bisect.bisect_right(timestamps, context_end_time) - 1
        context_start_idx = max(0, context_start_idx)
        context_end_idx = min(len(data) - 1, context_end_idx)
        segment_data = data[context_start_idx:context_end_idx + 1]

        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)
        start_str = start_dt.strftime("%Y%m%d_%H%M%S")
        end_str = end_dt.strftime("%Y%m%d_%H%M%S")

        filename = f"{self.appliance_name}_{start_str}_{end_str}_{duration}s.csv"
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            return output_file

        print(f"提取区间 {segment_id}: {start_str} - {end_str}, 持续 {duration}秒")

        if not segment_data:
            return None

        df = pd.DataFrame(segment_data, columns=["timestamp", "power"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df.to_csv(output_file, index=False)
        return output_file


class ExtractActiveDataStep(Step):
    def __init__(
        self,
        name: str = "ExtractActiveData",
        appliance_name: str = "",
        input_file: str = "",
        power_threshold: float = 1.0,
        min_duration_seconds: int = 30,
        context_seconds: int = 120,
        alpha: float = 0.2,
        min_stop_time: int = 5,
        sample_rate: int = 10,
        set_input_root: bool = True,
    ):
        super().__init__(name)
        self.appliance_name = appliance_name
        self.input_file = input_file
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_seconds = context_seconds
        self.alpha = alpha
        self.min_stop_time = min_stop_time
        self.sample_rate = sample_rate
        self.set_input_root = set_input_root

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
            print("未设置 input_file，跳过 ExtractActiveData")
            return context

        log_dir = self.get_log_dir(context)
        segments_dir = os.path.join(log_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        segmenter = ApplianceDataSegmenter(
            appliance_name=self.appliance_name or "appliance",
            power_threshold=self.power_threshold,
            min_duration_seconds=self.min_duration_seconds,
            context_seconds=self.context_seconds,
            alpha=self.alpha,
            min_stop_time=self.min_stop_time,
            sample_rate=self.sample_rate,
        )
        output_files = segmenter.process_dataset(self.input_file, segments_dir)

        if "data" not in context:
            context["data"] = {}
        context["data"]["extract_active_data"] = {
            "segments_dir": segments_dir,
            "segment_files": output_files,
        }

        if self.set_input_root and output_files:
            context["input_root"] = segments_dir

        return context
