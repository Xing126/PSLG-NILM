import os
import bisect
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from src.framework.step import Step


class ApplianceDataSegmenter:
    def __init__(
        self,
        appliance_name: str,
        power_threshold: float = 1.0,
        min_duration_seconds: int = 30,
        context_seconds: int = 120,
    ):
        self.appliance_name = appliance_name
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_seconds = context_seconds

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

    def _detect_working_segments_from_data(
        self, data: List[Tuple[int, float]]
    ) -> List[Tuple[int, int, int, int, int]]:
        segments: List[Tuple[int, int, int, int, int]] = []
        current_segment_start: Optional[int] = None

        for i, (timestamp, power) in enumerate(data):
            if i % 1000000 == 0:
                print(f"已处理 {i} 个数据点...")

            if power >= self.power_threshold:
                if current_segment_start is None:
                    current_segment_start = i
                continue

            if current_segment_start is not None:
                segment_end = i - 1
                start_time = data[current_segment_start][0]
                end_time = data[segment_end][0]
                duration_seconds = max(0, end_time - start_time)
                if duration_seconds >= self.min_duration_seconds:
                    segments.append((current_segment_start, segment_end, start_time, end_time, duration_seconds))
                current_segment_start = None

        if current_segment_start is not None:
            segment_end = len(data) - 1
            start_time = data[current_segment_start][0]
            end_time = data[segment_end][0]
            duration_seconds = max(0, end_time - start_time)
            if duration_seconds >= self.min_duration_seconds:
                segments.append((current_segment_start, segment_end, start_time, end_time, duration_seconds))

        return segments

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
        set_input_root: bool = True,
    ):
        super().__init__(name)
        self.appliance_name = appliance_name
        self.input_file = input_file
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_seconds = context_seconds
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
