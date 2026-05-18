import os
import gc
from datetime import datetime
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from src.framework.step import Step


class ApplianceDataSegmenter:
    def __init__(self, appliance_name: str, power_threshold: float = 1.0,
                 min_duration_seconds: int = 30,
                 context_seconds: int = 120):
        """
        电器数据切割器

        Args:
            appliance_name: 电器名称，用于文件命名
            power_threshold: 功率阈值，用于检测工作状态开始和结束
            min_duration_seconds: 最小持续时间(秒)，避免噪声误判
            context_seconds: 上下文时间(秒)，在工作区间前后额外包含的数据
        """
        self.appliance_name = appliance_name
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_seconds = context_seconds

    def process_dataset(self, input_file: str, output_dir: str) -> List[str]:
        """
        处理数据集并分割工作区间

        Args:
            input_file: 输入文件路径，支持 .dat、.csv 和 .npy 格式
            output_dir: 输出目录

        Returns:
            生成的CSV文件路径列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理电器数据: {self.appliance_name}")
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
        print(f"功率阈值: {self.power_threshold}")
        print(f"最小持续时间: {self.min_duration_seconds}秒")
        print(f"上下文: ±{self.context_seconds}秒")
        print("-" * 50)

        # 第一步：读取数据
        print("第一步：读取数据...")
        data = self._read_data(input_file)
        print(f"成功读取 {len(data)} 个数据点")

        # 第二步：检测所有工作区间
        print("第二步：检测工作区间...")
        segments = self._detect_working_segments_from_data(data)

        print(f"检测到 {len(segments)} 个潜在工作区间")

        # 第三步：提取并保存工作区间数据
        print("第三步：提取并保存工作区间数据...")
        output_files = self._extract_all_segments_from_data(data, segments, output_dir)

        print(f"处理完成！共提取 {len(output_files)} 个工作区间")
        return output_files

    def _read_data(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取不同格式的数据文件

        Args:
            input_file: 输入文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        file_ext = os.path.splitext(input_file)[1].lower()
        
        print(f"检测到文件格式: {file_ext}")
        
        if file_ext == '.dat':
            return self._read_dat_file(input_file)
        elif file_ext == '.csv':
            return self._read_csv_file(input_file)
        elif file_ext == '.npy':
            return self._read_npy_file(input_file)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .dat、.csv 和 .npy 格式")

    def _read_dat_file(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取 .dat 文件

        Args:
            input_file: 输入 .dat 文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        data = []
        
        with open(input_file, 'r') as f:
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
        """
        读取 .csv 文件

        Args:
            input_file: 输入 .csv 文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        try:
            df = pd.read_csv(input_file)
            # 确保列名正确
            if 'timestamp' in df.columns and 'power' in df.columns:
                data = [(int(row['timestamp']), float(row['power'])) for _, row in df.iterrows()]
            else:
                # 假设第一列是 timestamp，第二列是 power
                data = [(int(row[0]), float(row[1])) for _, row in df.iterrows()]
            return data
        except Exception as e:
            raise ValueError(f"读取 CSV 文件失败: {e}")

    def _read_npy_file(self, input_file: str) -> List[Tuple[int, float]]:
        """
        读取 .npy 文件

        Args:
            input_file: 输入 .npy 文件路径

        Returns:
            数据列表，每个元素是 (timestamp, power) 元组
        """
        try:
            data_array = np.load(input_file)
            # 确保数据格式正确
            if data_array.ndim == 2 and data_array.shape[1] >= 2:
                data = [(int(row[0]), float(row[1])) for row in data_array]
                return data
            else:
                raise ValueError(f"NPY 文件格式不正确，需要至少两列数据")
        except Exception as e:
            raise ValueError(f"读取 NPY 文件失败: {e}")

    def _detect_working_segments_from_data(self, data: List[Tuple[int, float]]) -> List[Tuple[int, int, int, int, int]]:
        """
        从数据列表中检测所有工作区间

        Args:
            data: 数据列表，每个元素是 (timestamp, power) 元组

        Returns:
            列表格式: [(start_idx, end_idx, start_time, end_time, duration), ...]
        """
        segments = []
        current_segment_start = None
        current_segment_start_time = None
        consecutive_above_count = 0

        print("正在检测工作区间...")

        for i, (timestamp, power) in enumerate(data):
            if i % 1000000 == 0:
                print(f"已处理 {i} 个数据点...")

            if power >= self.power_threshold:
                if current_segment_start is None:
                    current_segment_start = i  # 0-based index
                    current_segment_start_time = timestamp
                consecutive_above_count += 1
            else:
                # 检查是否结束了一个有效的工作区间
                if (current_segment_start is not None and
                        consecutive_above_count >= self.min_duration_seconds):
                    segment_end = i - 1  # 上一个数据点是结束点
                    segment_end_time = timestamp
                    duration = consecutive_above_count
                    segments.append((
                        current_segment_start,
                        segment_end,
                        current_segment_start_time,
                        segment_end_time,
                        duration
                    ))

                current_segment_start = None
                current_segment_start_time = None
                consecutive_above_count = 0

        # 处理数据末尾可能的工作区间
        if (current_segment_start is not None and
                consecutive_above_count >= self.min_duration_seconds):
            segment_end = len(data) - 1
            segment_end_time = data[-1][0]
            duration = consecutive_above_count
            segments.append((
                current_segment_start,
                segment_end,
                current_segment_start_time,
                segment_end_time,
                duration
            ))

        print(f"检测完成，共处理 {len(data)} 个数据点")
        return segments

    def _extract_all_segments_from_data(self, data: List[Tuple[int, float]], segments: List[Tuple[int, int, int, int, int]],
                                      output_dir: str) -> List[str]:
        """从数据列表中提取所有检测到的工作区间数据"""
        output_files = []

        for seg_idx, (start_idx, end_idx, start_time, end_time, duration) in enumerate(segments):
            try:
                output_file = self._extract_single_segment_from_data(
                    data, start_idx, end_idx, start_time, end_time, duration,
                    output_dir, seg_idx
                )
                if output_file:
                    output_files.append(output_file)

                if (seg_idx + 1) % 10 == 0:
                    print(f"已提取 {seg_idx + 1}/{len(segments)} 个工作区间")

            except Exception as e:
                print(f"提取区间 {seg_idx} 时出错: {e}")
                continue

        return output_files

    def _extract_single_segment_from_data(self, data: List[Tuple[int, float]], start_idx: int, end_idx: int,
                                        start_time: int, end_time: int, duration: int,
                                        output_dir: str, segment_id: int) -> Optional[str]:
        """从数据列表中提取单个工作区间数据（包含上下文）"""
        # 计算包含上下文的边界
        context_start_idx = max(0, start_idx - self.context_seconds)
        context_end_idx = min(len(data) - 1, end_idx + self.context_seconds)

        # 提取数据
        segment_data = data[context_start_idx:context_end_idx + 1]

        # 生成文件名
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)

        # 格式化时间字符串，用于文件名
        start_str = start_dt.strftime("%Y%m%d_%H%M%S")
        end_str = end_dt.strftime("%Y%m%d_%H%M%S")

        # 创建文件名：{电器名称_起始时间_结束时间_持续时长}
        filename = f"{self.appliance_name}_{start_str}_{end_str}_{duration}s.csv"
        output_file = os.path.join(output_dir, filename)

        # 如果文件已存在，跳过
        if os.path.exists(output_file):
            return output_file

        print(f"提取区间 {segment_id}: {start_str} - {end_str}, 持续 {duration}秒")

        # 转换为DataFrame并保存
        if segment_data:
            df = pd.DataFrame(segment_data, columns=['timestamp', 'power'])

            # 添加可读时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            # 保存CSV文件
            df.to_csv(output_file, index=False)

            return output_file

        return None


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
            appliance_name=self.appliance_name or context.get("appliance_name", "appliance"),
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
