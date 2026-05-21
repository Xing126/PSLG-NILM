# TimeSegmentationStep 步骤说明文档

`TimeSegmentationStep` 是工作流中的核心信号处理步骤，负责对原始功率信号进行离散小波分解（DWT）、自动切分以及张量化处理，为后续深度学习模型提供规整的输入数据。

该步骤支持多种切分算法，包括基于形状特征的 `Clasp` 和基于矩阵轮廓（Matrix Profile）的 `FLUSS`。

---

## 1. 输入 (Input)

该步骤从上一个步骤（通常是 `ExtractActiveData`）的缓存目录中读取数据：
- **来源路径**: `log/{sequence_id}/ExtractActiveData/segments/*.csv`
- **数据格式**: 
    - 必须包含 `power` 列（原始功率采样值）。
    - 可选包含 `timestamp` 列（用于绘图和标签生成）。

---

## 2. 核心处理逻辑 (Processing)

1.  **异常值去除**: 使用中值滤波（Median Filter, kernel_size=5）对原始信号进行平滑处理，生成“清洗后波形”。
2.  **小波分解与重构**:
    - 使用离散小波变换（默认 `db4` 小波，2 层分解）。
    - 重构出**低频分量（Low-frequency）**：反映信号的稳态趋势（如电器运行的平均功率）。
    - 重构出**高频分量（High-frequency）**：反映信号的瞬态特征（如电器开关瞬间的冲击电流）。
3.  **自动切分 (Segmentation)**:
    - 根据配置选择切分算法：
        - **Clasp**: 利用 `BinaryClaSPSegmentation` 算法分别在低频和高频分量上寻找变化点。
        - **FLUSS**: 基于 `stumpy` 库的矩阵轮廓算法，通过弧曲线（Arc Curve）检测状态切换点。
        - **ESPRESSO**: 基于 `pyeda` 的逻辑化简算法，用于处理逻辑化简需求（实验性集成）。
    - 合并各组件的切分点，生成最终的合成切分点 `synthesized_cp`。
4.  **张量化与填充 (Tensorization & Padding)**:
    - 根据合成切分点将长信号切割成多个独立的样本片段（Samples）。
    - 识别批次中最长片段的长度 $T_{max}$。
    - 对所有较短片段进行**零填充（Zero-padding）**，使所有样本具有统一的形状。

---

## 3. 输出 (Output)

### 3.1 持久化文件 (Log Files)

输出文件保存在 `log/{sequence_id}/TimeSegmentation/` 目录下：

-   **`X.npy`**: 
    - **形状**: `(n_samples, timestamp, 4)`
    - **特征通道定义**:
        - **Channel 0**: 原始波形 (Original Signal)
        - **Channel 1**: 清洗后波形 (Cleaned Signal)
        - **Channel 2**: 低频重构波形 (Low-frequency)
        - **Channel 3**: 高频重构波形 (High-frequency)
-   **`lengths.npy`**: 
    - **形状**: `(n_samples, 1)`
    - **含义**: 记录每个样本在填充前的真实长度，用于后续模型处理变长序列。

### 3.2 上下文传递 (Context)

该步骤会将处理后的张量直接存入 `context['data']`，供下游 Step（如 `FeatureExtract`）直接使用：
```python
context['data'] = {
    'X': np_array,       # (n_samples, timestamp, 4)
    'lengths': np_array  # (n_samples, 1)
}
```

---

## 4. 配置参数 (Configuration)

在 [config.yaml](file:///home/scnu2023024258/data/code/PSLG-NILM/config/config.yaml) 的 `time_segmentation` 节点下进行设定：

- **`enabled`**: 步骤开关。
- **`segment_method`**: 
    - **含义**: 选择切分算法。
    - **取值**: `clasp` (默认) / `fluss` / `espresso`。
- **FLUSS 专用参数**:
    - **`window_size`**: 子序列窗口大小。
    - **`n_regimes`**: 期望分割的段数。
    - **`excl_factor`**: 排除区域因子。

---

## 5. 验证工具 (Verification Tool)

为了确保输出的 `.npy` 张量符合深度学习模型的输入要求，可以使用项目根目录下的 `check_outputs.py` 进行检测。
