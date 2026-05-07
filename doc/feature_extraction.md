# FeatureExtract Step 使用说明

本文档说明 `FeatureExtractStep` 的作用、配置方式、运行流程和产物结构。

## 术语与字段约定（统一）

- `data_path`：样本张量文件路径（通常是 `X` 张量），用于当前 Step 的外部输入。
- `seq_len_path`：样本真实长度文件路径，和 `data_path` 配套使用。
- `feature_path`：特征文件路径，属于 `TimeClusteringStep`，不是本 Step 的配置键。
- `context['data']['X']`：上游步骤传递的原始样本张量。
- `context['features']`：本 Step 输出的特征矩阵，下游聚类直接读取。

## 1. 功能概述

`FeatureExtractStep` 用于将时序样本张量 `X` 提取为低维特征向量 `features`，供后续聚类步骤直接使用。

当前支持的模型：

- `lstm_ae`
- `bilstm_ae`
- `bilstm_ae_attention`

输出契约：

- `context['features']` 为 `numpy.ndarray`
- `context['feature_extract_config']` 保存本次提取配置和训练摘要


## 2. 输入优先级与数据契约

`FeatureExtractStep` 的输入优先级如下：

1. `data_path` / `seq_len_path`（配置了文件路径时）
2. `context['data']['X']`（来自上游 Step，如 WaveletSeparation）
3. 若都没有，抛错退出

### 2.1 外部文件输入契约

- `data_path` 必须是 3D 张量：`(num_samples, timesteps, dims)`
- `seq_len_path` 可选：
	- 若提供，会读取真实长度
	- 若不提供，会自动用固定长度 `timesteps` 生成

### 2.2 Context 输入契约

- 需要 `context['data']['X']`
- 可选 `context['data']['lengths']`


## 3. 配置方法

在 `config/config.yaml` 的 `steps.feature_extract` 下配置。

示例：

```yaml
steps:
  feature_extract:
    enabled: true

    # 数据输入（本 Step 专用）
    data_path: "D:/path/to/data_fusion.npy"
    seq_len_path: "D:/path/to/seq_length_fusion.npy"

    # 模型与训练参数
    model_name: "bilstm_ae"
    latent_dim: 64
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    patience: 5
    attention_size: 32
```

说明：

- `enabled=false` 时整个 Step 跳过
- `model_name` 取值需与实现保持一致
- `attention_size` 主要用于 `bilstm_ae_attention`
- 不要在本 Step 配置 `feature_path`，该字段属于 `time_clustering`


## 4. main.py 参数传递

`main.py` 会从 `steps.feature_extract` 读取参数并构造 `FeatureExtractStep`。

关键映射关系：

- `data_path -> FeatureExtractStep(data_path=...)`
- `seq_len_path -> FeatureExtractStep(seq_len_path=...)`
- 其余训练参数按同名映射


## 5. 运行方法

### 5.1 使用配置文件运行

```bash
python main.py --config config/config.yaml
```

### 5.2 只跑特征提取的建议开关

为减少干扰，建议把无关 Step 先关掉：

```yaml
steps:
	data_loader:
		enabled: false
	wavelet_separation:
		enabled: false
	feature_extract:
		enabled: true
	time_clustering:
		enabled: false
```


## 6. 原理解析

`FeatureExtractStep` 的核心流程：

1. 读取输入张量 `X`
2. 根据 `model_name` 训练自编码器
3. 从编码器隐空间导出特征向量 `features`
4. 将特征写入 `context['features']`

### 6.1 为什么用自编码器做特征提取

- 原始时序维度高，直接聚类容易受噪声和尺度影响
- 自编码器通过重构任务学习低维表示，通常更适合作为聚类输入

### 6.2 三种模型差异（简述）

- `lstm_ae`：单向时序编码，结构简单
- `bilstm_ae`：双向编码，能利用前后文时序信息
- `bilstm_ae_attention`：引入注意力机制，强调关键时间片


## 7. 产物与日志

`FeatureExtractStep` 的日志目录：

- `log/{appliance_name}_{sequence_id}/FeatureExtract/`
	- 或无 appliance_name 时：`log/{sequence_id}/FeatureExtract/`

典型产物：

- `training_history/*.png`：训练曲线图
- `training_history/*.json`：训练历史
- `extracted_features/*.npy`：提取特征

上下文产物：

- `context['features']`
- `context['feature_extract_config']`


## 8. 常见问题

### 8.1 报错：No data source available

原因：

- 未配置 `data_path`
- 且 `context['data']['X']` 不存在

排查：

1. 确认 `feature_extract.enabled=true`
2. 确认 `data_path` 文件存在且可读
3. 确认 `data_path` 内容是 3D 张量

### 8.2 报错：Invalid input shape

原因：

- `data_path` 不是 3D 张量

建议：

- 先用 `np.load(path).shape` 检查，目标应为 `(N, T, D)`

### 8.3 想让下游聚类直接用该特征

确保后续 `TimeClusteringStep` 的 `feature_path` 指向本 Step 生成的 `*.npy`，或者沿用 `context['features']`。

