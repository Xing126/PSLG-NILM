# TimeClustering Step 使用说明

本文档说明 `TimeClusteringStep` 的用途、参数配置、执行流程、产物结构与常见问题。

## 术语与字段约定（统一）

- `data_path`：原始样本张量路径（可选），用于保存簇样本与可视化时索引原始序列。
- `feature_path`：聚类输入特征路径（核心输入），通常来自 `FeatureExtractStep` 输出。
- `seq_len_path`：样本真实长度路径，和 `data_path` 配套。
- `context['features']`：若未配置 `feature_path`，从上下文读取聚类特征。
- `enable_visualization`：聚类评估可视化总开关（center/stack/tsne/样本图）。

## 1. 功能概述

`TimeClusteringStep` 负责对样本进行聚类，核心流程是：

1. 读取数据张量、特征矩阵、序列长度
2. 对特征做归一化
3. 基于特征计算样本间距离矩阵
4. 使用 DBSCAN 聚类
5. 输出标签、簇样本、评估指标与可视化结果

重要说明：

- 当前实现使用 `feature_matrix` 进行聚类，不是直接用原始 `data_np` 聚类。
- `data_np` 主要用于结果保存与可视化。


## 2. 输入优先级与数据契约

### 2.1 输入优先级

按下列顺序读取：

- `data_np`：
  1) `data_path`
  2) `context['data']['X']`
- `feature_matrix`：
  1) `feature_path`
  2) `context['features']`
- `seq_len`：
  1) `seq_len_path`
  2) `context['seq_len']`
  3) `context['data']['lengths']`

### 2.2 数据契约

- `feature_matrix` 需为 `numpy.ndarray`
- `data_np` 建议为 3D：`(N, T, D)`
- `seq_len` 建议为 `(N,)` 或 `(N,1)`


## 3. 配置方法

在 `config/config.yaml` 的 `steps.time_clustering` 下配置。

示例：

```yaml
steps:
  time_clustering:
    enabled: true

    # 输入
    data_path: "D:/path/to/data_fusion.npy"
    feature_path: "D:/path/to/features.npy"  # 通常来自 FeatureExtract 的 extracted_features/*.npy
    seq_len_path: "D:/path/to/seq_length.npy"

    # 聚类参数
    eps: 0.25
    min_pts: 20
    metric: "euclidean"         # euclidean / dtw / fastdtw
    normalization_method: "zscore"  # zscore / minmax
    col_index: 2

    # 可视化总开关（缺省 true）
    enable_visualization: true
```

兼容旧配置：

- 若未配置 `enable_visualization`，`main.py` 会回退读取 `enable_heatmap`
- 两者都没有时默认 `true`
- 推荐仅使用 `enable_visualization`，避免新旧键混用


## 4. main.py 参数传递

`main.py` 会把配置映射到 `TimeClusteringStep`：

- `data_path`
- `feature_path`
- `seq_len_path`
- `eps`
- `min_pts`
- `metric`
- `normalization_method`
- `col_index`
- `enable_visualization`


## 5. 运行方法

### 5.1 常规运行

```bash
python main.py --config config/config.yaml
```

### 5.2 只跑聚类步骤（建议排查时使用）

```yaml
steps:
  data_loader:
    enabled: false
  wavelet_separation:
    enabled: false
  feature_extract:
    enabled: false
  time_clustering:
    enabled: true
```


## 6. 原理解析

### 6.1 特征归一化

根据 `normalization_method`：

- `zscore`：StandardScaler
- `minmax`：MinMaxScaler

### 6.2 距离矩阵

- `euclidean`：`scipy.spatial.distance.cdist`
- `dtw` / `fastdtw`：`tslearn.metrics.cdist_dtw`
- DTW 距离矩阵支持缓存到当前步骤目录

### 6.3 DBSCAN 聚类

- 使用 `metric="precomputed"`，即直接基于距离矩阵聚类
- 参数：`eps` 与 `min_pts`

### 6.4 评估与可视化

评估与可视化由 `clustering_utils.cluster_result_quantification` 统一执行：

- 指标：
  - Silhouette
  - Davies-Bouldin
  - Calinski-Harabasz
- 结果 JSON：
  - `evaluation_metrics.json`
- 可视化（开启时）：
  - 每簇前 N 个样本图（`cluster_x/item_y.png`）
  - `cluster_center.png`
  - `clusters_stacked.png`
  - `tsne.png`


## 7. 产物结构

当前代码中，聚类产物保存在当前 Step 日志目录：

- `log/{appliance_name}_{sequence_id}/TimeClustering/`
  - `cluster_labels.npy`
  - `Cluster_-1.npy`（若存在噪声）
  - `Cluster_0.npy`, `Cluster_1.npy`, ...
  - `evaluation_metrics.json`
  - `dtw_dist_matrix_*.npy`（仅 DTW 度量时）
  - `cluster_center.png`（可视化开启时）
  - `clusters_stacked.png`（可视化开启时）
  - `tsne.png`（可视化开启时）
  - `cluster_{id}/item_*.png`（可视化开启时）

并更新 context：

- `context['cluster_labels']`
- `context['cluster_save_dir']`
- `context['n_clusters']`
- `context['n_noise']`


## 8. 常见问题

### 8.1 没有生成 center/stack/tsne 图片

常见原因：

- `enable_visualization` 实际为 `false`
- `main.py` 未正确读取配置开关

排查建议：

1. 检查 `config.yaml` 是否设置 `steps.time_clustering.enable_visualization: true`
2. 检查 `main.py` 是否将该参数传给 `TimeClusteringStep`
3. 检查运行日志中是否出现 `Quantification warning`

### 8.2 报错 features 类型不对

- 需要 `context['features']` 是 `numpy.ndarray`
- 若从文件输入，请确保 `feature_path` 指向 `.npy` 特征矩阵

### 8.3 报错 data/feature 缺失

- 至少保证能读取到 `data_np` 与 `feature_matrix`
- 若关闭了上游 Step，请在配置里显式提供对应文件路径
