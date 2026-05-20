# TimeClustering Step 使用说明（精简版）

本文档仅说明当前代码已实现能力。

## 1. 功能概览

`TimeClusteringStep` 支持 5 种模式：

- `dbscan`
- `dbscan-scan`
- `kmeans`
- `kmeans-scan`
- `hdbscan`

输入仍以特征矩阵 `feature_path`（或 `context['features']`）为聚类主输入；`data_path` 主要用于可视化与样本导出。

## 2. 配置要点

配置入口：`config/config.yaml -> steps.time_clustering`

- `cluster_method`:
  `dbscan | dbscan-scan | kmeans | kmeans-scan | hdbscan`
- `method_specific`: 按方法分组参数
- `visualization_specific`: 可视化二级配置
  - `enabled`
  - `visualize_noise`
  - `language`
  - `cluster_stack_count`

示例（扫描模式）：

```yaml
steps:
  time_clustering:
    cluster_method: "dbscan-scan"
    method_specific:
      dbscan-scan:
        min_eps: 0.02
        max_eps: 2.0
        eps_gap: 0.02
        min_pts: 20
        metric: "euclidean"
    visualization_specific:
      enabled: true
      visualize_noise: true
      language: "en"
      cluster_stack_count: 50
```

## 3. 各模式参数

### 3.1 dbscan

- `eps`
- `min_pts`
- `metric` (`euclidean | dtw | fastdtw`)

### 3.2 dbscan-scan

- `min_eps`（必填）
- `max_eps`（必填）
- `eps_gap`（默认可配，> 0）
- `min_pts`（固定值，不扫描）
- `metric`

说明：扫描维度是 `eps`，会记录每个 eps 对应的 `SCI/DBI/CHI/n_noise/n_clusters`。

### 3.3 kmeans

- `n_clusters`
- `random_state`
- `n_init`
- `max_iter`

### 3.4 kmeans-scan

- `min_cluster`
- `max_cluster`
- `random_state`
- `n_init`
- `max_iter`

说明：扫描维度是 `n_clusters`，记录每个 k 的 `SCI/DBI/CHI`。

### 3.5 hdbscan

- `min_cluster_size`
- `min_samples`（可选，默认由库内部处理）
- `cluster_selection_method`（`eom | leaf`）
- `cluster_selection_epsilon`（默认 0.0）
- `metric`（支持 `euclidean`；当 `metric=dtw/fastdtw` 时会使用预计算距离矩阵）

## 4. 输出规则（重点）

### 4.1 普通模式（dbscan / kmeans / hdbscan）

会保存聚类结果与评估产物：

- `cluster_labels.npy`
- `Cluster_*.npy`
- `evaluation_metrics.json`
- 可视化图（center/stack/tsne 等，受 `visualization_specific.enabled` 控制）

### 4.2 扫描模式（dbscan-scan / kmeans-scan）

只输出扫描结果，不保存“最佳聚类结果”文件。

- `dbscan-scan`：
  - `dbscan_scan_metrics.json`
  - `dbscan_scan_metrics.png`
- `kmeans-scan`：
  - `kmeans_scan_metrics.json`
  - `kmeans_scan_metrics.png`

即不会落 `cluster_labels.npy` / `Cluster_*.npy`。

## 5. 指标与 JSON

`evaluation_metrics.json`（普通模式）中包含：

- `clustering_method`
- `clustering_hyperparameters`
- `distance_method_for_quantification`
- `cluster_distribution`
- `n_clusters`
- `n_noise`
- `silhouette_score`（可选）
- `davies_bouldin_score`（可选）
- `calinski_harabasz_score`（可选）

## 6. 约束与常见问题

- `col_index` 会在运行前校验，必须在 `0..(dim-1)`。
- `dbscan-scan` 下必须提供 `min_eps` 和 `max_eps`。
- 当某次扫描仅得到 1 个有效簇时，`SCI/DBI/CHI` 可能为 `null`（这是预期行为）。
- DTW 距离矩阵仅在 `metric=dtw/fastdtw` 时缓存。
