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
- `metric` (仅支持 `euclidean`)

说明：该模式强制对特征提取后的结果进行聚类扫描，使用特征向量间的欧式距离作为度量，不再支持直接的 DTW 序列扫描以提升计算效率。扫描维度是 `n_clusters`，记录每个 k 的 `SCI/DBI/CHI`。

### 3.5 hdbscan

- `min_cluster_size`
- `min_samples`（可选，默认由库内部处理）
- `cluster_selection_method`（`eom | leaf`）
- `cluster_selection_epsilon`（默认 0.0）
- `metric`（支持 `euclidean`；当 `metric=dtw/fastdtw` 时会使用预计算距离矩阵）

## 4. 输出规则（重点）

所有输出文件存放在 `log/{run_id}/TimeClustering_{suffix}/` 目录下。

### 4.1 普通模式（dbscan / kmeans / hdbscan）

该模式会保存完整的聚类结果、中间数据以及评估产物，用于后续分析或可视化。

**核心结果文件：**
- **`cluster_labels.npy`**: 长度为 N 的一维数组，存储每个输入片段的聚类标签（Cluster ID）。`-1` 表示噪声点。
- **`Cluster_{id}.npy`**: 每个聚类一个文件（如 `Cluster_0.npy`），存储属于该类别的所有原始片段波形数据（已 Padding 齐平）。
- **`{cluster_method}_{feature_model}_{segment_method}.json`**: 详细的聚类评估报告，包含超参数、类别分布、各项评分等。

**中间数据文件（供离线可视化使用）：**
- **`org_data.npy`**: 原始输入的片段波形数据。
- **`feature_matrix.npy`**: 用于聚类的特征矩阵（降维或提取后的向量）。
- **`seq_len.npy`**: 每个片段的原始有效长度。
- **`{segment_method}_{feature_model}.npy`**: 包含三个核心指标的简易数组：`[Silhouette, Davies-Bouldin, Calinski-Harabasz]`。

**可视化产物（受 `visualization_specific.enabled` 控制）：**
- **`{cluster_method}_{feature_model}_{segment_method}_center.png`**: 聚类中心波形图。
- **`{cluster_method}_{feature_model}_{segment_method}_stacked.png`**: 类别内的波形堆叠图。
- **`{cluster_method}_{feature_model}_{segment_method}_tsne.png`**: 聚类结果的 t-SNE 二维投影散点图。
- **`cluster_{id}/` 文件夹**: 包含该类别内前 N 个样本的独立波形图片（如 `item_1.png`）。

### 4.2 扫描模式（dbscan-scan / kmeans-scan）

扫描模式旨在搜索最佳超参数，因此**不会**保存具体的 `Cluster_*.npy` 分类结果。

- **扫描报告**：
  - **`{cluster_method}_{feature_model}_{segment_method}.json`**: 记录扫描过程中每个参数（如 k 或 eps）对应的评估指标。
  - **`{cluster_method}_{feature_model}_{segment_method}.png`**: 扫描指标的可视化折线图，用于辅助观察拐点。

---

## 5. 指标与 JSON 字段说明

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
