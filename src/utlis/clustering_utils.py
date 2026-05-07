import datetime
import json
import os
import warnings
from typing import Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tslearn.barycenters import dtw_barycenter_averaging


def setup_chinese_font() -> bool:
	"""Configure Chinese fonts for matplotlib and return whether setup succeeded."""
	chinese_font_paths = [
		'/home/scnu2023024258/.local/share/fonts/wqy-microhei.ttc',
		'/home/scnu2023024258/.local/share/fonts/SourceHanSansSC-Regular.otf',
		'/home/scnu2023024258/.local/share/fonts/NotoSansCJKsc-Regular.otf',
	]

	for font_path in chinese_font_paths:
		if os.path.exists(font_path):
			try:
				font_prop = fm.FontProperties(fname=font_path)
				font_name = font_prop.get_name()
				plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
				plt.rcParams['axes.unicode_minus'] = False
				return True
			except Exception:
				continue

	chinese_font_names = [
		'WenQuanYi Micro Hei',
		'Source Han Sans SC',
		'Noto Sans CJK SC',
		'SimHei',
		'Microsoft YaHei',
	]
	all_fonts = [f.name for f in fm.fontManager.ttflist]
	for font in chinese_font_names:
		if font in all_fonts:
			plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
			plt.rcParams['axes.unicode_minus'] = False
			return True

	plt.rcParams['axes.unicode_minus'] = False
	warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')
	return False


def visualize_dict_data_layered(
	data_dict,
	title='Layered Visualization',
	bar_width=0.8,
	x_axis=None,
	max_labels=5,
	show=True,
):
	"""Visualize dict values as layered bar charts and return the figure."""
	if x_axis is None:
		raise ValueError('x_axis cannot be None')

	n_items = len(data_dict)
	if n_items == 0:
		return None

	if n_items <= 2:
		n_cols, n_rows = n_items, 1
	elif n_items <= 6:
		n_cols, n_rows = 2, (n_items + 1) // 2
	elif n_items <= 12:
		n_cols, n_rows = 3, (n_items + 2) // 3
	elif n_items <= 20:
		n_cols, n_rows = 4, (n_items + 3) // 4
	else:
		n_cols, n_rows = 5, (n_items + 4) // 5

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=150)
	if n_items == 1:
		axes = [axes]
	else:
		axes = axes.flatten() if hasattr(axes, 'flatten') else axes

	colors = plt.cm.tab10(np.linspace(0, 1, n_items)) if n_items <= 10 else plt.cm.hsv(np.linspace(0, 1, n_items))

	for idx, (key, value) in enumerate(data_dict.items()):
		if not isinstance(value, np.ndarray):
			continue

		ax = axes[idx]
		if len(x_axis) > len(value):
			value = np.pad(value, (0, len(x_axis) - len(value)), mode='constant', constant_values=0)
		elif len(x_axis) < len(value):
			raise ValueError(f'x_axis length ({len(x_axis)}) is less than value length ({len(value)})')

		x_pos = np.arange(len(value))
		ax.bar(x_pos, value, width=bar_width, color=colors[idx])
		ax.set_title(f'Cluster_{key}')
		ax.set_xlabel('Time')
		ax.set_ylabel('Power')

		n_ticks = len(x_pos)
		if max_labels > 0 and n_ticks > max_labels:
			indices = np.linspace(0, n_ticks - 1, max_labels, dtype=int)
			ax.set_xticks(indices)
			ax.set_xticklabels([x_axis[i] for i in indices], rotation=45, ha='right')
		else:
			ax.set_xticks(range(len(x_pos)))
			ax.set_xticklabels(x_axis, rotation=45, ha='right')
		ax.grid(True, alpha=0.3)

	if hasattr(axes, '__iter__') and len(axes) > n_items:
		for idx in range(n_items, len(axes)):
			if hasattr(axes[idx], 'set_visible'):
				axes[idx].set_visible(False)

	fig.suptitle(title, fontsize=16)
	plt.tight_layout()
	if show:
		plt.show()
	return fig


def _build_time_bins(start_datetime, end_datetime, time_gap_type='days', time_gap_value=1):
	bin_start_datetimes = []
	current_datetime = start_datetime
	if time_gap_type == 'days':
		while current_datetime <= end_datetime:
			bin_start_datetimes.append(current_datetime)
			current_datetime += datetime.timedelta(days=time_gap_value)
	elif time_gap_type == 'months':
		while current_datetime <= end_datetime:
			bin_start_datetimes.append(current_datetime)
			year = current_datetime.year
			month = current_datetime.month + time_gap_value
			while month > 12:
				month -= 12
				year += 1
			current_datetime = current_datetime.replace(year=year, month=month)
	else:
		raise ValueError("time_gap_type must be 'days' or 'months'")
	return bin_start_datetimes


def visualize_cluster_by_time_gap(
	data_mapping,
	cluster_result,
	time_gap_type='days',
	time_gap_value=1,
	max_duration=3600 * 24,
	save_json_path='./',
	show=True,
):
	"""Aggregate and visualize cluster durations over fixed time gaps."""
	if len(data_mapping) == 0 or len(cluster_result) == 0:
		raise ValueError('data_mapping and cluster_result must be non-empty')
	if len(data_mapping) != len(cluster_result):
		raise ValueError('data_mapping length must match cluster_result length')

	total_start_time = data_mapping[0]['start_timestamp']
	total_end_time = data_mapping[len(cluster_result) - 1]['end_timestamp']
	start_datetime = datetime.datetime.fromtimestamp(total_start_time)
	end_datetime = datetime.datetime.fromtimestamp(total_end_time)
	bin_start_datetimes = _build_time_bins(start_datetime, end_datetime, time_gap_type, time_gap_value)

	n_bins = len(bin_start_datetimes) - 1
	time_gap_start_datetimes = bin_start_datetimes[:n_bins]
	time_gap_end_datetimes = bin_start_datetimes[1:]
	time_gap_start_timestamps = np.array([dt.timestamp() for dt in time_gap_start_datetimes])
	time_gap_end_timestamps = np.array([dt.timestamp() for dt in time_gap_end_datetimes])
	time_gap_start_datetimes_str = np.array([dt.strftime('%y/%m/%d') for dt in time_gap_start_datetimes])
	time_gap_end_datetimes_str = np.array([dt.strftime('%y/%m/%d') for dt in time_gap_end_datetimes])

	unique_clusters = np.unique(cluster_result)
	cluster_time_stats = {cluster_id: np.zeros(n_bins) for cluster_id in unique_clusters}

	for i in range(len(cluster_result)):
		mapping_info = data_mapping[i]
		start_time = mapping_info['start_timestamp']
		end_time = mapping_info['end_timestamp']
		cluster_id = cluster_result[i]
		duration = end_time - start_time
		if duration > max_duration:
			continue

		start_dt = datetime.datetime.fromtimestamp(start_time)
		data_bin = -1
		for j in range(n_bins):
			if bin_start_datetimes[j] <= start_dt < bin_start_datetimes[j + 1]:
				data_bin = j
				break
		if 0 <= data_bin < n_bins:
			cluster_time_stats[cluster_id][data_bin] += duration

	if save_json_path:
		json_data = {}
		for cluster_id in unique_clusters:
			time_data = []
			for i in range(n_bins):
				time_data.append(
					{
						'start_timestamp': float(time_gap_start_timestamps[i]),
						'start_datetime': str(time_gap_start_datetimes_str[i]),
						'end_timestamp': float(time_gap_end_timestamps[i]),
						'end_datetime': str(time_gap_end_datetimes_str[i]),
						'interval_total_duration': float(cluster_time_stats[cluster_id][i]),
					}
				)
			json_data[str(cluster_id)] = time_data

		os.makedirs(save_json_path, exist_ok=True)
		with open(os.path.join(save_json_path, 'time_gap_data.json'), 'w', encoding='utf-8') as f:
			json.dump(json_data, f, ensure_ascii=False, indent=2)

	fig = visualize_dict_data_layered(
		cluster_time_stats,
		title='Cluster Time Statistics',
		x_axis=time_gap_start_datetimes_str,
		show=show,
	)
	return fig


def cluster_result_pic_save(data_array, seq_length, cluster_result, save_dir, threshold=200, col_index=1):
	"""Save per-series figures grouped by cluster label."""
	import shutil

	cluster_groups = {}
	for i in range(len(data_array)):
		cluster_id = cluster_result[i]
		cluster_groups.setdefault(cluster_id, []).append(i)

	for cluster_id, indices in cluster_groups.items():
		if len(indices) > threshold:
			indices = indices[:threshold]

		cluster_dir = os.path.join(save_dir, f'cluster_{cluster_id}')
		if os.path.exists(cluster_dir):
			shutil.rmtree(cluster_dir)
		os.makedirs(cluster_dir, exist_ok=True)

		for idx, data_idx in enumerate(indices):
			length = int(np.asarray(seq_length[data_idx]).reshape(-1)[0]) if len(seq_length) > data_idx else len(data_array[data_idx])
			data = data_array[data_idx][:length][:, col_index]
			plt.figure(figsize=(10, 6))
			plt.plot(data)
			plt.title(f'Cluster {cluster_id} - Item {idx + 1}')
			plt.xlabel('Time')
			plt.ylabel('Value')
			plt.savefig(os.path.join(cluster_dir, f'item_{idx + 1}.png'))
			plt.close()


def preprocess_cluster_data(cluster_labels: np.ndarray, dist_matrix: np.ndarray, org_data: np.ndarray, feature_matrix: np.ndarray):
	"""Validate dimensions and filter out noise points (-1 labels)."""
	n_samples = len(cluster_labels)
	if dist_matrix.shape != (n_samples, n_samples):
		raise ValueError(f'dist_matrix shape {dist_matrix.shape} does not match n_samples {n_samples}')
	if org_data.shape[0] != n_samples:
		raise ValueError(f'org_data rows {org_data.shape[0]} does not match n_samples {n_samples}')
	if feature_matrix.shape[0] != n_samples:
		raise ValueError(f'feature_matrix rows {feature_matrix.shape[0]} does not match n_samples {n_samples}')

	valid_idx = cluster_labels != -1
	valid_dist_matrix = dist_matrix[valid_idx][:, valid_idx]
	valid_labels = cluster_labels[valid_idx]
	valid_org_data = org_data[valid_idx]
	valid_feature_matrix = feature_matrix[valid_idx]
	n_clusters = len(np.unique(valid_labels))
	return valid_idx, valid_dist_matrix, valid_labels, valid_org_data, valid_feature_matrix, n_clusters


def calculate_cluster_metrics(valid_dist_matrix: np.ndarray, valid_labels: np.ndarray, valid_feature_matrix: np.ndarray, cluster_labels: np.ndarray):
	"""Compute silhouette, DB, and CH metrics."""
	n_clusters = len(np.unique(valid_labels))
	if n_clusters < 2:
		return None, None, None

	sil_score = silhouette_score(valid_dist_matrix, valid_labels, metric='precomputed')
	db_score = davies_bouldin_score(valid_feature_matrix, valid_labels)
	ch_score = calinski_harabasz_score(valid_feature_matrix, valid_labels)
	return sil_score, db_score, ch_score


def _tsne_perplexity(n_samples: int) -> int:
	if n_samples < 5:
		return 2
	return max(2, min(30, n_samples // 10))


def visualize_cluster_results(
	cluster_labels: np.ndarray,
	valid_labels: np.ndarray,
	valid_org_data: np.ndarray,
	feature_matrix: np.ndarray,
	org_data: np.ndarray,
	save_dir: Optional[str] = None,
	dist_method: str = 'dtw',
	col_index: int = 1,
	sampling_threshold: int = 200,
	visualize_noise: int = 2,
	show: bool = True,
) -> None:
	"""Render cluster center, stacked series, and tSNE visualizations."""
	valid_org_data = valid_org_data[:, :, col_index]
	org_data = org_data[:, :, col_index]
	n_clusters = len(np.unique(valid_labels))

	plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
	plt.rcParams['axes.unicode_minus'] = False
	cluster_colors = plt.cm.tab10(np.arange(n_clusters + 1))

	figsize_height = max(8, n_clusters * 2)
	fig, axes = plt.subplots(n_clusters, 1, figsize=(12, figsize_height))
	fig.suptitle('时序聚类-各簇中心轮廓分布图 (DTW重心)', fontsize=14, fontweight='bold')
	axes = [axes] if n_clusters == 1 else axes.flatten()

	for i, cluster_id in enumerate(np.unique(valid_labels)):
		cluster_seq = valid_org_data[valid_labels == cluster_id]
		if len(cluster_seq) > sampling_threshold:
			np.random.seed(42)
			sampled_indices = np.random.choice(len(cluster_seq), size=sampling_threshold, replace=False)
			cluster_seq = cluster_seq[sampled_indices]

		if len(cluster_seq) > 0:
			if dist_method == 'dtw':
				cluster_center = dtw_barycenter_averaging(cluster_seq)
			else:
				min_len = min(len(s) for s in cluster_seq)
				cluster_center = np.mean([s[:min_len] for s in cluster_seq], axis=0)

			axes[i].plot(
				cluster_center,
				color=cluster_colors[cluster_id % 10],
				linewidth=2.5,
				label=f'簇 {cluster_id} (样本数:{len(valid_org_data[valid_labels == cluster_id])})',
			)
			axes[i].set_title(f'簇 {cluster_id} 中心轮廓', fontsize=12)
			axes[i].set_xlabel('时间步 / 序列长度', fontsize=10)
			axes[i].set_ylabel('时序数值', fontsize=10)
			axes[i].legend(fontsize=9)
			axes[i].grid(alpha=0.3, linestyle='--')

	plt.tight_layout()
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'cluster_center.png'), dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	total_plots = n_clusters + 1 if visualize_noise > 0 else n_clusters
	n_cols = min(3, total_plots)
	n_rows = (total_plots + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
	fig.suptitle('各簇数据堆叠可视化（含噪声点）', fontsize=16, fontweight='bold')
	axes = [axes] if total_plots == 1 else axes.flatten()

	for i, cluster_id in enumerate(np.unique(valid_labels)):
		cluster_seq = valid_org_data[valid_labels == cluster_id]
		cluster_seq_subset = cluster_seq[:sampling_threshold]
		for j, series in enumerate(cluster_seq_subset):
			axes[i].plot(series, alpha=0.6, label=f'Series {j}' if j < 3 else '')
		axes[i].set_title(f'Cluster {cluster_id} (前{len(cluster_seq_subset)}个数据)', fontsize=12)
		axes[i].set_xlabel('时间步 / 序列长度', fontsize=10)
		axes[i].set_ylabel('时序数值', fontsize=10)
		axes[i].grid(alpha=0.3, linestyle='--')

	if visualize_noise > 0:
		noise_idx = cluster_labels == -1
		noise_ax = axes[n_clusters]
		noise_seq = org_data[noise_idx]
		noise_seq_subset = noise_seq[:sampling_threshold]
		if len(noise_seq_subset) > 0:
			if visualize_noise == 2:
				for j, series in enumerate(noise_seq_subset):
					noise_ax.plot(series, alpha=0.6, color='gray', label=f'Noise {j}' if j < 3 else '')
				noise_ax.set_title(f'Noise Points (-1) (前{len(noise_seq_subset)}个数据)', fontsize=12)
			else:
				for j, series in enumerate(noise_seq_subset):
					noise_ax.plot(series, alpha=0.6, label=f'Series {j}' if j < 3 else '')
				noise_ax.set_title(f'Cluster {n_clusters} (前{len(noise_seq_subset)}个数据)', fontsize=12)
			noise_ax.set_xlabel('时间步 / 序列长度', fontsize=10)
			noise_ax.set_ylabel('时序数值', fontsize=10)
			noise_ax.grid(alpha=0.3, linestyle='--')

	for j in range(total_plots, len(axes)):
		axes[j].set_visible(False)

	plt.tight_layout()
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'clusters_stacked.png'), dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	tsne = TSNE(
		n_components=2,
		perplexity=_tsne_perplexity(len(feature_matrix)),
		random_state=42,
		n_jobs=-1,
		init='pca',
	)
	tsne_2d = tsne.fit_transform(feature_matrix)

	plt.figure(figsize=(10, 8))
	for cluster_id in np.unique(valid_labels):
		idx = (cluster_labels == cluster_id) & (cluster_labels != -1)
		plt.scatter(
			tsne_2d[idx, 0],
			tsne_2d[idx, 1],
			c=[cluster_colors[cluster_id % 10]],
			label=f'簇 {cluster_id}',
			s=70,
			alpha=0.8,
			edgecolors='white',
			linewidth=0.5,
		)

	noise_idx = cluster_labels == -1
	if np.sum(noise_idx) > 0 and visualize_noise > 0:
		if visualize_noise == 2:
			plt.scatter(tsne_2d[noise_idx, 0], tsne_2d[noise_idx, 1], c='black', marker='x', label='噪声点', s=90, alpha=0.8)
		else:
			plt.scatter(
				tsne_2d[noise_idx, 0],
				tsne_2d[noise_idx, 1],
				c=[cluster_colors[n_clusters % 10]],
				label=f'簇 {n_clusters}',
				s=70,
				alpha=0.8,
				edgecolors='white',
				linewidth=0.5,
			)

	plt.title('时序聚类-tSNE降维分布图 (特征矩阵)', fontsize=14, fontweight='bold')
	plt.xlabel('tSNE维度1', fontsize=11)
	plt.ylabel('tSNE维度2', fontsize=11)
	plt.legend(fontsize=10, loc='best')
	plt.grid(alpha=0.2, linestyle='--')
	plt.tight_layout()
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'tsne.png'), dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()


def cluster_result_quantification(
	cluster_labels: np.ndarray,
	dist_matrix: np.ndarray,
	org_data: np.ndarray,
	feature_matrix: np.ndarray,
	save_dir: Optional[str] = None,
	seq_length: Optional[np.ndarray] = None,
	col_index: int = 1,
	visualize: bool = True,
	visualize_noise: int = 2,
	sampling_threshold: int = 200,
):
	"""Unified entry for clustering quantification, metrics persistence and visualization."""
	(
		valid_idx,
		valid_dist_matrix,
		valid_labels,
		valid_org_data,
		valid_feature_matrix,
		n_clusters,
	) = preprocess_cluster_data(
		cluster_labels=cluster_labels,
		dist_matrix=dist_matrix,
		org_data=org_data,
		feature_matrix=feature_matrix,
	)

	sil_score, db_score, ch_score = calculate_cluster_metrics(
		valid_dist_matrix=valid_dist_matrix,
		valid_labels=valid_labels,
		valid_feature_matrix=valid_feature_matrix,
		cluster_labels=cluster_labels,
	)

	metrics = {
		'cluster_distribution': {
			('noise' if int(label) == -1 else f'cluster_{int(label)}'): int(count)
			for label, count in zip(*np.unique(cluster_labels, return_counts=True))
		},
		'n_clusters': int(n_clusters),
		'n_noise': int(np.sum(cluster_labels == -1)),
	}
	if sil_score is not None:
		metrics['silhouette_score'] = float(sil_score)
	if db_score is not None:
		metrics['davies_bouldin_score'] = float(db_score)
	if ch_score is not None:
		metrics['calinski_harabasz_score'] = float(ch_score)

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
		with open(metrics_path, 'w', encoding='utf-8') as f:
			json.dump(metrics, f, ensure_ascii=False, indent=2)

	if visualize:
		if seq_length is None:
			seq_length = np.full((org_data.shape[0],), org_data.shape[1], dtype=np.int32)

		cluster_result_pic_save(
			data_array=org_data,
			seq_length=seq_length,
			cluster_result=cluster_labels,
			save_dir=save_dir if save_dir else './',
			threshold=sampling_threshold,
			col_index=col_index,
		)

		visualize_cluster_results(
			cluster_labels=cluster_labels,
			valid_labels=valid_labels,
			valid_org_data=valid_org_data,
			feature_matrix=feature_matrix,
			org_data=org_data,
			save_dir=save_dir,
			col_index=col_index,
			sampling_threshold=sampling_threshold,
			visualize_noise=visualize_noise,
			show=False,
		)

	return sil_score, db_score, ch_score
