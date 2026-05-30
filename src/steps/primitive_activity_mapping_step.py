import json
import os
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.framework.step import Step


class PrimitiveActivityMappingStep(Step):
	"""
	Prepare inputs for primitive-to-activity mapping.

	Current scope:
	1) Traverse activity CSV folder and collect start/end timestamp from column 0.
	2) Traverse primitive npy folder and collect start/end timestamp from each primitive sequence.
	3) Match primitive ranges to activity ranges, then split activities into few-shot/non-few-shot groups.
	"""

	def __init__(
		self,
		name: str = "PrimitiveActivityMapping",
		activity_sequence_dir: Optional[str] = None,
		primitive_sequence_dir: Optional[str] = None,
		enable_tolerant_match: bool = False,
		timestamp_tolerance: float = 0.0,
		appliance_name: str = "",
	):
		super().__init__(name)
		self.activity_sequence_dir = activity_sequence_dir
		self.primitive_sequence_dir = primitive_sequence_dir
		self.enable_tolerant_match = bool(enable_tolerant_match)
		self.timestamp_tolerance = max(0.0, float(timestamp_tolerance))
		self.appliance_name = appliance_name

	def _resolve_activity_dir(self, context: Dict[str, Any]) -> str:
		if self.activity_sequence_dir:
			return self.activity_sequence_dir
		if context.get('activity_sequence_dir'):
			return str(context['activity_sequence_dir'])
		if context.get('input_root'):
			return str(context['input_root'])
		raise ValueError(
			"[PrimitiveActivityMapping] activity_sequence_dir is missing. "
			"Set it in config (steps.primitive_activity_mapping.activity_sequence_dir) "
			"or provide context['activity_sequence_dir']."
		)

	def _collect_activity_ranges(self, activity_dir: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
		activity_path = Path(activity_dir)
		if not activity_path.exists():
			raise FileNotFoundError(f"[PrimitiveActivityMapping] Activity directory not found: {activity_dir}")

		records: List[Dict[str, Any]] = []
		csv_files = sorted(activity_path.glob('*.csv'))

		for csv_file in csv_files:
			try:
				df = pd.read_csv(csv_file)
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Skip unreadable csv {csv_file.name}: {e}")
				continue

			if df.empty:
				print(f"[PrimitiveActivityMapping] Skip empty csv: {csv_file.name}")
				continue

			if df.shape[1] < 1:
				print(f"[PrimitiveActivityMapping] Skip csv with no columns: {csv_file.name}")
				continue

			start_ts = df.iloc[0, 0]
			end_ts = df.iloc[-1, 0]

			records.append(
				{
					'file_name': csv_file.name,
					'file_path': str(csv_file),
					'start_timestamp': start_ts.item() if hasattr(start_ts, 'item') else start_ts,
					'end_timestamp': end_ts.item() if hasattr(end_ts, 'item') else end_ts,
					'row_count': int(df.shape[0]),
					'col_count': int(df.shape[1]),
				}
			)

			# Intermediate saving mechanism
			if self.should_save_intermediate(len(records), context):
				print(f"[PrimitiveActivityMapping] Intermediate progress (activity) at {len(records)} files...")
				checkpoint_path = os.path.join(self.get_log_dir(context), f'activity_records_checkpoint_{len(records)}.json')
				with open(checkpoint_path, 'w', encoding='utf-8') as f:
					json.dump(records, f, indent=4)

		return records

	def _resolve_primitive_dir(self, context: Dict[str, Any]) -> str:
		if self.primitive_sequence_dir:
			return self.primitive_sequence_dir
		if context.get('primitive_sequence_dir'):
			return str(context['primitive_sequence_dir'])
		raise ValueError(
			"[PrimitiveActivityMapping] primitive_sequence_dir is missing. "
			"Set it in config (steps.primitive_activity_mapping.primitive_sequence_dir) "
			"or provide context['primitive_sequence_dir']."
		)

	def _to_numeric_ts(self, value: Any) -> Optional[float]:
		try:
			num = float(value)
			if np.isfinite(num):
				return num
		except Exception:
			pass
		return None

	def _extract_ranges_from_array(
		self, 
		arr: np.ndarray, 
		file_name: str, 
		file_path: str,
		indices: Optional[np.ndarray] = None,
		lengths: Optional[np.ndarray] = None
	) -> List[Dict[str, Any]]:
		records: List[Dict[str, Any]] = []

		def build_record(ts_values: Optional[np.ndarray], primitive_idx: int) -> Optional[Dict[str, Any]]:
			# If we have indices, we don't rely on timestamps from the array itself
			# (especially since X.npy channel 0 is original signal, not timestamps)
			if indices is not None and lengths is not None:
				return {
					'primitive_file_name': file_name,
					'primitive_file_path': file_path,
					'primitive_index': int(primitive_idx),
					'activity_csv_idx': int(indices[primitive_idx, 0]),
					'start_index_in_csv': int(indices[primitive_idx, 1]),
					'sample_length': int(lengths[primitive_idx, 0]),
					'start_timestamp': 0.0, # Placeholder, will be updated in matching
					'end_timestamp': 0.0,   # Placeholder, will be updated in matching
				}

			if ts_values is None or ts_values.ndim != 1 or ts_values.size == 0:
				return None
			valid = np.isfinite(ts_values)
			# Padding points are often zeros; keep positive finite timestamps.
			valid = valid & (ts_values > 0)
			if not np.any(valid):
				return None
			valid_idx = np.where(valid)[0]
			start_idx = int(valid_idx[0])
			end_idx = int(valid_idx[-1])
			start_ts = float(ts_values[start_idx])
			end_ts = float(ts_values[end_idx])
			if end_ts < start_ts:
				start_ts, end_ts = end_ts, start_ts
			return {
				'primitive_file_name': file_name,
				'primitive_file_path': file_path,
				'primitive_index': int(primitive_idx),
				'start_timestamp': start_ts,
				'end_timestamp': end_ts,
			}

		if arr.ndim == 3:
			for i in range(arr.shape[0]):
				sample = arr[i]
				if sample.ndim != 2 or sample.shape[1] < 1:
					continue
				
				# If we have indices, build record directly without checking timestamps in array
				if indices is not None and lengths is not None:
					record = build_record(None, i)
				else:
					record = build_record(sample[:, 0], i)
					
				if record is not None:
					records.append(record)
		elif arr.ndim == 2:
			# Fallback for 2D arrays (not produced by TimeSegmentation)
			ts_values = arr[:, 0] if arr.shape[1] >= 1 else arr.reshape(-1)
			record = build_record(ts_values, 0)
			if record is not None:
				records.append(record)
		elif arr.ndim == 1:
			record = build_record(arr, 0)
			if record is not None:
				records.append(record)

		return records

	def _collect_primitive_ranges(self, primitive_dir: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
		primitive_path = Path(primitive_dir)
		if not primitive_path.exists():
			raise FileNotFoundError(f"[PrimitiveActivityMapping] Primitive directory not found: {primitive_dir}")

		# Load indices and lengths if they exist (produced by TimeSegmentationStep)
		indices = None
		lengths = None
		indices_path = primitive_path / "indices.npy"
		lengths_path = primitive_path / "lengths.npy"
		
		if indices_path.exists():
			try:
				indices = np.load(indices_path)
				print(f"[PrimitiveActivityMapping] Loaded indices.npy from {primitive_dir}")
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Error loading indices.npy: {e}")
				
		if lengths_path.exists():
			try:
				lengths = np.load(lengths_path)
				print(f"[PrimitiveActivityMapping] Loaded lengths.npy from {primitive_dir}")
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Error loading lengths.npy: {e}")

		records: List[Dict[str, Any]] = []
		npy_files = sorted(primitive_path.glob('*.npy'))

		for npy_idx, npy_file in enumerate(npy_files):
			# Skip metadata files
			if npy_file.name in ("indices.npy", "lengths.npy"):
				continue
				
			try:
				arr = np.load(npy_file)
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Skip unreadable npy {npy_file.name}: {e}")
				continue

			# Use indices only for X.npy (main tensor)
			curr_indices = indices if npy_file.name == "X.npy" else None
			curr_lengths = lengths if npy_file.name == "X.npy" else None
			
			file_records = self._extract_ranges_from_array(
				arr, npy_file.name, str(npy_file),
				indices=curr_indices, lengths=curr_lengths
			)
			if not file_records:
				print(f"[PrimitiveActivityMapping] No valid primitive ranges from {npy_file.name}")
				continue

			records.extend(file_records)

			# Intermediate saving mechanism
			if self.should_save_intermediate(npy_idx + 1, context):
				print(f"[PrimitiveActivityMapping] Intermediate progress (primitive) at {npy_idx + 1} files...")
				checkpoint_path = os.path.join(self.get_log_dir(context), f'primitive_records_checkpoint_{npy_idx + 1}.json')
				with open(checkpoint_path, 'w', encoding='utf-8') as f:
					json.dump(records, f, indent=4)

		for i, rec in enumerate(records):
			rec['primitive_global_index'] = int(i)

		return records

	def _get_timestamps_from_csv(self, file_path: str, start_idx: int, length: int) -> tuple[Optional[float], Optional[float]]:
		"""Read start and end timestamps from a specific CSV segment."""
		try:
			# Use pandas to read only the necessary rows and first column (timestamp)
			# skiprows handles the header and rows before our segment
			# nrows handles the segment length
			df = pd.read_csv(file_path, usecols=[0], skiprows=range(1, start_idx + 1), nrows=length)
			if df.empty:
				return None, None
			
			start_ts = df.iloc[0, 0]
			end_ts = df.iloc[-1, 0]
			
			# Handle potential numpy types
			start_ts = start_ts.item() if hasattr(start_ts, 'item') else start_ts
			end_ts = end_ts.item() if hasattr(end_ts, 'item') else end_ts
			
			return float(start_ts), float(end_ts)
		except Exception as e:
			print(f"[PrimitiveActivityMapping] Error reading timestamps from {file_path}: {e}")
			return None, None

	def _match_primitive_to_activity(
		self,
		primitive_records: List[Dict[str, Any]],
		activity_records: List[Dict[str, Any]],
	) -> List[Dict[str, Any]]:
		matches: List[Dict[str, Any]] = []
		tol = float(self.timestamp_tolerance) if self.enable_tolerant_match else 0.0

		norm_activities: List[Dict[str, Any]] = []
		for a in activity_records:
			a_start = self._to_numeric_ts(a.get('start_timestamp'))
			a_end = self._to_numeric_ts(a.get('end_timestamp'))
			if a_start is None or a_end is None:
				continue
			if a_end < a_start:
				a_start, a_end = a_end, a_start
			norm_activities.append(
				{
					'file_name': a.get('file_name'),
					'file_path': a.get('file_path'),
					'start': float(a_start),
					'end': float(a_end),
					'orig_start': a.get('start_timestamp'),
					'orig_end': a.get('end_timestamp'),
				}
			)

		def choose_best_containing(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
			if not candidates:
				return None
			# Prefer the tightest containing interval.
			return min(candidates, key=lambda x: (x['end'] - x['start'], x['start']))

		def nearest_activity_and_direction(p_start: float, p_end: float) -> tuple[Optional[Dict[str, Any]], str, Optional[float]]:
			if not norm_activities:
				return None, 'no_activity', None

			best = None
			best_gap = np.inf
			best_direction = 'unmatched'

			for a in norm_activities:
				if p_end < a['start']:
					gap = float(a['start'] - p_end)
					direction = 'unmatched_left'
				elif p_start > a['end']:
					gap = float(p_start - a['end'])
					direction = 'unmatched_right'
				else:
					# Overlap but not containment.
					left_over = max(0.0, a['start'] - p_start)
					right_over = max(0.0, p_end - a['end'])
					gap = float(left_over + right_over)
					direction = 'overlap_not_contain'

				if gap < best_gap:
					best_gap = gap
					best = a
					best_direction = direction

			return best, best_direction, (None if not np.isfinite(best_gap) else float(best_gap))

		for p in primitive_records:
			# Case A: Index-based matching (High priority, from TimeSegmentation indices.npy)
			if 'activity_csv_idx' in p:
				csv_idx = p['activity_csv_idx']
				if 0 <= csv_idx < len(activity_records):
					target_a = activity_records[csv_idx]
					
					# Get precise timestamps from CSV segment
					start_idx = p.get('start_index_in_csv', 0)
					length = p.get('sample_length', 1)
					p_start, p_end = self._get_timestamps_from_csv(target_a['file_path'], start_idx, length)
					
					if p_start is not None and p_end is not None:
						matches.append({
							'primitive_global_index': p['primitive_global_index'],
							'primitive_file_name': p['primitive_file_name'],
							'primitive_index': p['primitive_index'],
							'primitive_start_timestamp': p_start,
							'primitive_end_timestamp': p_end,
							'activity_file_name': target_a['file_name'],
							'activity_file_path': target_a['file_path'],
							'activity_start_timestamp': target_a['start_timestamp'],
							'activity_end_timestamp': target_a['end_timestamp'],
							'match_type': 'index_match',
							'tolerance_used': 0.0,
						})
						continue
					else:
						print(f"[PrimitiveActivityMapping][WARN] Index match failed for primitive {p['primitive_index']} in {p['primitive_file_name']} (timestamp extraction failed)")

			# Case B: Timestamp-based matching (Fallback or for other types of primitives)
			p_start = self._to_numeric_ts(p.get('start_timestamp'))
			p_end = self._to_numeric_ts(p.get('end_timestamp'))
			if p_start is None or p_end is None:
				continue

			if p_end < p_start:
				p_start, p_end = p_end, p_start

			strict_candidates = [
				a for a in norm_activities
				if a['start'] <= p_start and p_end <= a['end']
			]
			strict_hit = choose_best_containing(strict_candidates)
			if strict_hit is not None:
				matches.append(
					{
						'primitive_global_index': p['primitive_global_index'],
						'primitive_file_name': p['primitive_file_name'],
						'primitive_index': p['primitive_index'],
						'primitive_start_timestamp': p_start,
						'primitive_end_timestamp': p_end,
						'activity_file_name': strict_hit['file_name'],
						'activity_file_path': strict_hit['file_path'],
						'activity_start_timestamp': strict_hit['orig_start'],
						'activity_end_timestamp': strict_hit['orig_end'],
						'match_type': 'contain',
						'tolerance_used': 0.0,
					}
				)
				continue

			if tol > 0:
				tolerant_candidates = [
					a for a in norm_activities
					if (a['start'] - tol) <= p_start and p_end <= (a['end'] + tol)
				]
				tolerant_hit = choose_best_containing(tolerant_candidates)
				if tolerant_hit is not None:
					matches.append(
						{
							'primitive_global_index': p['primitive_global_index'],
							'primitive_file_name': p['primitive_file_name'],
							'primitive_index': p['primitive_index'],
							'primitive_start_timestamp': p_start,
							'primitive_end_timestamp': p_end,
							'activity_file_name': tolerant_hit['file_name'],
							'activity_file_path': tolerant_hit['file_path'],
							'activity_start_timestamp': tolerant_hit['orig_start'],
							'activity_end_timestamp': tolerant_hit['orig_end'],
							'match_type': 'tolerant_contain',
							'tolerance_used': tol,
						}
					)
					continue

			nearest_a, direction, gap = nearest_activity_and_direction(p_start, p_end)
			if nearest_a is not None:
				if tol > 0:
					print(
						"[PrimitiveActivityMapping][WARN] tolerance mismatch: "
						f"primitive({p['primitive_file_name']}#{p['primitive_index']}, global={p['primitive_global_index']}, "
						f"[{p_start}, {p_end}]) vs activity({nearest_a['file_name']}, [{nearest_a['start']}, {nearest_a['end']}]), "
						f"direction={direction}, gap={gap}, tolerance={tol}"
					)
				else:
					print(
						"[PrimitiveActivityMapping][WARN] strict mismatch: "
						f"primitive({p['primitive_file_name']}#{p['primitive_index']}, global={p['primitive_global_index']}, "
						f"[{p_start}, {p_end}]) vs activity({nearest_a['file_name']}, [{nearest_a['start']}, {nearest_a['end']}]), "
						f"direction={direction}, gap={gap}"
					)

			matches.append(
				{
					'primitive_global_index': p['primitive_global_index'],
					'primitive_file_name': p['primitive_file_name'],
					'primitive_index': p['primitive_index'],
					'primitive_start_timestamp': p_start,
					'primitive_end_timestamp': p_end,
					'activity_file_name': nearest_a['file_name'] if nearest_a else None,
					'activity_file_path': nearest_a['file_path'] if nearest_a else None,
					'activity_start_timestamp': nearest_a['orig_start'] if nearest_a else None,
					'activity_end_timestamp': nearest_a['orig_end'] if nearest_a else None,
					'match_type': direction,
					'tolerance_used': tol,
					'mismatch_gap': gap,
				}
			)

		return matches

	def _build_activity_tensor(
		self,
		activity_records: List[Dict[str, Any]],
	) -> tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
		"""
		Read matched activity CSV files and stack them to a 3D tensor.

		Returns:
			tensor: np.ndarray with shape (n_samples, seq_len, feature_dim)
			seq_lens: np.ndarray with shape (n_samples,)
			feature_columns: list of aligned feature column names
			valid_records: records that are successfully converted into tensor samples
		"""
		if not activity_records:
			return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32), [], []

		feature_columns: Optional[List[str]] = None
		samples: List[np.ndarray] = []
		seq_lens: List[int] = []
		valid_records: List[Dict[str, Any]] = []

		for rec in activity_records:
			file_path = rec.get('file_path')
			if not file_path or not Path(file_path).exists():
				print(f"[PrimitiveActivityMapping] Skip missing activity csv: {file_path}")
				continue

			try:
				df = pd.read_csv(file_path)
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Skip unreadable activity csv {file_path}: {e}")
				continue

			if df.empty:
				print(f"[PrimitiveActivityMapping] Skip empty activity csv: {file_path}")
				continue

			if feature_columns is None:
				feature_columns = [str(c) for c in df.columns]
			else:
				# Align columns to a stable schema so feature_dim is consistent.
				df = df.reindex(columns=feature_columns)

			for col in df.columns:
				df[col] = pd.to_numeric(df[col], errors='coerce')

			arr = df.to_numpy(dtype=np.float32, copy=False)
			if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
				print(f"[PrimitiveActivityMapping] Skip invalid activity matrix from {file_path}")
				continue

			samples.append(arr)
			seq_lens.append(int(arr.shape[0]))
			valid_records.append(rec)

		if not samples:
			return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32), (feature_columns or []), []

		max_seq_len = int(max(seq_lens))
		feature_dim = int(samples[0].shape[1])
		tensor = np.zeros((len(samples), max_seq_len, feature_dim), dtype=np.float32)

		for i, arr in enumerate(samples):
			cur_len = int(arr.shape[0])
			tensor[i, :cur_len, :] = arr

		return tensor, np.asarray(seq_lens, dtype=np.int32), (feature_columns or []), valid_records

	def run(self, context: dict) -> dict:
		log_dir = self.get_log_dir(context)
		os.makedirs(log_dir, exist_ok=True)

		activity_dir = self._resolve_activity_dir(context)
		activity_records = self._collect_activity_ranges(activity_dir, context)
		activity_df = pd.DataFrame(activity_records)

		primitive_dir = self._resolve_primitive_dir(context)
		primitive_records = self._collect_primitive_ranges(primitive_dir, context)
		primitive_df = pd.DataFrame(primitive_records)

		match_records = self._match_primitive_to_activity(primitive_records, activity_records)
		match_df = pd.DataFrame(match_records)

		matched_activity_files = {
			rec['activity_file_name']
			for rec in match_records
			if rec.get('activity_file_name') and rec.get('match_type') in ('contain', 'tolerant_contain', 'index_match')
		}
		few_shot_activity_records = [
			rec for rec in activity_records
			if rec.get('file_name') in matched_activity_files
		]
		non_few_shot_activity_records = [
			rec for rec in activity_records
			if rec.get('file_name') not in matched_activity_files
		]

		ranges_json_path = os.path.join(log_dir, 'activity_sequence_ranges.json')
		with open(ranges_json_path, 'w', encoding='utf-8') as f:
			json.dump(activity_records, f, ensure_ascii=False, indent=2)

		primitive_ranges_json = os.path.join(log_dir, 'primitive_sequence_ranges.json')
		with open(primitive_ranges_json, 'w', encoding='utf-8') as f:
			json.dump(primitive_records, f, ensure_ascii=False, indent=2)

		mapping_json = os.path.join(log_dir, 'primitive_activity_mapping.json')
		with open(mapping_json, 'w', encoding='utf-8') as f:
			json.dump(match_records, f, ensure_ascii=False, indent=2)

		few_shot_input_count = len(few_shot_activity_records)
		non_few_shot_input_count = len(non_few_shot_activity_records)

		few_shot_tensor, few_shot_seq_lens, few_shot_feature_columns, few_shot_activity_records = self._build_activity_tensor(few_shot_activity_records)
		non_few_shot_tensor, non_few_shot_seq_lens, non_few_shot_feature_columns, non_few_shot_activity_records = self._build_activity_tensor(non_few_shot_activity_records)

		few_shot_activity_df = pd.DataFrame(few_shot_activity_records)
		non_few_shot_activity_df = pd.DataFrame(non_few_shot_activity_records)

		few_shot_json = os.path.join(log_dir, 'few_shot_activity_sequences.json')
		non_few_shot_json = os.path.join(log_dir, 'non_few_shot_activity_sequences.json')
		with open(few_shot_json, 'w', encoding='utf-8') as f:
			json.dump(few_shot_activity_records, f, ensure_ascii=False, indent=2)
		with open(non_few_shot_json, 'w', encoding='utf-8') as f:
			json.dump(non_few_shot_activity_records, f, ensure_ascii=False, indent=2)

		few_shot_tensor_npy = os.path.join(log_dir, 'few_shot_activity_tensor.npy')
		non_few_shot_tensor_npy = os.path.join(log_dir, 'non_few_shot_activity_tensor.npy')
		np.save(few_shot_tensor_npy, few_shot_tensor)
		np.save(non_few_shot_tensor_npy, non_few_shot_tensor)

		context['activity_sequence_source_dir'] = activity_dir
		context['activity_sequence_ranges_json'] = ranges_json_path

		context['primitive_sequence_source_dir'] = primitive_dir
		context['primitive_sequence_ranges_json'] = primitive_ranges_json

		context['primitive_activity_mapping_json'] = mapping_json

		context['few_shot_activity_sequences_json'] = few_shot_json
		context['few_shot_activity_seq_lens'] = few_shot_seq_lens
		context['few_shot_activity_feature_columns'] = few_shot_feature_columns
		context['few_shot_activity_tensor_npy'] = few_shot_tensor_npy

		context['non_few_shot_activity_sequences_json'] = non_few_shot_json
		context['non_few_shot_activity_seq_lens'] = non_few_shot_seq_lens
		context['non_few_shot_activity_feature_columns'] = non_few_shot_feature_columns
		context['non_few_shot_activity_tensor_npy'] = non_few_shot_tensor_npy

		print(
			f"[PrimitiveActivityMapping] Activities={len(activity_records)}, primitives={len(primitive_records)}, "
			f"few-shot activities={len(few_shot_activity_records)}/{few_shot_input_count}, "
			f"non-few-shot activities={len(non_few_shot_activity_records)}/{non_few_shot_input_count}, "
			f"few-shot tensor shape={tuple(few_shot_tensor.shape)}, non-few-shot tensor shape={tuple(non_few_shot_tensor.shape)}"
		)
		
		# Optimization: Clean up large intermediate objects
		del activity_records, activity_df, primitive_records, primitive_df
		del match_records, match_df, few_shot_activity_records, non_few_shot_activity_records
		del few_shot_tensor, non_few_shot_tensor, few_shot_activity_df, non_few_shot_activity_df
		
		# Sliding context release: Step 5 (PrimitiveActivityMapping) releases Step 3 (FeatureExtract) data
		if 'features' in context:
			print("[PrimitiveActivityMapping] Releasing Step 3 (FeatureExtract) context data: features")
			del context['features']

		gc.collect()

		return context
