# Copyright (c) OpenMMLab. All rights reserved.
"""Canonical nuScenes dataset for semantic occupancy forecasting."""

import copy
import os
from pathlib import PurePath

import numpy as np

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import OccupancyMetrics


def _validate_future_frame_indices(frame_indices):
    if not isinstance(frame_indices, (list, tuple)) or not frame_indices:
        raise ValueError(
            'future_frame_indices must be a non-empty list or tuple.')
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in frame_indices):
        raise TypeError('future_frame_indices must contain only integers.')
    if any(index <= 0 for index in frame_indices):
        raise ValueError('future_frame_indices must contain positive values.')
    if any(current >= following
           for current, following in zip(frame_indices, frame_indices[1:])):
        raise ValueError(
            'future_frame_indices must be strictly increasing and unique.')
    return tuple(frame_indices)


def _format_horizon(frame_index):
    """Format a 2 Hz nuScenes frame offset as an exact time horizon."""
    seconds = frame_index / 2
    return f'+{seconds:g}s'


def _resolve_data_path(path, root, markers):
    """Resolve a relative or machine-specific nuScenes path under ``root``."""
    if path is None:
        return None
    path = os.path.normpath(os.fspath(path))
    root = os.path.normpath(os.fspath(root))
    parts = PurePath(path).parts
    for marker in markers:
        if marker in parts:
            return os.path.join(root, *parts[parts.index(marker):])
    if os.path.isabs(path):
        try:
            if os.path.commonpath((path, root)) == root:
                return path
        except ValueError:
            pass
        raise ValueError(
            f'Cannot relocate absolute data path without one of {markers}: '
            f'{path}')
    return os.path.join(root, path)


@DATASETS.register_module()
class NuScenesForecastOccDataset(NuScenesDataset):
    """NuScenes samples with one or more explicit future occupancy horizons."""

    def __init__(self,
                 ann_file,
                 future_frame_indices,
                 pipeline=None,
                 data_root=None,
                 occ_gt_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='CAM_FRONT',
                 stereo=False):
        self.future_frame_indices = _validate_future_frame_indices(
            future_frame_indices)
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag,
            img_info_prototype=img_info_prototype,
            multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
            ego_cam=ego_cam,
            stereo=stereo)
        self.occ_gt_root = occ_gt_root or self.data_root
        source_data_infos = self.data_infos
        self.valid_sample_indices = tuple(
            index for index in range(len(source_data_infos))
            if self._raw_sample_has_future_frames(index, source_data_infos))
        if not self.valid_sample_indices:
            raise ValueError(
                'No samples have all requested future occupancy frames.')
        self._source_data_infos = source_data_infos
        self.data_infos = [
            source_data_infos[index] for index in self.valid_sample_indices
        ]
        self._set_group_flag()

    @property
    def max_future_frame_index(self):
        return self.future_frame_indices[-1]

    def _raw_sample_has_future_frames(self, raw_index, source_data_infos):
        scene_token = source_data_infos[raw_index]['scene_token']
        return all(
            raw_index + frame_index < len(source_data_infos) and
            source_data_infos[raw_index + frame_index]['scene_token'] ==
            scene_token for frame_index in self.future_frame_indices)

    def _resolve_data_index(self, dataset_index):
        if not -len(self) <= dataset_index < len(self):
            raise IndexError(
                f'Dataset index {dataset_index} is outside a '
                f'{len(self)}-sample dataset.')
        return self.valid_sample_indices[dataset_index]

    def _get_future_infos(self, raw_index):
        return [
            self._resolve_frame_paths(copy.deepcopy(
                self._source_data_infos[raw_index + frame_index]))
            for frame_index in self.future_frame_indices
        ]

    def _resolve_frame_paths(self, info):
        if 'lidar_path' in info:
            info['lidar_path'] = _resolve_data_path(
                info['lidar_path'], self.data_root, ('samples', 'sweeps'))
        for sweep in info.get('sweeps', []):
            sweep['data_path'] = _resolve_data_path(
                sweep['data_path'], self.data_root, ('samples', 'sweeps'))
        for camera in info.get('cams', {}).values():
            camera['data_path'] = _resolve_data_path(
                camera['data_path'], self.data_root, ('samples',))
        return info

    def fix_data_path(self, input_dict):
        """Relocate a copied pipeline input without mutating annotations."""
        input_dict = copy.deepcopy(input_dict)
        input_dict['pts_filename'] = _resolve_data_path(
            input_dict['pts_filename'], self.data_root,
            ('samples', 'sweeps'))
        for sweep in input_dict.get('sweeps', []):
            sweep['data_path'] = _resolve_data_path(
                sweep['data_path'], self.data_root, ('samples', 'sweeps'))
        if 'curr' in input_dict:
            input_dict['curr'] = self._resolve_frame_paths(
                input_dict['curr'])
        if 'adjacent' in input_dict:
            input_dict['adjacent'] = [
                self._resolve_frame_paths(info)
                for info in input_dict['adjacent']
            ]
        return input_dict

    def _get_occ_paths(self, raw_index):
        raw_indices = [raw_index] + [
            raw_index + offset for offset in self.future_frame_indices]
        return [
            _resolve_data_path(
                self._source_data_infos[index]['occ_path'], self.occ_gt_root,
                ('gts',))
            for index in raw_indices
        ]

    def get_data_info(self, index):
        raw_index = self._resolve_data_index(index)
        input_dict = super().get_data_info(index)
        input_dict['future_infos'] = self._get_future_infos(raw_index)
        occ_paths = self._get_occ_paths(raw_index)
        input_dict['occ_gt_path'] = occ_paths[0]
        input_dict['future_occ_gt_path'] = occ_paths[1:]
        input_dict['future_frame_count'] = len(self.future_frame_indices)
        return input_dict

    @staticmethod
    def _load_occ_semantics(path):
        label_path = os.path.join(path, 'labels.npz')
        with np.load(label_path) as ground_truth:
            return ground_truth['semantics'].copy()

    @staticmethod
    def _validate_prediction(prediction, future_frame_count):
        if not isinstance(prediction, dict):
            raise TypeError('Each prediction must be a dictionary.')
        if 'future_occ' not in prediction:
            raise KeyError('Prediction is missing key: future_occ')
        future_predictions = prediction['future_occ']
        if not isinstance(future_predictions, (list, tuple, np.ndarray)):
            raise TypeError('future_occ must be an ordered sequence.')
        if len(future_predictions) != future_frame_count:
            raise ValueError(
                f'Expected {future_frame_count} future predictions, got '
                f'{len(future_predictions)}.')
        return future_predictions

    @staticmethod
    def _log_metrics(metrics, logger):
        if logger is None:
            return
        for name, value in metrics.items():
            logger.info('%s: %.2f', name, value)

    def evaluate(self, occ_results, metric=None, logger=None, runner=None,
                 **kwargs):
        """Evaluate future semantic occupancy predictions."""
        if metric is not None:
            metrics = [metric] if isinstance(metric, str) else metric
            if (not isinstance(metrics, (list, tuple)) or
                    list(metrics) != ['mIoU']):
                raise ValueError(
                    "ForecastOcc evaluation supports only 'mIoU'.")
        if kwargs:
            raise TypeError(
                f'Unexpected evaluation options: {sorted(kwargs)}')
        if runner is not None:
            runner_logger = getattr(runner, 'logger', None)
            if logger is not None and runner_logger is not logger:
                raise ValueError('logger and runner.logger do not match.')
            logger = runner_logger
        if not isinstance(occ_results, (list, tuple)) or not occ_results:
            raise ValueError(
                'occ_results must be a non-empty result sequence.')
        if len(occ_results) > len(self):
            raise ValueError(
                f'Got {len(occ_results)} results for a {len(self)}-sample '
                'dataset.')

        future_metrics = [OccupancyMetrics()
                          for _ in self.future_frame_indices]
        for dataset_index, prediction in enumerate(occ_results):
            raw_index = self._resolve_data_index(dataset_index)
            future_predictions = self._validate_prediction(
                prediction, len(self.future_frame_indices))
            future_paths = self._get_occ_paths(raw_index)[1:]

            for horizon_index, future_path in enumerate(future_paths):
                future_semantics = self._load_occ_semantics(future_path)
                future_metrics[horizon_index].add_batch(
                    future_predictions[horizon_index], future_semantics)

        metrics = {}
        for frame_index, accumulator in zip(
                self.future_frame_indices, future_metrics):
            metrics.update(accumulator.compute(_format_horizon(frame_index)))
        self._log_metrics(metrics, logger)
        return metrics


__all__ = ['NuScenesForecastOccDataset']
