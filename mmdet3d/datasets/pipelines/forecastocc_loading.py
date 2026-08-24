# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import torch
from pyquaternion import Quaternion

from ..builder import PIPELINES
from .loading import LoadPointsFromFile


def _require_future_infos(results):
    future_infos = results.get('future_infos')
    if not isinstance(future_infos, list) or not future_infos:
        raise ValueError('future_infos must be a non-empty list')
    return future_infos


@PIPELINES.register_module()
class LoadForecastOccGT:
    """Load current and future ForecastOcc voxel labels."""

    @staticmethod
    def _load_semantics(occ_gt_path):
        with np.load(os.path.join(occ_gt_path, 'labels.npz')) as labels:
            return labels['semantics'].copy()

    def __call__(self, results):
        results['voxel_semantics'] = self._load_semantics(
            results['occ_gt_path'])

        future_paths = results.get('future_occ_gt_path')
        if not isinstance(future_paths, list) or not future_paths:
            raise ValueError('future_occ_gt_path must be a non-empty list')

        future_infos = _require_future_infos(results)
        if len(future_paths) != len(future_infos):
            raise ValueError('future_occ_gt_path and future_infos must have '
                             'the same length')

        results['future_voxel_semantics'] = [
            self._load_semantics(path) for path in future_paths
        ]
        return results


@PIPELINES.register_module()
class LoadForecastOccPoints(LoadPointsFromFile):
    """Load current and future LiDAR points for ForecastOcc."""

    def __call__(self, results):
        super().__call__(results)
        future_infos = _require_future_infos(results)
        results['future_points'] = [
            self._load_sample(info['lidar_path']) for info in future_infos
        ]
        return results


@PIPELINES.register_module()
class GenerateForecastOccDepthTargets:
    """Project current and future LiDAR points into camera depth maps."""

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    @staticmethod
    def _rotation_to_matrix(rotation):
        return Quaternion(rotation).rotation_matrix

    def _points_to_depth_map(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coordinates = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept = ((coordinates[:, 0] >= 0)
                & (coordinates[:, 0] < width)
                & (coordinates[:, 1] >= 0)
                & (coordinates[:, 1] < height)
                & (depth < self.grid_config['depth'][1])
                & (depth >= self.grid_config['depth'][0]))
        coordinates, depth = coordinates[kept], depth[kept]

        ranks = coordinates[:, 0] + coordinates[:, 1] * width
        order = (ranks + depth / 100.).argsort()
        coordinates, depth, ranks = (coordinates[order], depth[order],
                                     ranks[order])
        nearest = torch.ones(
            coordinates.shape[0], device=coordinates.device, dtype=torch.bool)
        nearest[1:] = ranks[1:] != ranks[:-1]
        coordinates, depth = coordinates[nearest], depth[nearest]
        coordinates = coordinates.to(torch.long)
        depth_map[coordinates[:, 1], coordinates[:, 0]] = depth
        return depth_map

    def _load_sample(self, points_lidar, sample_info, cam_names, imgs, intrins,
                     post_rots, post_trans):
        depth_maps = []
        for camera_index, camera_name in enumerate(cam_names):
            lidar_to_lidar_ego = np.eye(4, dtype=np.float32)
            lidar_to_lidar_ego[:3, :3] = self._rotation_to_matrix(
                sample_info['lidar2ego_rotation'])
            lidar_to_lidar_ego[:3, 3] = sample_info[
                'lidar2ego_translation']
            lidar_to_lidar_ego = torch.from_numpy(lidar_to_lidar_ego)

            lidar_ego_to_global = np.eye(4, dtype=np.float32)
            lidar_ego_to_global[:3, :3] = self._rotation_to_matrix(
                sample_info['ego2global_rotation'])
            lidar_ego_to_global[:3, 3] = sample_info[
                'ego2global_translation']
            lidar_ego_to_global = torch.from_numpy(lidar_ego_to_global)

            camera_info = sample_info['cams'][camera_name]
            camera_to_camera_ego = np.eye(4, dtype=np.float32)
            camera_to_camera_ego[:3, :3] = self._rotation_to_matrix(
                camera_info['sensor2ego_rotation'])
            camera_to_camera_ego[:3, 3] = camera_info[
                'sensor2ego_translation']
            camera_to_camera_ego = torch.from_numpy(camera_to_camera_ego)

            camera_ego_to_global = np.eye(4, dtype=np.float32)
            camera_ego_to_global[:3, :3] = self._rotation_to_matrix(
                camera_info['ego2global_rotation'])
            camera_ego_to_global[:3, 3] = camera_info[
                'ego2global_translation']
            camera_ego_to_global = torch.from_numpy(camera_ego_to_global)

            camera_to_image = torch.eye(4, dtype=torch.float32)
            camera_to_image[:3, :3] = intrins[camera_index]
            lidar_to_camera = torch.inverse(
                camera_ego_to_global.matmul(camera_to_camera_ego)).matmul(
                    lidar_ego_to_global.matmul(lidar_to_lidar_ego))
            lidar_to_image = camera_to_image.matmul(lidar_to_camera)

            image_points = points_lidar.tensor[:, :3].matmul(
                lidar_to_image[:3, :3].T)
            image_points += lidar_to_image[:3, 3].unsqueeze(0)
            image_points = torch.cat([
                image_points[:, :2] / image_points[:, 2:3],
                image_points[:, 2:3]
            ], 1)
            image_points = image_points.matmul(
                post_rots[camera_index].T)
            image_points += post_trans[camera_index:camera_index + 1, :]
            depth_maps.append(
                self._points_to_depth_map(image_points, imgs.shape[2],
                                          imgs.shape[3]))
        return torch.stack(depth_maps)

    def __call__(self, results):
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:6]
        camera_count = len(results['cam_names'])
        results['gt_depth'] = self._load_sample(
            results['points'], results['curr'], results['cam_names'], imgs,
            intrins[:camera_count], post_rots[:camera_count],
            post_trans[:camera_count])

        future_infos = _require_future_infos(results)
        future_points = results.get('future_points')
        if not isinstance(future_points, list):
            raise ValueError('future_points must be a list')
        if len(future_points) != len(future_infos):
            raise ValueError('future_points and future_infos must have the '
                             'same length')

        future_start = len(rots) - camera_count * len(future_infos)
        expected_transform_count = camera_count * (
            1 + len(results.get('adjacent', [])) + len(future_infos))
        temporal_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
        if (future_start < camera_count or any(
                len(item) != expected_transform_count
                for item in temporal_inputs)):
            raise ValueError('img_inputs do not match adjacent and future '
                             'frame metadata')

        future_depth = []
        for index, (points, info) in enumerate(
                zip(future_points, future_infos)):
            frame_start = future_start + index * camera_count
            frame_end = frame_start + camera_count
            future_depth.append(
                self._load_sample(points, info, results['cam_names'], imgs,
                                  intrins[frame_start:frame_end],
                                  post_rots[frame_start:frame_end],
                                  post_trans[frame_start:frame_end]))
        results['future_gt_depth'] = future_depth
        return results


__all__ = [
    'GenerateForecastOccDepthTargets', 'LoadForecastOccGT',
    'LoadForecastOccPoints'
]
