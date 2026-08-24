# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
SUPPORTED_VERSIONS = ('v1.0-trainval', 'v1.0-mini')
INFO_PREFIX = 'forecastocc-nuscenes'


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info (dict): Metadata needed to generate ground-truth labels.

    Returns:
        tuple: Ground-truth boxes and labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    This creates the train and validation pickle files consumed by the
    ForecastOcc dataset.

    Args:
        root_path (str): Path of dataset root.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, INFO_PREFIX, version=version, max_sweeps=max_sweeps)


def add_forecasting_info(root_path, version):
    """Add scene and occupancy paths to generated nuScenes metadata."""
    nuscenes = NuScenes(version=version, dataroot=root_path)
    for split in ('train', 'val'):
        info_path = os.path.join(
            root_path, f'{INFO_PREFIX}_infos_{split}.pkl')
        with open(info_path, 'rb') as file:
            dataset = pickle.load(file)

        for info_idx, info in enumerate(dataset['infos']):
            if info_idx % 10 == 0:
                print(f'{split}: {info_idx}/{len(dataset["infos"])}')
            # Collect the object annotations used by the BEV pipeline.
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            info['ann_infos'] = ann_infos
            info['ann_infos'] = get_gt(info)
            info['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            info['occ_path'] = os.path.join('gts', scene['name'],
                                            info['token'])

        with open(info_path, 'wb') as file:
            pickle.dump(dataset, file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare nuScenes metadata for ForecastOcc.')
    parser.add_argument(
        '--root-path',
        default='data/nuscenes',
        help='nuScenes root containing samples, sweeps, metadata, and gts.')
    parser.add_argument(
        '--version',
        default='v1.0-trainval',
        choices=SUPPORTED_VERSIONS,
        help='nuScenes dataset version to prepare.')
    parser.add_argument('--max-sweeps', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nuscenes_data_prep(
        root_path=args.root_path,
        version=args.version,
        max_sweeps=args.max_sweeps)

    print('Adding ForecastOcc metadata fields')
    add_forecasting_info(args.root_path, args.version)
