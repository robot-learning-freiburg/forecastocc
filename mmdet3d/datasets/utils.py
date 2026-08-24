# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple

import mmcv

# yapf: disable
from mmdet3d.datasets.pipelines import (Collect3D, DefaultFormatBundle3D,
                                        LoadAnnotations3D,
                                        LoadImageFromFileMono3D,
                                        LoadMultiViewImageFromFiles,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        MultiScaleFlipAug3D,
                                        PointSegClassMapping)
from mmdet.datasets.pipelines import LoadImageFromFile, MultiScaleFlipAug
# yapf: enable
from .builder import PIPELINES

ClassInfo = namedtuple("ClassInfo", ["train_id", "id", "name", "category", "type", "color"])
id_to_trainid_info = {}
#     1:  ClassInfo(255, "void / ignore", "animal", "void", (0, 0, 0)),
#     5:  ClassInfo(255, "void / ignore", "human.pedestrian.personal_mobility", "void", (0, 0, 0)),
#     7:  ClassInfo(255, "void / ignore", "human.pedestrian.stroller", "void", (0, 0, 0)),
#     8:  ClassInfo(255, "void / ignore", "human.pedestrian.wheelchair", "void", (0, 0, 0)),
#     10: ClassInfo(255, "void / ignore", "movable_object.debris", "void", (0, 0, 0)),
#     11: ClassInfo(255, "void / ignore", "movable_object.pushable_pullable", "void", (0, 0, 0)),
#     13: ClassInfo(255, "void / ignore", "static_object.bicycle_rack", "void", (0, 0, 0)),
#     19: ClassInfo(255, "void / ignore", "vehicle.emergency.ambulance", "void", (0, 0, 0)),
#     20: ClassInfo(255, "void / ignore", "vehicle.emergency.police", "void", (0, 0, 0)),
#     0:  ClassInfo(255, "void / ignore", "noise", "void", (0, 0, 0)),
#     29: ClassInfo(255, "void / ignore", "static.other", "void", (0, 0, 0)),
#     31: ClassInfo(255, "void / ignore", "vehicle.ego", "void", (0, 0, 0)),
#     9:  ClassInfo(0, "barrier (thing)", "movable_object.barrier", "thing", (112, 128, 144)),
#     14: ClassInfo(1, "bicycle (thing)", "vehicle.bicycle", "thing", (119, 11, 32)),
#     15: ClassInfo(2, "bus (thing)", "vehicle.bus.bendy", "thing", (0, 60, 100)),
#     16: ClassInfo(2, "bus (thing)", "vehicle.bus.rigid", "thing", (0, 60, 100)),
#     17: ClassInfo(3, "car (thing)", "vehicle.car", "thing", (0, 0, 142)),
#     18: ClassInfo(4, "construction_vehicle (thing)", "vehicle.construction", "thing", (70, 70, 70)),
#     21: ClassInfo(5, "motorcycle (thing)", "vehicle.motorcycle", "thing", (0, 0, 230)),
#     2:  ClassInfo(6, "pedestrian (thing)", "human.pedestrian.adult", "thing", (220, 20, 60)),
#     3:  ClassInfo(6, "pedestrian (thing)", "human.pedestrian.child", "thing", (220, 20, 60)),
#     4:  ClassInfo(6, "pedestrian (thing)", "human.pedestrian.construction_worker", "thing", (220, 20, 60)),
#     6:  ClassInfo(6, "pedestrian (thing)", "human.pedestrian.police_officer", "thing", (220, 20, 60)),
#     12: ClassInfo(7, "traffic_cone (thing)", "movable_object.trafficcone", "thing", (255, 170, 30)),
#     22: ClassInfo(8, "trailer (thing)", "vehicle.trailer", "thing", (0, 0, 110)),
#     23: ClassInfo(9, "truck (thing)", "vehicle.truck", "thing", (0, 0, 70)),
#     24: ClassInfo(10, "driveable_surface (stuff)", "flat.driveable_surface", "stuff", (128, 64, 128)),
#     25: ClassInfo(11, "other_flat (stuff)", "flat.other", "stuff", (110, 110, 110)),
#     26: ClassInfo(12, "sidewalk (stuff)", "flat.sidewalk", "stuff", (244, 35, 232)),
#     27: ClassInfo(13, "terrain (stuff)", "flat.terrain", "stuff", (152, 251, 152)),
#     28: ClassInfo(14, "manmade (stuff)", "static.manmade", "stuff", (70, 70, 70)),
#     30: ClassInfo(15, "vegetation (stuff)", "static.vegetation", "stuff", (107, 142, 35)),
# }



skitti_id_to_trainid_info = {
    0:   ClassInfo(255, 0,"void or ignore", "void", "void", (0, 0, 0)),
    1:   ClassInfo(255, 1, "void or ignore", "void.outlier", "void", (0, 0, 0)),
    10:  ClassInfo(0,  10, "car (thing)", "vehicle.car", "thing", (100, 150, 245)),
    11:  ClassInfo(1,  11, "bicycle (thing)", "vehicle.bicycle", "thing", (100, 230, 245)),
    13:  ClassInfo(4,  13, "bus (thing)", "vehicle.bus", "thing", (100, 80, 250)),
    15:  ClassInfo(2,  15, "motorcycle (thing)", "vehicle.motorcycle", "thing", (30, 60, 150)),
    16:  ClassInfo(4,  16, "on rails (thing)", "vehicle.on_rails", "thing", (0, 0, 255)),
    18:  ClassInfo(3,  18, "truck (thing)", "vehicle.truck", "thing", (80, 30, 180)),
    20:  ClassInfo(4,  20, "other vehicle (thing)", "vehicle.other", "thing", (0, 0, 255)),
    30:  ClassInfo(5,  30, "person (thing)", "human.pedestrian", "thing", (255, 30, 30)),
    31:  ClassInfo(6,  31, "bicyclist (thing)", "human.rider.bicyclist", "thing", (255, 40, 200)),
    32:  ClassInfo(7,  32, "motorcyclist (thing)", "human.rider.motorcyclist", "thing", (150, 30, 90)),
    40:  ClassInfo(8,  40, "road (stuff)", "flat.road", "stuff", (255, 0, 255)),
    44:  ClassInfo(9,  44, "parking (stuff)", "flat.parking", "stuff", (255, 150, 255)),
    48:  ClassInfo(10, 48, "sidewalk (stuff)", "flat.sidewalk", "stuff", (75, 0, 75)),
    49:  ClassInfo(11, 49, "other ground (stuff)", "flat.other_ground", "stuff", (175, 0, 75)),
    50:  ClassInfo(12, 50, "building (stuff)", "construction.building", "stuff", (255, 200, 0)),
    51:  ClassInfo(13, 51, "fence (stuff)", "construction.fence", "stuff", (255, 120, 50)),
    52:  ClassInfo(255, 52, "void or ignore", "construction.other", "void", (0, 0, 0)),
    60:  ClassInfo(8,   60, "lane marking (stuff)", "flat.lane_marking", "stuff", (150, 255, 170)),
    70:  ClassInfo(14, 70, "vegetation (stuff)", "static.vegetation", "stuff", (0, 175, 0)),
    71:  ClassInfo(15, 71, "trunk (stuff)", "static.trunk", "stuff", (135, 60, 0)),
    72:  ClassInfo(16, 72, "terrain (stuff)", "flat.terrain", "stuff", (150, 240, 80)),
    80:  ClassInfo(17, 80, "pole (stuff)", "static.pole", "stuff", (255, 240, 150)),
    81:  ClassInfo(18, 81, "traffic sign (stuff)", "static.traffic_sign", "stuff", (255, 0, 0)),
    99:  ClassInfo(255, 99, "void or ignore", "object.other", "void", (0, 0, 0)),
    252: ClassInfo(0,   252, "moving car (thing)", "vehicle.car", "thing", (100, 150, 245)),
    253: ClassInfo(6,   253, "moving bicyclist (thing)", "human.rider.bicyclist", "thing", (255, 40, 200)),
    254: ClassInfo(5,   254, "moving person (thing)", "human.pedestrian", "thing", (255, 30, 30)),
    255: ClassInfo(7,   255, "moving motorcyclist (thing)", "human.rider.motorcyclist", "thing", (150, 30, 90)),
    256: ClassInfo(4,   256, "moving on rails (thing)", "vehicle.on_rails", "thing", (0, 0, 255)),
    257: ClassInfo(4,   257, "moving bus (thing)", "vehicle.bus", "thing", (100, 80, 250)),
    258: ClassInfo(3,   258, "moving truck (thing)", "vehicle.truck", "thing", (80, 30, 180)),
    259: ClassInfo(4,   259, "moving other vehicle (thing)", "vehicle.other", "thing", (0, 0, 255)),
}

skitti_trainid_to_id_info = {}
for k, v in skitti_id_to_trainid_info.items():
    if v.train_id == 255:
        continue
    if v.train_id not in skitti_trainid_to_id_info:
        skitti_trainid_to_id_info[v.train_id] = v


waymo_id_to_trainid_info = {
    0:  ClassInfo(255, 0, "undefined", "void", "void", (0, 0, 0)),

    # Thing classes (continuous)
    1:  ClassInfo(0, 1, "car", "vehicle.car", "thing", (100, 150, 245)),
    2:  ClassInfo(1, 2, "truck", "vehicle.truck", "thing", (80, 30, 180)),
    3:  ClassInfo(2, 3, "bus", "vehicle.bus", "thing", (100, 80, 250)),
    4:  ClassInfo(3, 4, "other vehicle", "vehicle.other", "thing", (0, 0, 255)),
    5:  ClassInfo(4, 5, "motorcyclist", "human.rider.motorcyclist", "thing", (150, 30, 90)),
    6:  ClassInfo(5, 6, "bicyclist", "human.rider.bicyclist", "thing", (255, 40, 200)),
    7:  ClassInfo(6, 7, "pedestrian", "human.pedestrian", "thing", (255, 30, 30)),
    12: ClassInfo(7, 12, "bicycle", "vehicle.bicycle", "thing", (100, 230, 245)),
    13: ClassInfo(8, 13, "motorcycle", "vehicle.motorcycle", "thing", (30, 60, 150)),

    # Stuff classes (after thing classes)
    8:  ClassInfo(9, 8, "sign", "static.traffic_sign", "stuff", (255, 0, 0)),
    9:  ClassInfo(10, 9, "traffic light", "static.traffic_light", "stuff", (255, 200, 0)),
    10: ClassInfo(11, 10, "pole", "static.pole", "stuff", (255, 240, 150)),
    11: ClassInfo(12, 11, "construction cone", "construction.cone", "stuff", (255, 120, 50)),
    14: ClassInfo(13, 14, "building", "construction.building", "stuff", (255, 200, 0)),
    15: ClassInfo(14, 15, "vegetation", "nature.vegetation", "stuff", (0, 175, 0)),
    16: ClassInfo(15, 16, "tree trunk", "nature.trunk", "stuff", (135, 60, 0)),
    17: ClassInfo(16, 17, "curb", "flat.curb", "stuff", (210, 200, 160)),
    18: ClassInfo(17, 18, "road", "flat.road", "stuff", (255, 0, 255)),
    19: ClassInfo(18, 19, "lane marker", "flat.lane_marker", "stuff", (150, 255, 170)),
    20: ClassInfo(19, 20, "other ground", "flat.other_ground", "stuff", (175, 0, 75)),
    21: ClassInfo(20, 21, "walkable", "flat.walkable", "stuff", (255, 150, 255)),
    22: ClassInfo(21, 22, "sidewalk", "flat.sidewalk", "stuff", (75, 0, 75)),
}

waymo_trainid_to_id_info = {}
for k, v in waymo_id_to_trainid_info.items():
    if v.train_id == 255:
        continue
    if v.train_id not in waymo_trainid_to_id_info:
        waymo_trainid_to_id_info[v.train_id] = v

def is_loading_function(transform):
    """Judge whether a transform function is a loading function.

    Note: `MultiScaleFlipAug3D` is a wrapper for multiple pipeline functions,
    so we need to search if its inner transforms contain any loading function.

    Args:
        transform (dict | :obj:`Pipeline`): A transform config or a function.

    Returns:
        bool: Whether it is a loading function. None means can't judge.
            When transform is `MultiScaleFlipAug3D`, we return None.
    """
    # TODO: use more elegant way to distinguish loading modules
    loading_functions = (LoadImageFromFile, LoadPointsFromFile,
                         LoadAnnotations3D, LoadMultiViewImageFromFiles,
                         LoadPointsFromMultiSweeps, DefaultFormatBundle3D,
                         Collect3D, LoadImageFromFileMono3D,
                         PointSegClassMapping)
    if isinstance(transform, dict):
        obj_cls = PIPELINES.get(transform['type'])
        if obj_cls is None:
            return False
        if obj_cls in loading_functions:
            return True
        if obj_cls in (MultiScaleFlipAug3D, MultiScaleFlipAug):
            return None
    elif callable(transform):
        if isinstance(transform, loading_functions):
            return True
        if isinstance(transform, (MultiScaleFlipAug3D, MultiScaleFlipAug)):
            return None
    return False


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict] | list[:obj:`Pipeline`]):
            Data pipeline configs or list of pipeline functions.

    Returns:
        list[dict] | list[:obj:`Pipeline`]): The new pipeline list with only
            keep loading image, points and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='Resize',
        ...         img_scale=[(640, 192), (2560, 768)], keep_ratio=True),
        ...    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        ...    dict(type='PointsRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='ObjectRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='PointShuffle'),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> assert expected_pipelines == \
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline = []
    for transform in pipeline:
        is_loading = is_loading_function(transform)
        if is_loading is None:  # MultiScaleFlipAug3D
            # extract its inner pipeline
            if isinstance(transform, dict):
                inner_pipeline = transform.get('transforms', [])
            else:
                inner_pipeline = transform.transforms.transforms
            loading_pipeline.extend(get_loading_pipeline(inner_pipeline))
        elif is_loading:
            loading_pipeline.append(transform)
    assert len(loading_pipeline) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data
