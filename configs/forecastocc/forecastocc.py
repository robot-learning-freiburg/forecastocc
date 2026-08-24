# Copyright (c) OpenMMLab. All rights reserved.
import os

_base_ = ['../_base_/default_runtime.py']

data_root = os.environ.get('NUSCENES_DATA_ROOT', 'data/nuscenes')
ann_root = os.environ.get('NUSCENES_ANN_ROOT', data_root)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.0,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

num_trans_channels = 64

multi_adj_frame_id_cfg = (1, 3, 1)

future_frame_indices = [2, 4, 6]
num_future_frames = len(future_frame_indices)

model = dict(
    type='ForecastOcc',
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    num_future_frames=num_future_frames,
    future_state_alignment_loss=dict(
        weight=30.0,
        type='huber_cosine',
    ),
    forecast_head=dict(
        type='ForecastingModule',
        num_layers=8,
        embed_dims=256,
        num_heads=16,
        mlp_ratio=4,
        num_cams=6,
        scale_dims=[16, 24, 64, 152],
        num_context=4,
        num_future_frames=num_future_frames),
    with_prev=True,
    img_backbone=dict(
        _scope_='mmcls',
        type='EfficientNet',
        arch='b3',
        out_indices=(2, 3, 4, 6)),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[32, 48, 136, 1536],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[16, 24, 64, 152]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=num_trans_channels,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.0),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=num_trans_channels * len(range(*multi_adj_frame_id_cfg)),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[
            num_trans_channels,
            num_trans_channels * 2,
            num_trans_channels * 4,
        ],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    img_bev_encoder_neck=dict(
        type='LSSFPN3D',
        in_channels=num_trans_channels * 7,
        out_channels=num_trans_channels),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=num_trans_channels,
        with_cp=False,
        num_layer=[1],
        num_channels=[num_trans_channels],
        stride=[1],
        backbone_output_ids=[0]),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
)

dataset_type = 'NuScenesForecastOccDataset'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(type='LoadForecastOccGT'),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(
        type='LoadForecastOccPoints',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='GenerateForecastOccDepthTargets',
        downsample=1,
        grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_depth', 'voxel_semantics', 'future_gt_depth',
            'future_voxel_semantics'
        ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    future_frame_indices=future_frame_indices,
)

test_data_config = dict(
    data_root=data_root,
    occ_gt_root=ann_root,
    pipeline=test_pipeline,
    ann_file=os.path.join(ann_root, 'forecastocc-nuscenes_infos_val.pkl'),
    test_mode=True,
    box_type_3d='LiDAR')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        data_root=data_root,
        occ_gt_root=ann_root,
        ann_file=os.path.join(
            ann_root, 'forecastocc-nuscenes_infos_train.pkl'),
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ('val', 'train', 'test'):
    data[key].update(share_data_config)

optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={'forecast_head': dict(lr_mult=100.0)}))

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[100])

checkpoint_config = dict(interval=1)

evaluation = dict(interval=2)

runner = dict(type='EpochBasedRunner', max_epochs=12)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# The released training recipe warm-starts every non-forecasting parameter
# from the current-occupancy model.
load_from = os.environ.get('FORECASTOCC_INIT_CHECKPOINT')
