# dataset settings
USEPE = True
dataset_type = 'KITTIDataset'
data_root = 'data/kitti'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (352, 704)
train_pipeline = [
    dict(type='LoadImageFromFile',USEPE=USEPE),
    dict(type='DepthLoadAnnotations'),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=True),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 704)),
    dict(type='ColorAug', prob=0.5, gamma_range=[0.9, 1.1], brightness_range=[0.9, 1.1], color_range=[0.9, 1.1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'depth_gt'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
test_pipeline = [
    dict(type='LoadImageFromFile',USEPE=USEPE),
    dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1216, 352),
        flip=True,
        flip_direction='horizontal',
        transforms=[
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='gt_depth',
        depth_scale=256,
        split='kitti_eigen_train.txt',
        pipeline=train_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='gt_depth',
        depth_scale=256,
        split='kitti_eigen_test.txt',
        pipeline=test_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='input',
        ann_dir='gt_depth',
        depth_scale=256,
        split='kitti_eigen_test.txt',
        pipeline=test_pipeline,
        garg_crop=True,
        eigen_crop=False,
        min_depth=1e-3,
        max_depth=80))

