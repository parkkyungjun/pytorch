# dataset settings

classes = ('background', 'container_truck', 'forklift', 'reach_stacker', 'ship')
palette = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]

dataset_type = 'CustomDataset'
data_root = 'D:\\film\\pytorch\\mmsegmentation\\configs\\_base_\\data\\custom_dataset'
img_norm_cfg = dict(
    mean=[117.871, 115.512, 118.21], std=[45.079, 45.176, 42.465], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        palette=palette,
        classes=classes,
        img_dir='labeled_images\\train',
        ann_dir='labels\\train',
        pipeline=train_pipeline),
    val=dict(
        palette=palette,
        classes=classes,
        type=dataset_type,
        data_root=data_root,
        img_dir='labeled_images\\train',
        ann_dir='labels\\train',
        pipeline=test_pipeline))
    # test=dict(
    #     palette=palette,
    #     classes=classes,
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='images/validation',
    #     ann_dir='annotations/validation',
    #     pipeline=test_pipeline))
