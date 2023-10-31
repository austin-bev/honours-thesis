num_classes = 12
dataset_type = 'Camvid11Dataset'
data_root = 'data/CamVid11/'
img_scale=(720, 960)
crop_size = (720, 960)
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNet',
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.01),
        depth=34,
        pretrained='open-mmlab://resnet34',
        num_stages=2,
        strides=(1, 2),
        dilations=(1, 1),
        out_indices=(0, 1),
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=128,
        channels=128,
        concat_input=False,
        num_classes=num_classes,
        in_index=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True, momentum=0.01),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='train_labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val',
        ann_dir='val_labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test_labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='MMSegWandbHook',
            by_epoch=True,
            init_kwargs=dict(
                entity='austin-bevac',
                project='2023-honours-austin',
                config=dict(
                    learning_rate=0.00045,
                    momentum=0.9,
                    batch_size=4,
                    backbone='ResNet',
                    decodehead='FCNHead',
                    dataset='CityscapesDataset',
                    epochs=500,
                    is_pretrained=True,
                    comment=
                    'Validation synthetic, all labels, higher resolution. ResNet34, 2 stages'
                ),
                name=
                'CamVid new base'
            ))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.0045, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=500, max_iters=None)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(
    interval=1, metric='mIoU', pre_eval=True, save_best='mIoU', by_epoch=True)
workers_per_gpu = 1
work_dir = './work_dirs/experimentation/6021-camvid'
seed = 0
gpu_ids = range(0, 1)
device = 'cuda'
