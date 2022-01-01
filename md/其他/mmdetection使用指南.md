---
title: mmdetection使用指南
date: 2019-11-13 20:44:29
tags:
- 目标检测
- mmdetection
- 使用指南
---

> 主要是目标检测方面的使用记录，mmdetection还有分类网络，分割等功能，但这篇博客主要关注目标检测，之后如果涉及到分割会再开一篇博客进行记录。

## 1. 安装

mmdetection需要的环境是cuda10.0为基础的环境，对驱动版本也有一定的要求，cuda8.0的我始终没有配通，主要的错误都是跟cuda相关的。

具体安装过程参见：<https://blog.csdn.net/DD_PP_JJ/article/details/103055629>

安装完大体环境以后，开始配置mmdetection

> - 操作系统：Linux
> - Python 3.5+
> - PyTorch 1.0+ 或 PyTorch-nightly
> - CUDA 9.0+
> - NCCL 2+
> - GCC 4.9+

然后cd进入mmdetection， 运行：

```shell
python setup.py develop
```

进行编译，如果你的mmdetection是从另外一台机器复制过来，只要他存在build文件夹，都有可能造成报错。直接`rm -rf build`, 删除build文件夹后重新运行，就可能能够顺利通过。

## 2. 准备VOC格式数据集

具体数据集构建可以看：

1. <https://www.cnblogs.com/pprp/p/10863496.html#%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E5%BB%BA>
2. <https://blog.csdn.net/weicao1990/article/details/93484603>

有一个库有一些脚本进行检查和生成：

<https://github.com/pprp/voc2007_for_yolo_torch>

## 3. 个性化配置

训练之前首先要根据自己的数据集对配置文件进行修改：

- 修改类别数量， num_classes = 类别数+1

    ```python
            dict(
                type='SharedFCBBoxHead',
                num_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2, #  修改 81 -> 2
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1],
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    ```

- 修改数据集地址：

    ```python
    # dataset settings
    dataset_type = 'VOCDataset' #  修改，'CocoDataset' -> 'VOCDataset'
    data_root = 'data/VOCdevkit/' #  修改，'data/coco/'  -> 'data/VOCdevkit'
    ```

    ```python
        train=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/train.txt', #  data_root + 'annotations/instances_train2017.json' -> data_root + 'VOC2007/ImageSets/Main/train.txt'
            img_prefix=data_root + 'VOC2007/', # 11/11, TC修改 data_root + 'train2017/' -> data_root + 'VOC2007/'
            pipeline=train_pipeline),
        val=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt', #  data_root + 'annotations/instances_train2017.json' -> data_root + 'VOC2007/ImageSets/Main/val.txt'
            img_prefix=data_root + 'VOC2007/',  # 11/11, TC修改 data_root + 'val2017/' -> data_root + 'VOC2007/'
            pipeline=test_pipeline),
        test=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt', # data_root + 'annotations/instances_train2017.json' -> data_root + 'VOC2007/ImageSets/Main/test.txt'
            img_prefix=data_root + 'VOC2007/',  #  data_root + 'test2017/' -> data_root + 'VOC2007/'
            pipeline=test_pipeline))
    ```

- 修改数据集voc.py文件：

    ```python
    @DATASETS.register_module
    class VOCDataset(XMLDataset):
    
        CLASSES = ('pos',) # 注意即便只有一个了类也要加逗号
        # ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        #            'tvmonitor')
    
        def __init__(self, **kwargs):
            super(VOCDataset, self).__init__(**kwargs)
            if 'VOC2007' in self.img_prefix:
                self.year = 2007
            elif 'VOC2012' in self.img_prefix:
                self.year = 2012
            else:
                raise ValueError('Cannot infer dataset year from img_prefix')
    ```

- 运行参数处理：

    ```python
    total_epochs = 20 # 训练最大的epoch数
    dist_params = dict(backend='nccl') # 分布式参数
    log_level = 'INFO' # 输出信息的完整度级别
    work_dir = './work_dirs/libra_faster_rcnn_x101_64x4d_fpn_1x' # log文件和模型文件存储路径
    load_from = None # 加载模型的路径，None表示从预训练模型加载
    resume_from = None # 恢复训练模型的路径，None表示不进行训练模型的恢复
    workflow = [('train', 1)] 
    # ======================================================
    # 训练与验证策略，[('train', 1)]表示只训练，不验证；
    # [('train', 2), ('val', 1)] 表示2个epoch训练，1个epoch验证
    # ======================================================
    ```

## 4. 训练命令

训练格式：

```shell
python tools/train.py ${config_files}
```

可选参数：

>  --validate : 每隔1个epoch就进行一次evaluation， 测一下map之类的指标。
>
> --work_dir: 指定训练的结果保存的位置，一般默认就行
>
> --resume_from: 需要指定到对应的权重文件
>
> --gpus： 指定使用哪个gpu
>
> --autoscale-lr: 根据GPU个数进行自动处理learning rate

举例，训练cascade R-CNN进行目标检测：

```shell
python tools/train.py configs/cascade_rcnn_r101_fpn_1x.py --validate
```



## 5. 测试命令

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

可选参数：

> --out : 输出结果文件，results.pkl
>
> --json_out: 输出结果文件，不需要后缀
>
> --eval: ['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'], 选择其中一个模式
>
> --show: 会跳出来一张图片给你展示
>
> --tmpdir : 将一些结果写入tmpdir

在VOC数据集下，应该采用以下方式进行测试：

```shell
python tools/test.py configs/retinanet_r101_fpn_1x.py work_dirs/retinanet_r101_fpn_1x/latest.pth --out ./result_retinanet.pkl
```

将结果输出到pkl文件夹中，然后在进行计算：

```
python tools/voc_eval.py result_retinanet.pkl configs/retinanet_r101_fpn_1x.py
```

得到以下结果：

```
+-------+-----+------+--------+-----------+-------+
| class | gts | dets | recall | precision | ap    |
+-------+-----+------+--------+-----------+-------+
| pos   | 186 | 2356 | 0.801  | 0.063     | 0.398 |
+-------+-----+------+--------+-----------+-------+
| mAP   |     |      |        |           | 0.398 |
+-------+-----+------+--------+-----------+-------+
```

## 6. 工具

首先安装：`pip install seaborn`

可视化格式：

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

- 具体的keys要看你的json文件中是否存在这个键值
- 画出分类误差

```
python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```

- 画出分类和回归结果，并输出到pdf

```
python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_reg --out losses.pdf
```

- 在同一个图中比较两个模型map

```
python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
```

- 计算平均训练速度

```
python tools/analyze_logs.py cal_train_time ${CONFIG_FILE} [--include-outliers]
```

- 获得模型训练所需参数：

```
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```



## 7. 遇到的问题

在测试的时候出现报错：

>   File "tools/test.py", line 224, in <module>
>     main()
>   File "tools/test.py", line 215, in main
>     results2json(dataset, outputs, args.json_out)
>   File "/home/cie320/mmdetection/mmdet/core/evaluation/coco_utils.py", line 203, in results2json
>     json_results = det2json(dataset, results)
>   File "/home/cie320/mmdetection/mmdet/core/evaluation/coco_utils.py", line 149, in det2json
>     img_id = dataset.img_ids[idx]
> AttributeError: 'VOCDataset' object has no attribute 'img_ids'

可以看到调用的是coco，这是你的命令的问题，你应该采用上述方法，分两步进行计算。

- 先生成results.pkl文件
- 然后运行voc_eval进行解析，得到最终结果。