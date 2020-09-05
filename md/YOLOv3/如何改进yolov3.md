# 我们是如何改进YOLOv3进行红外小目标检测的？

【GiantPandCV导语】本文将介绍BBuf、小武和笔者一起在过年期间完成的一个目标检测项目，将描述我们如何改进模型、改进的思路、实验思路、结果汇总和经验性总结。声明：这篇文章经过了三人同意，并且核心创新点也将被公布。此外，由于经验上的不足，可能整个实验思路不够成熟，比不上CV大组的严谨性和完备性，如有问题还烦请指教。

## 1. 红外小目标检测

红外小目标检测的目标比较小，目标极其容易和其他物体混淆，有一定的挑战性。

另外，这本质上也是一个小目标领域的问题，很多适用于小目标的创新点也会被借鉴进来。

![数据来源自@小武](https://img-blog.csdnimg.cn/20200831212549491.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

此外，该数据集还有一个特点，就是分背景，虽然同样是检测红外小目标，区别是背景的不同，我们对数据集进行了统计以及通过人工翻看的方式总结了其特点，如下表所示：

| 背景类别             | 数量 | 特点                                                         | 数据难度 | 测试mAP+F1 | 建议                                            |
| -------------------- | ---- | ------------------------------------------------------------ | -------- | ---------- | ----------------------------------------------- |
| trees                | 581  | 背景干净，目标明显，数量较多                                 | 低       | 0.99+0.97  | 无                                              |
| cloudless_sky        | 1320 | 背景干净，目标明显，数量多                                   | 低       | 0.98+0.99  | 无                                              |
| architecture         | 506  | 背景变化较大，目标形态变化较大，数量较多                     | 一般     | 0.92+0.96  | focal loss                                      |
| continuous_cloud_sky | 878  | 背景干净，目标形态变化不大，但个别目标容易会发生和背景中的云混淆 | 一般     | 0.93+0.95  | focal loss                                      |
| complex_cloud        | 561  | 目标形态基本无变化，但背景对目标的定位影响巨大               | 较难     | 0.85+0.89  | focal loss                                      |
| sea                  | 17   | 背景干净，目标明显，数量极少                                 | 一般     | 0.87+0.88  | 生成高质量新样本，可以让其转为简单样本（Mixup） |
| sea_sky              | 45   | 背景变化较大，且单张图像中目标个数差异变化大，有密集的难点，且数量少 | 困难     | 0.68+0.77  | paste策略                                       |

通过以上结果，可以看出背景的不同对结果影响还是蛮大的，最后一列也给出了针对性的建议，打算后续实施。

## 2. 实验过程

首先，我们使用的是U版的yolov3: `https://github.com/ultralytics/yolov3`，那时候YOLOv4/5、PPYOLO还都没出，当时出了一个《从零开始学习YOLOv3》就是做项目的时候写的电子书，其中的在YOLOv3中添加注意力机制那篇很受欢迎（可以水很多文章出来，毕业要紧:）

我们项目的代码以及修改情况可以查看：`https://github.com/GiantPandaCV/yolov3-point`

将数据集转成VOC格式的数据集，之前文章有详细讲述如何转化为标准的VOC数据集，以及如何将VOC格式数据集转化为U版的讲解。当时接触到几个项目，都需要用YOLOv3，由于每次都需要转化，大概分别调用4、5个脚本吧，感觉很累，所以当时花了一段时间构建了一个一键从VOC转U版YOLOv3格式的脚本库: `https://github.com/pprp/voc2007_for_yolo_torch`。

到此时为止，我们项目就已经可以运行了，然后就是很多细节调整了。

### 2.1 修改Anchor

红外小目标的Anchor和COCO等数据集的Anchor是差距很大的，为了更好更快速的收敛，采用了BBuf总结的一套专门计算Anchor的脚本：

```python
#coding=utf-8
import xml.etree.ElementTree as ET
import numpy as np

 
def iou(box, clusters):
    """
    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。
    参数box: 元组或者数据，代表ground truth的长宽。
    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    返回：ground truth和每个Anchor框的交并比。
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算一个ground truth和k个Anchor的交并比的均值。
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    参数k: Anchor的个数
    参数dist: 距离函数
    返回值：形状为(k, 2)的k个Anchor框
    """
    # 即是上面提到的r
    rows = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离
    distances = np.empty((rows, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_clusters = np.zeros((rows,))
    # 设置随机数种子
    np.random.seed()

    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 开始聚类
    while True:
        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        nearest_clusters = np.argmin(distances, axis=1)
        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心为簇里面所有的ground truth框的均值
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        # 更新每个ground truth"距离"最近的Anchor索引
        last_clusters = nearest_clusters

    return clusters

# 加载自己的数据集，只需要所有labelimg标注出来的xml文件即可
def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)
        # 图片高度
        height = int(tree.findtext("./size/height"))
        # 图片宽度
        width = int(tree.findtext("./size/width"))
        
        for obj in tree.iter("object"):
            # 偏移量
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            # 将Anchor的长宽放入dateset，运行kmeans获得Anchor
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)
 
if __name__ == '__main__':
    
    ANNOTATIONS_PATH = "F:\Annotations" #xml文件所在文件夹
    CLUSTERS = 9 #聚类数量，anchor数量
    INPUTDIM = 416 #输入网络大小
 
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    print('Boxes:')
    print(np.array(out)*INPUTDIM)    
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))       
    final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(final_anchors))
    print("After Sort Ratios:\n {}".format(sorted(final_anchors)))
```

通过浏览脚本就可以知道，**Anchor和图片的输入分辨率有没有关系** 这个问题了，当时这个问题有很多群友都在问。通过kmeans函数得到的结果实际上是归一化到0-1之间的，然后Anchor的输出是在此基础上乘以输入分辨率的大小。所以个人认为Anchor和图片的输入分辨率是**有关系**的。

此外，U版也提供了Anchor计算，如下：

```python
def kmean_anchors(path='./2007_train.txt', n=5, img_size=(416, 416)):
    # from utils.utils import *; _ = kmean_anchors()
    # Produces a list of target kmeans suitable for use in *.cfg files
    from utils.datasets import LoadImagesAndLabels
    thr = 0.20  # IoU threshold

    def print_results(thr, wh, k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(torch.Tensor(wh), torch.Tensor(k))
        max_iou, min_iou = iou.max(1)[0], iou.min(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (
            iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' %
              (thr, bpr, aat))
        print(
            'kmeans anchors (n=%g, img_size=%s, IoU=%.3f/%.3f/%.3f-min/mean/best): '
            % (n, img_size, min_iou.mean(), iou.mean(), max_iou.mean()),
            end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])),
                  end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(thr, wh, k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k)).max(1)[0]  # max iou
        bpr = (iou > thr).float().mean()  # best possible recall
        return iou.mean() * bpr  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path,
                                  augment=True,
                                  rect=True,
                                  cache_labels=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] *
                  (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1],
                            size=(wh.shape[0],
                                  1))  # normalized to pixels (multi-scale)

    # Darknet yolov3.cfg anchors
    use_darknet = False
    if use_darknet:
        k = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                      [59, 119], [116, 90], [156, 198], [373, 326]])
    else:
        # Kmeans calculation
        from scipy.cluster.vq import kmeans
        print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
        s = wh.std(0)  # sigmas for whitening
        k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
        k *= s
    k = print_results(thr, wh, k)
    # Evolve
    wh = torch.Tensor(wh)
    f, ng = fitness(thr, wh, k), 2000  # fitness, generations
    for _ in tqdm(range(ng), desc='Evolving anchors'):
        kg = (
            k.copy() *
            (1 + np.random.random() * np.random.randn(*k.shape) * 0.30)).clip(
                min=2.0)
        fg = fitness(thr, wh, kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(thr, wh, k)
    k = print_results(thr, wh, k)

    return k
```

这个和超参数搜索那篇采用的方法类似，也是一种类似遗传算法的方法，通过一代一代的筛选找到合适的Anchor。以上两种方法笔者并没有对比，有兴趣可以试试这两种方法，对比看看。

Anchor这方面设置了三个不同的数量进行聚类：

3 anchor: 

```
13, 18, 16, 22, 19, 25
```

6 anchor:

```
12,17, 14,17, 15,19, 15,21, 13,20, 19,24
```

9 anchor:

```
10,16, 12,17, 13,20, 13,22, 15,18, 15,20, 15,23, 18,23, 21,26
```

## 2.2 构建Baseline

由于数据集是单类的，并且相对VOC等数据集来看，比较单一，所以不打算使用Darknet53这样的深度神经网络，采用的Baseline是YOLOv3-tiny模型，在使用原始Anchor的情况下，该模型可以在验证集上达到mAP@0.5=97.3%，在测试集上达到mAP@0.5=0.884的结果。

令人欣喜的是这个结果超过了传统方法，但令人难过的是，这个Baseline就已经很高，未来改进难度会很大。

那接下来换Anchor，用上一节得到的新Anchor替换掉原来的Anchor，该改掉的模型为yolov3-tiny-6a:

| Epoch    | Model           | P     | R     | mAP@0.5 | F1    | dataset |
| -------- | --------------- | ----- | ----- | ------- | ----- | ------- |
| baseline | yolov3-tiny原版 | 0.982 | 0.939 | 0.932   | 0.96  | valid   |
| baseline | yolov3-tiny原版 | 0.96  | 0.873 | 0.869   | 0.914 | test    |
| 6a       | yolov3-tiny-6a  | 0.973 | 0.98  | 0.984   | 0.977 | valid   |
| 6a       | yolov3-tiny-6a  | 0.936 | 0.925 | 0.915   | 0.931 | test    |
可以看到几乎所有的指标都提升了，这说明Anchor先验的引入是很有必要的。

## 2.3 数据集部分改进

上边已经分析过了，背景对目标检测的结果还是有一定影响的，所以我们先后使用了几种方法进行改进。

**第一个：过采样**

通过统计不同背景的图像的数量，比如以sea为背景的图像只有17张，而最多的cloudless_sky为背景的图像有1300+张，这就产生了严重的不平衡性。显然cloudless_sky为背景的很简单，sea为背景的难度更大，这样由于数据不平衡的原因，训练得到的模型很可能也会在cloudless_sky这类图片上效果很好，在其他背景下效果一般。

所以首先要采用过采样的方法，这里的过采样可能和别的地方的不太一样，这里指的是将某些背景数量小的图片通过复制的方式扩充。

| Epoch              | Model                 | P     | R     | mAP@0.5   | F1        | dataset     |
| ------------------ | --------------------- | ----- | ----- | --------- | --------- | ----------- |
| baseline(os) | yolov3-tiny原版 | 0.985 | 0.971 |0.973|0.978|valid|
| baseline(os) | yolov3-tiny原版 | 0.936 | 0.871 |0.86|0.902|test|
| baseline | yolov3-tiny原版 | 0.982 | 0.939 |0.932|0.96|valid|
| baseline | yolov3-tiny原版 | 0.96 | 0.873 |0.869|0.914|test|

:( 可惜实验结果不支持想法，一起分析一下。ps:os代表over sample

然后进行分背景测试，结果如下：

 **均衡后的分背景测试**


| data                 | num  | model          | P     | R     | mAP   | F1    |
| -------------------- | ---- | -------------- | ----- | ----- | ----- | ----- |
| trees                | 506  | yolov3-tiny-6a | 0.924 | 0.996 | 0.981 | 0.959 |
| sea_sky              | 495  | yolov3-tiny-6a | 0.927 | 0.978 | 0.771 | 0.85  |
| sea                  | 510  | yolov3-tiny-6a | 0.923 | 0.935 | 0.893 | 0.929 |
| continuous_cloud_sky | 878  | yolov3-tiny-6a | 0.957 | 0.95  | 0.933 | 0.953 |
| complex_cloud        | 561  | yolov3-tiny-6a | 0.943 | 0.833 | 0.831 | 0.885 |
| cloudless_sky        | 1320 | yolov3-tiny-6a | 0.993 | 0.981 | 0.984 | 0.987 |
| architecture         | 506  | yolov3-tiny-6a | 0.959 | 0.952 | 0.941 | 0.955 |

从分背景结果来看，确实sea训练数据很少的结果很好，mAP提高了2个点，但是complex_cloud等mAP有所下降。总结一下就是对于训练集中数据很少的背景类mAP有提升，但是其他本身数量就很多的背景mAP略微下降或者保持。

**第二个：**









