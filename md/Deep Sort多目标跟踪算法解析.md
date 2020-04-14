# Deep Sort多目标跟踪算法解析

Deep Sort是多目标跟踪(Multi-Object Tracking)中常用到的一种算法，是一个Detection Based Tracking的方法。这个算法工业界关注度非常高，在知乎上有很多文章都是使用了Deep Sort进行工程部署。笔者将参考前辈的博客，结合自己的实践(理论&代码)对Deep Sort算法进行解析。

## 1. MOT主要步骤

在《DEEP LEARNING IN VIDEO MULTI-OBJECT TRACKING: A SURVEY》这篇基于深度学习的多目标跟踪的综述中，描述了MOT问题中四个主要步骤：

![多目标跟踪众多的主要步骤](https://img-blog.csdnimg.cn/20200412204809695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

- 给定视频原始帧。
- 运行目标检测器如Faster R-CNN、YOLOv3、SSD等进行检测，获取目标检测框。
- 将所有目标框中对应的目标抠出来，进行特征提取（包括表观特征或者运动特征）。
- 进行相似度计算，计算前后两帧目标之间的匹配程度（前后属于同一个目标的之间的距离比较小，不同目标的距离比较大）
- 数据关联，为每个对象分配目标的ID。

以上就是四个核心步骤，其中核心是检测，SORT论文的摘要中提到，仅仅换一个更好的检测器，就可以将目标跟踪表现提升18.9%。


## 2. SORT

Deep Sort算法的前身是SORT, 全称是Simple Online and Realtime Tracking。简单介绍一下，SORT最大特点是基于Faster R-CNN的目标检测方法，并利用卡尔曼滤波算法+匈牙利算法，极大提高了多目标跟踪的速度，同时达到了SOTA的准确率。

这个算法确实是在实际应用中使用较为广泛的一个算法，核心就是两个算法：**卡尔曼滤波**和**匈牙利算法**。

**卡尔曼滤波算法**分为两个过程，预测和更新。该算法将目标的运动状态定义为8个正态分布的向量。

预测：当目标经过移动，通过上一帧的目标框和速度等参数，预测出当前帧的目标框位置和速度等参数。

更新：预测值和观测值，两个正态分布的状态进行线性加权，得到目前系统预测的状态。

**匈牙利算法：**解决的是一个分配问题，在MOT主要步骤中的计算相似度的，得到了前后两帧的相似度矩阵。匈牙利算法就是通过求解这个相似度矩阵，从而解决前后两帧真正匹配的目标。这部分sklearn库有对应的函数linear_assignment来进行求解。

**SORT算法**中相似度矩阵计算步骤是通过前后两帧IOU作为相似度评价标准。

![Harlek提供的SORT解析图](https://img-blog.csdnimg.cn/20200412214907925.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

Detections是通过目标检测器得到的目标框，Tracks是一段轨迹。核心是匹配过程和卡尔曼滤波的预测和更新过程。

过程如下：目标检测器得到目标框Detections，同时系统进行预测当前的目标框Tracks, 将Detections和Tracks进行IOU匹配，最终得到的结果分为：

- Unmatched Tracks，这部分被认为是失配，Detection和Track无法匹配，如果失配持续了$T_{lost}$次，该目标ID将从图片中删除。
- Unmatched Detections, 这部分说明没有任意一个Track能匹配Detection, 所以要为这个detection分配一个新的track。
- Matched Track，这部分说明得到了匹配。

卡尔曼滤波可以根据Tracks状态**预测**下一帧的目标框状态。卡尔曼滤波**更新**是对观测值(匹配上的Track)和估计值更新所有track的状态。

## 3. Deep Sort

DeepSort中最大的特点是加入外观信息，借用了ReID领域模型来提取特征，减少了ID switch的次数。整体流程图如下：

![图片来自知乎Harlek](https://img-blog.csdnimg.cn/20200412221106751.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

可以看出，Deep Sort算法在SORT算法的基础上增加了级联匹配(Matching Cascade)+新轨迹的确认(confirmed)。总体流程就是：

- 卡尔曼滤波器预测轨迹Tracks
- 使用匈牙利算法将预测得到的轨迹Tracks和当前帧中的detections进行匹配(级联匹配和IOU匹配)
- 卡尔曼滤波更新。

其中上图中的级联匹配展开如下：

![图片来自知乎Harlek](https://img-blog.csdnimg.cn/20200412222541236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图非常清晰地解释了如何进行级联匹配，上图由虚线划分为两部分：

上半部分中计算相似度矩阵的方法使用到了**外观模型**(ReID)和**运动模型**(马氏距离)来计算相似度。

下半部分中是是级联匹配的数据关联步骤，匹配过程是一个循环(missing age个迭代，默认为70)，也就是从missing age=0到missing age=70的轨迹和Detections进行匹配，没有丢失过的轨迹优先匹配，丢失较为久远的就靠后匹配。

将Detection和Track进行匹配，所以出现几种情况

1. Detection和Track匹配，也就是Matched Tracks。普通连续跟踪的目标都属于这种情况，前后两帧都有目标，能够匹配上。
2. Detection没有找到匹配的Track，也就是Unmatched Detections。图像中突然出现新的目标的时候，Detection无法在之前的Track找到匹配的目标。
3. Track没有找到匹配的Detection，也就是Unmatched Tracks。连续追踪的目标超出图像区域，Track无法与当前任意一个Detection匹配。
4. 以上没有涉及一种特殊的情况，就是两个目标遮挡的情况。刚刚被遮挡的目标的Track也无法匹配Detection，目标暂时从图像中消失。之后被遮挡目标不再被遮挡，先出现的时候，这个时候需要新的算法来处理，应该尽量让被遮挡目标分配的ID不发生变动，减少ID Switch出现的次数，这就需要用到级联匹配了。

## 4. Deep Sort代码解析

论文中提供的代码是如下地址: https://github.com/nwojke/deep_sort 

![Github库中Deep_sort文件结构](https://img-blog.csdnimg.cn/20200414174713413.png)

上图是Github库中有关Deep SORT的核心代码，不包括Faster R-CNN检测部分，所以主要将讲解这部分的几个文件，笔者也对其中核心代码进行了部分注释，地址在:  https://github.com/pprp/deep_sort_yolov3_pytorch , 将其中的目标检测器换成了U版的yolov3, 将deep_sort文件中的核心进行了调用。

下图是笔者总结的这几个类调用的类图(不是特别严谨哈，大概展示各个模块的关系)：

![Deep Sort类图](https://img-blog.csdnimg.cn/20200413102815883.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

DeepSort是核心类，调用其他模块，大体上可以分为三个模块：

- ReID模块，用于提取表观特征，原论文中是生成了128维的embedding。
- Track模块，轨迹类，用于保存一个Track的状态信息，是一个基本单位。
- Tracker模块，Tracker模块掌握最核心的算法，**卡尔曼滤波**和**匈牙利算法**都是通过调用这个模块来完成的。

DeepSort类对外接口非常简单：

```python
self.deepsort = DeepSort(args.deepsort_checkpoint)#实例化
outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)#通过接收目标检测结果进行更新
```

在外部调用的时候只需要以上两步即可，非常简单。

通过类图，对整体模块有了框架上理解，下面深入理解一下这些模块。

### 4.1 流程图



![知乎@猫弟](https://img-blog.csdnimg.cn/2020041418343015.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这个真的非常感谢知乎@猫弟总结的流程图，讲解非常地清晰，如果单纯看代码，非常容易混淆。比如说代价矩阵的计算这部分，连续套了三个函数，才被真正调用。上图将整体流程总结地非常棒。笔者将跟着流程图+类图来进行代码地讲解。

### 4.2 Detection类&Track类

```python
class Detection(object):
    """
    This class represents a bounding box detection in a single image.
	"""
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
```

Detection类用于保存通过目标检测器得到的一个检测框，包含top left坐标+框的宽和高，以及该bbox的置信度还有通过reid获取得到的对应的embedding。除此以外提供了不同bbox位置格式的转换方法：

- tlwh: 代表左上角坐标+宽高
- tlbr: 代表左上角坐标+右下角坐标
- xyah: 代表中心坐标+宽高比+高

















现在Deep Sort基础上最新的改进是将**表观模型提取特征的部分**和**检测部分**同时做，做成一个多任务网络。比如FairMOT, JDE等都是在进行目标检测的同时学习ReID特征。



## 参考

 https://www.cnblogs.com/yanwei-li/p/8643446.html 

 https://zhuanlan.zhihu.com/p/97449724 

 https://zhuanlan.zhihu.com/p/80764724 