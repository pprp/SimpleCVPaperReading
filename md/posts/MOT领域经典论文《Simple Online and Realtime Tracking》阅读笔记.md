---
title: 《Simple Online and Realtime Tracking》阅读笔记
date: 2019-11-4 14:54:46
tags: 
- sort
- mot
categories:
- 深度学习
---



# MOT领域经典论文《Simple Online and Realtime Tracking》阅读笔记

> 前言：目前打算将多目标检测MOT作为自己的毕设题目，相当于重新进入一个领域，在这个领域中听到最多的就是SORT论文，也就是今天要阅读的论文。
>
> 自己阅读论文的能力有点差，所以趁还没有进入研究生阶段，尽早提高自己的论文阅读理解能力，让自己在未来的路上走的更远一些。光有鸡汤还不够，需要有理论进行支撑，我打算采用<https://zhuanlan.zhihu.com/p/78328408>中提到的方法进行论文的阅读，文章中有一些论文总结框架可能跟计算机专业框架不太一样，所以我这里也尽量摸索，总结适合于深度学习领域的框架。



## 1. 目的

> This paper explores a pragmatic approach to multiple object tracking where the main focus is to associate objects efficiently for online and realtime applications.

文章的目的是：

- 更有效地关联检测出的目标
- 实现在线实时跟踪

## 2. 竞品分析

竞品分析：

- batch based tracking approaches:
    - The way they move: Tracking multiple targets with similar appearance
    - Joint Probabilistic Data Association Revisited
    - Multiple Hypothesis Tracking Revisited
- 使用模拟运动的方法： modelling the motion
    - The way they move: Tracking multiple targets with similar appearance
    - Bayesian Multi-Object Tracking Using Motion Context from Multiple Objects
- 使用物体表征信息： apperance of objects
    - Multiple Hypothesis Tracking Revisited
    - “ALEx-TRAC: Affinity Learning by Exploring Temporal Rein-forcement within Association Chains,”
- 常用的成熟的数据关联方法：
    - Multiple Hypothesis Tracking MHT:
        - Multiple Hypothesis Tracking Revisited
        - An Algorithm for Tracking Multiple Targets,
    - Joint Probabilistic Data Association JPDA
        - Joint Probabilistic Data Association Revisited

## 3. 方法

> Despite only using a rudimentary combination of familiar techniques such as the Kalman Filter and Hungarian algorithm for the tracking components, this approach achieves an accuracy comparable to state-of-the-art online trackers.

MOT领域解决的关键问题还是数据关联(data association)

SORT使用的方法是：

- Tracking-by-Detection（TBD）类别的方法：基于检测进行目标跟踪, 只根据**当前和上一帧**得到的内容进行跟踪。
- 强调加强**实时跟踪**，并且引入自动驾驶领域的**行人跟踪**进行提升
- 之前的跟踪方法只是基于Bounding Box的**位置和大小**进行数据关联和移动估计(motion estimation),忽略了appearance feature
- 忽视短期/长期的遮挡
- 不考虑使用物体重识别方法（object reid），因为会影响到检测速度（sort目的在于尽可能的快）

- 使用了faster rcnn作为目标检测器，使用了基于VOC数据集训练得到的权重，只关心person这个类，置信度阈值为0.5来判断是否是一个人

- 运动预测模型：使用了卡尔曼滤波算法进行预测，使用了线性恒速运动模型进行位置预测。如果没有检测框对应目标，那就使用线性恒速运行模型对其进行简单的位置预测而不需要修正。

- 数据关联模型：使用了匈牙利算法进行数据关联，其中匈牙利算法中框与框之间的距离由IOU来决定的，但是必须大于$IOU_{min}$ ，如果小于该阈值。

- ID的创建与销毁：目标进入或者离开图片会导致ID的创建和销毁，如果有检测框与现存在的检测框之间的IOU小于$IOU_{min}$的时候，就认为他是一个新的目标框（而sort只关心前一帧与当前帧，如果检测器得到的偏差较大，那在SORT中可能就会被认为是新的目标，从而分配新的ID）

    > 如果连续 $T_{lost}$ 帧没有实现已追踪目标预测位置和检测框的IOU匹配，则认为目标消失。实验中设置 $T_{lost}=1$ ，原因有二，一是匀速运动假设不合理，二是作者主要关注短时目标追踪。另外，尽早删除已丢失的目标有助于提升追踪效率。但是，问题就出现了，这样的话目标的ID一定会频繁的切换，这样就会造成跟踪计数的不准确！
    >
    > 知乎作者：TeddyZhang

## 4. 结果以及分析

![](https://img-blog.csdnimg.cn/20191217192211383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

从结果可以看出，SORT的MOTA在Online算法中比较高，缺点在于ID sw太高，实际测试的时候也会发现，其ID频繁的翻转。

![](https://img-blog.csdnimg.cn/20191217200703527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

通过比较Accuracy和MOTA这两项，SORT遥遥领先。

## 5. 结论

使用了Faster RCNN来进行模型的检测，并使用Kalman滤波预测状态，基于检测框位置和IOU的匈牙利算法，使得算法有很高的效率。

跟踪质量严重依赖于检测结果，ID sw比较严重，难以支持实际应用。

## 6. 总结

- **最重要想要记住的观点，或者将来会引用的**
  - SORT 主要由检测器，卡尔曼滤波算法，匈牙利算法匹配三个重要部分组成：
  - 检测器：必须精度足够高，能够维持框，因为SORT只用到了当前帧与前一帧的检测结果，检测器不灵敏很容易出问题。
  - 卡尔曼滤波：使用了基于坐标框的卡尔曼滤波算法，做了一个比较好的应用。
  - 匈牙利算法匹配：这个部分比较关键，采用何种方法对检测框与目标框的匹配？这里采用的是IOU单纯为了速度能更快，但是确实忽略了一些信息，比如表观信息appearance。

- 可能用到的结论

    - 创新性的使用目标检测、卡尔曼滤波、匈牙利算法组合而成了SORT算法，速度确实非常快
    - 其效果严重依赖于目标检测效果，能影响到18.9%
    - ID switch严重，严重影响效果。
- 可能用到的方法
    - 理解SORT的贡献，为之后理解DEEP SORT算法做准备

- 文章在研究设计上有哪些不足？有没有更好的改进方法？

    - 感觉可能使用了Faster R-CNN进行检测，可能由于他是一个二阶段的检测器，效果相对一阶段要好一点，但是奇怪为何速度如此之快，比一阶段的YOLOv3还要快。
    - SORT是一个Online的方法，但是仅仅依靠前一帧和当前帧严重依赖了目标检测方法，如果目标检测器稍微差一点的情况下，如何更好的提升模型效果？能不能尝试使用near online方法，结合前3-5帧的框进行预测（如果目标运行不是特别迅速的情况下）。
- 文章让你想到了哪些观点类似或者是完全不同的其他文章？
    - DEEP SORT算法，之后会进行解读。
- 对文章中观点、论述、方法、讨论部分有什么想法和评价？
    - 论文中方法部分：如果能进行多个检测器的比较，更详细的说明检测器的作用就好了，这篇直接使用了Faster R-CNN，并且说了检测器对模型影响最多18.9%, 检测器这块不够充实。
    - 实验部分：评价标准有点偏激，图表上展示的结果只关心MOTA和Speed， 没有说明ID switch， ID switch明显要比其他模型要高，这一点严重的影响了效果。



## 7. 其他知识

**奥卡姆剃刀原理**：如无必要，勿增实体；切勿浪费较多东西去做用较少的东西同样可以做好的事情。

代码：<https://github.com/abewley/sort>

在自己的项目中如何使用？（上述库中sort.py复制过来，作为自己的第三方模块）

```
from sort import *

#create instance of SORT
mot_tracker = Sort() 

# get detections
...
# 注意这里的detections是类似[[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]这种格式

# update SORT
track_bbs_ids = mot_tracker.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
...
```

我这里已经实现了sort和deep sort,分别运行sort.py和deep_sort.py即可。

repo:<https://github.com/pprp/simple_deep_sort>

其中检测器使用的是<https://github.com/ultralytics/yolov3>,这个比较受欢迎，支持训练，在一直不断更新，在原版使用的检测器是解析darknet得到的yolov3.weights权重进行检测的，灵活性较差。

如果有帮助，请不吝点个star :kissing_smiling_eyes: