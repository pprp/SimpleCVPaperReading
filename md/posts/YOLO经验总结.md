---
title: YOLO经验总结
date: 2019-06-20 14:54:46
tags: 
- 深度学习
- 特征融合
- 调参经验
categories:
- YOLOv3
---



> YOLO 网络架构方法
>
> YOLO参数计算
>
> YOLO调参经验
>
> 学习博客总结

### 网络更改经验

输出图像的计算方法：

<center>output = （input-filter_size+2*padding）/（stride）+ 1</center>

yolo层的前一层filter计算方法：

<center>filters = (classes + 5) * 预测框的个数</center>

特征融合一般操作：

1. Route from previous layer
2. Conv it for 1~ times
3. Do upsample
4. Route from the corresponding layer with same size of feature map
5. Continue


- res结构：

filter: 128->256->128->256.....

```
[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
```

- 只改filter不改feature大小：

```
[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky
```

or

```
[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
```

- feature减半：

```
[maxpool]
size=2
stride=2
```

or

```
[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky
```

- 普通的filter变大变小：

```
[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky
```

- feature扩大为原来二倍：

```
[upsample]
stride=2
```

- 不改变任何参数：

```
[maxpool]
size=2
stride=1
```


#### 参数说明


- batch_normalize:[了解链接](https://blog.csdn.net/yujianmin1990/article/details/78764597)

- filter: 卷积核个数，也是输出通道个数

- size: 卷积核尺寸

- stride: 卷积步长

- pad: 补0


### 教程

- 思路

- 测试mAP,画图
- 看论文，准备论文相关的内容
  - 画loss图
  - 跑实验
  - 跑一下
- 通过加大 w, h 的值提升小目标检测效果
- 学习率的设置，先大后小效果比较好
- 模型剪枝
- 创造更改网络
- 更改v3_improved网络，将maxpooling换成conv
- 训练集中增加小目标可以提高小目标的检测效果
- 目前抛弃模型压缩这种想法先将基础网络部分搞懂

### 改进的Point

1. 基础网络的选择tiny, yolov2-tiny?
2. input Size的增大： 增大最大到672
3. anchor的选择，6,7,9个anchor，查看效果
4. 改进 激活函数 Leaky RELU 尝试一下，ELU, RELU

### 靠谱的思路

> 注意调参的时候将尺度关闭，batch改大，图片正常
>
> 投论文尽量投CCF上有分类的
>
> 写论文的时候如果设计一个自己的网络要进行数学上的证明

大小目标只是一个卷积的感受野的问题，yolo这方面做得还不够好，yolo是一个三分类的目标等级做的，SSD的结构更适合小目标，里边是特征金字塔，将Yolo, SSD的优点融合然后再自己网络的构造。高级做法是细粒度（待看）。

下一阶段工作：

1. FPN论文阅读+网络结构梳理
2. SSD论文阅读+网络结构梳理
3. 有时间再看一下细粒度方面的
4. yolo源码-对具体使用以及内容理解有帮助



几个小的点：

1. batch 32 64
2. max替换为conv
3. 使用有切割图片的训练集进行进一步的测试
4. 可视化研究

### 调参经验

1. learning-rate 学习速率：学习速率越小，模型收敛花费的时间就越长，但是可以提高模型精确度。一般初始设置为0.1，然后每次除以0.2或者0.5来改进，得到最终值；

2. batchsize 样本批次容量：影响模型的优化程度和收敛速度，需要参考你的数据集大小来设置，具体问题具体分析

3. weightdecay 权重衰减：用来在反向传播中更新权重和偏置，一般设置为0.005；

4. epoch-number 训练次数：包括所有训练样本的一个正向传递和一个反向传递，训练至模型收敛即可；（注：和迭代周期iteration不一样）

5. 经过交叉验证，dropout率等于0.5的时候效果最好，原因是0.5的时候dropout随机生成的网络结构最多。


ref:

https://blog.csdn.net/weixin_38437404/article/details/74927694

https://blog.csdn.net/weixin_38437404/article/details/78837176

### 学习博客总结


[yolov3实战理解cfg文件](https://blog.csdn.net/phinoo/article/details/83022101)

[卷积网络参数计算方法](https://blog.csdn.net/qian99/article/details/79008053)

[YOLOV3实战5：利用三方工具绘制P-R曲线](https://blog.csdn.net/phinoo/article/details/83025690)

[YOLOV3可视化](https://blog.csdn.net/qq_34806812/article/details/81459982)

[Yolov3可视化2](https://blog.csdn.net/oTengYue/article/details/81365185)

[Opencv yolov3](https://blog.csdn.net/qq_27158179/article/details/81915740?tdsourcetag=s_pctim_aiomsg)

[OPENCV YOLOv3 实践](https://blog.csdn.net/haoqimao_hard/article/details/82081285)

[darknet 预训练模型与cfg文件](https://pjreddie.com/darknet/imagenet/#darknet19_448)

[deeplearning.ai](https://www.deeplearning.ai/)

[Fast. ai](https://www.youtube.com/results?search_query=fast.ai)

[Yolov3修改基础网络darknet19](https://blog.csdn.net/cgt19910923/article/details/83176997?tdsourcetag=s_pctim_aiomsg)

[Yolov3网络改进以及修改](https://blog.csdn.net/sum_nap/article/details/80781587)

[准确率召回率的理解](https://www.cnblogs.com/Zhi-Z/p/8728168.html)

[YOLOv3增加网络结构](https://blog.csdn.net/sum_nap/article/details/80781587)

[yotube yolo9000](https://www.youtube.com/watch?v=GBu2jofRJtk)

[Opencv-python教程](https://www.kancloud.cn/aollo/aolloopencv/269602)

[darknet anchor设计](https://blog.csdn.net/m_buddy/article/details/82926024)

[Yolov2 可视化研究](https://blog.csdn.net/cgt19910923/article/details/80784525)

[模型剪枝总结](https://jacobgil.github.io/deeplearning/pruning-deep-learning)

[一个比较详细的yolo指南](https://medium.com/@monocasero/object-detection-with-yolo-implementations-and-how-to-use-them-5da928356035)

[coursera 课程](https://www.coursera.org/learn/convolutional-neural-networks/lecture/4Trod/edge-detection-example)

[yolov2 yolo9000](https://www.leiphone.com/news/201708/7pRPkwvzEG1jgimW.html)

[YOLO LITE轻量级](https://blog.csdn.net/ghw15221836342/article/details/84427923)

[YOLO实验总结](https://blog.csdn.net/qq_20657717/article/details/81669006)

[YOLOv3 darknet源码细节上优化](https://blog.csdn.net/u012554092/article/details/78594425)

[github darknet 可视化1](https://github.com/GZHermit/darknet-visualization_script/tree/master)

[github darknet 可视化2](https://github.com/xueeinstein/darknet-vis)

[调参模型方法](https://blog.csdn.net/u013228894/article/details/79544109)

[c++调用模型](https://blog.csdn.net/weixin_33860450/article/details/84890877)

[详细解释YOLOv3边框预测分析](https://blog.csdn.net/qq_34199326/article/details/84109828?tdsourcetag=s_pctim_aiomsg)

[opencv-python教程](https://www.kancloud.cn/aollo/aolloopencv/259610)

[pytorch使用遇到的问题以及技巧](https://oldpan.me/archives/pytorch-conmon-problem-in-training)

[learning Rate 相关调参](https://nanfei.xyz/2018/01/23/YOLOv2%E8%B0%83%E5%8F%82%E6%80%BB%E7%BB%93/#more)
