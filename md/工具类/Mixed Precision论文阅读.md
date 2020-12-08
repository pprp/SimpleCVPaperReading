# 【论文阅读】Mixed Precision Traning

【GiantPandaCV导语】混合精度是一个非常**简单并且实用**的技术，由百度和谷歌联合发表于ICLR2018，可以让模型以半精度的方式训练模型，既能够**降低显存占用**，又可以**保持精度**。这篇文章不是最先提出使用更低精度来进行训练，但是其影响力比较深远，很多现在的方案都是基于这篇文章设计的。

## 1. 摘要

提高网络模型的大小可以有效提升准确了，但是也增加了内存的压力和计算力的需求。本论文引入了半精度来训练深度神经网络，在不损失精度、不需要更改超参数的情况下，几乎可以减半内存占用。

权重、激活、梯度都是以IEEE半精度的方式存储的，由于半精度的表示范围要比单精度要小，本文提出了三种方法来防止关键信息的丢失。

- 推荐维护一个**单精度版本的权重weights**，用于累积梯度。
- 提出使用**Loss scale**来保证小的梯度值不会下溢出。
- 在加载到内存之前，就将模型转化到**半精度**来进行加速。

## 2. 介绍

大的模型往往需要更多的计算量、更多的内存资源来训练，如果想要降低使用的资源，可以通过降低精度的方法。通常情况下，一个程序运行速度的快慢主要取决于三个因素：

- 算法带宽
- 内存带宽
- 时延

使用半精度减少了一半bit，可以影响**算法带宽和内存带宽**，好处有：

- 由于是用更少的bit代替，占用空间会减小差不多一半
- 对于处理器来说也是一个好消息，使用半精度可以提高处理器吞吐量；在当时的GPU上，半精度的吞吐量可以达到单精度的2-8倍。

现在大部分深度学习训练系统使用单精度FP32的格式进行存储和训练。本文采用了IEEE的半精度格式(FP16)。但是由于FP16可以表示的范围要比FP32更为狭窄，所以引入了一些技术来规避模型的精度损失。

- 保留一份FP32的主备份。
- 使用loss scale来避免梯度过小。
- FP16计算但是用FP32进行累加。

## 3. 实现

![Mixed Precision](https://img-blog.csdnimg.cn/20201207210116687.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

下面详细讲解第二节提到的三种技术。

### FP32的主备份

![这部分示意图](https://img-blog.csdnimg.cn/20201207210716127.png)

在混合精度训练中，权重、激活、梯度都采用的是半精度存储的。为了匹配单精度网络的精度，在优化的整个过程中，需要拷贝一份**单精度模型FP32**作为主备份，而**训练过程中使用的是FP16**进行计算。

这样做的原因有两个：

第一：**溢出错误**：更新的时候是**学习率**乘以梯度，这个值可能非常小，超过半精度最小范围($2^{-24}$)，就会导致**下溢出**。如下图展示了梯度和FP16的表示范围，大约有5%的梯度已经下溢出，如果直接对**半精度表示的模型**更新，这样模型的精度肯定有所损失；而对**单精度的模型**进行更新，就就可以规避下溢出的问题。

![梯度和半精度范围](https://img-blog.csdnimg.cn/20201207211232738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

- 第二：**舍入误差**：指的是当梯度过小，小于当前区间内的最小间隔，那么梯度更新可能会失败。想要详细了解可以自行查看wiki上的详解。同样的，使用FP32单精度可以尽可能减少舍入误差的影响。

![图源知乎@瓦砾](https://img-blog.csdnimg.cn/20201207212530618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### Loss Scale

下图展示的是激活值的范围和FP16表示范围之间的差异，50%以上的值都集中在不可表示的范围中，而右侧FP16可表示范围却空荡荡的，那么一个简单的想法就是向右平移，将激活平移到FP16可表示范围内。比如说可以将激活乘以8，这样可以平移3位，这样表示最低范围就从$2^{-24}$到了$2^{-27}$, 因为激活值低于$2^{-27}$的部分对模型训练来说不重要，所以这样操作就可以达到FP32的精度。

![Loss Scale](https://img-blog.csdnimg.cn/20201207213222570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

那实际上是如何实现平移呢？很简单，就是在loss基础上乘以一个很大的系数loss scale, 那么由于链式法则，可以确保梯度更新的时候也采用相同的尺度进行缩放，最后在更新梯度之前再除以loss scale， 对FP32的单精度备份进行更新，就完成了。

![实际上操作非常简单](https://img-blog.csdnimg.cn/20201207214107365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



## 参考

https://arxiv.org/pdf/1710.03740v3

https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/

https://zhuanlan.zhihu.com/p/163493798

https://zhuanlan.zhihu.com/p/79887894