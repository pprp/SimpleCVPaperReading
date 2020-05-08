---
title: Attention机制- Convolutional Block Attention Module(CBAM)
date: 2019-11-19 21:09:56
tags:
- 论文阅读
- Attention机制
- CV
---

> 前言：Attention机制最开始是从NLP开始火起来的，后来才遍地开花，很多地方都用到了attention机制。
>
> 第一次接触Attention机制还是在一个比赛中，一个目标检测的比赛，一个哈工大的本科生使用attention机制中的CBAM对YOLOv3进行改进，竟然能让一个一阶段的方法获得第一名的好成绩。
>
> 这篇论文发表在ECCV2018, 感觉是一个值得一读的好文章，链接：http://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

## 1. 摘要

本研究提出一个Convolutional Block Attention Module(CBAM)模块，简单并且有效的注意力模型。处理对象是中间的feature map（意思是各种基础网络提取最后一层得到的除了全连接层以外的feature）。这个模块主要从空间和通道两个维度进行推理得到注意力权重，然后与原特征图进行相乘对特征进行自适应调整。

CBAM是一个轻量级通用模块，可以被任何结构的CNN所集成，额外开销可以忽略不计，并且可以进行端到端的训练。

在分类和检测任务重，使用了CBAM可以一直提升分类和检测的表现，这体现了模块广泛的应用性。

## 2. 介绍

- 目前的研究集中于网络的三要素：深度，宽度，基数。
    - 深度：resnet, inceptionv4, inception-resnet
    - 宽度:  GoogLeNet
    - 基数：Xception, ResNeXt
- 本文集中于神经网络的另外一个要素：注意力。注意力不仅告诉我们应该集中到哪个位置，而且还提升了感兴趣区域的表示能力。
- 本文目标是通过使用关注重要特征和压制五官特征的注意力机制来提高模型的表示能力。
- 普通卷积功能是通过混合跨通道和空间的信息来提取更富信息量的特征。所以CBAM选择关注通道和空间两个方面，进行提取更有意义的特征。
- 具体实现是顺序经过通道注意力模块和空间注意力模块：
    - 通道注意力模块：学习注意什么“what”
    - 空间注意力模块：学习注意哪里“where”
- 主要贡献：
    1. 提出了CBAM模块，简单，有效，易于集成
    2. 使用CBAM进行广泛的验证。
    3. 使用多个模型在不同数据集上使用CBAM模块，并验证了模型的表现。



## 3. CBAM模块

CBAM模块有两个模块组成。

1. 通道注意力模块（Channel Attention Module）
2. 空间注意力模块（Spatial Attention Module）



### 3.1 通道注意力机制

![](https://img-blog.csdnimg.cn/2019111922173278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20191119221811390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)