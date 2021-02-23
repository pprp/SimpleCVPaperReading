# 【神经网络搜索】Efficient Neural Architecture Search

【GiantPandaCV导语】本文介绍的是Efficient Neural Architecture Search方法，主要是为了解决之前NAS中无法完成权重重用的问题，首次提出了参数共享Parameter Sharing的方法来训练网络，要比原先标准的NAS方法降低了1000倍的计算代价。从一个大的计算图中挑选出最优的子图就是ENAS的核心思想，而子图之间都是共享权重的。

![https://arxiv.org/pdf/1802.03268v2.pdf](https://img-blog.csdnimg.cn/20210223122801437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 1. 摘要

ENAS是一个快速、代价低的自动网络设计方法。在ENAS中，控制器controller通过在大的计算图中搜索挑选一个最优的子图来得到网络结构。

- controller使用Policy Gradient算法进行训练，通过最大化验证集上的期望准确率作为奖励reward。
- 被挑选的子图将使用经典的CrossEntropy Loss进行训练。

子网络之间的权重共享可以让ENAS性能更强大的性能，同时要比经典的NAS方法降低了约1000倍的计算代价。

## 2. 简介



