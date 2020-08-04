# CenterNet的骨干网络之DLA-34

DLA全称是Deep Layer Aggregation, 于2018年发表于CVPR。被CenterNet, FairMOT等框架所采用，并且**DLA-34得到的反响比较好**，准确率和模型复杂度平衡的比较好。

CenterNet中使用的是在DLA-34的基础上添加了Deformable Convolution后的网络，先来看看Deep Layer Aggregation的理论基础。

## 简介

Aggretation聚合是目前设计网络结构的常用的一种技术。如何将不同深度，将不同stage、block之间的信息进行融合是本文探索的目标。

目前常见的聚合方式有skip connection, 如ResNet，这种融合方式仅限于块内部，并且融合方式仅限于简单的叠加。

本文提出了DLA的结构，能够迭代式地将网络结构地特征信息融合起来，让模型有更高的精度和更少的参数。

![DLA的设计思路](https://img-blog.csdnimg.cn/20200804202908321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图展示了DLA的设计思路，Dense Connections来自DenseNet，可以聚合语义信息。Feature Pyramids空间特征金字塔可以聚合空间信息。DLA则是将两者更好地结合起来从而可以更好的获取what和where的信息。仔细看一下DLA的其中一个模块，如下图所示：

![DLA其中一个Tree结构](https://img-blog.csdnimg.cn/20200804203952451.png)

研读过代码以后，可以看出这个花里胡哨的结构其实是按照树的结构进行组织的，红框框住的就是两个树，树之间又采用了类似ResNet的残差链接结构。

## 核心

先来重新梳理一下上边提到的语义信息和空间信息，文章给出了详细解释：

- 语义融合：在通道方向进行的聚合，能够提高模型推断“是什么”的能力（what）
- 空间融合：在分辨率和尺度方向的融合，能够提高模型推断“在哪里”的能力（where）

![DLA34完整结构图](https://img-blog.csdnimg.cn/20200804205203420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

Deep Layer Aggregation核心模块有两个IDA(Iterative Deep Aggregation)和HDA(Hierarchical Deep Aggregation)，如上图所示。

- 红色框代表的是用树结构链接的层次结构，能够更好地传播特征和梯度。

- 黄色链接代表的是IDA，负责链接相邻两个stage地特征让深层和浅层的表达能更好地融合。
- 蓝色连线代表进行了下采样，网络一开始也和ResNet一样进行了快速下采样。

论文中也给了公式推导，感兴趣的可以去理解一下。本文还是将重点放在代码实现上。

## 实现



## Reference

https://arxiv.org/abs/1707.06484