# 【神经网络搜索】DARTS: Differentiable Architecture Search

【GiantPandaCV】DARTS将离散的搜索空间松弛，从而可以用梯度的方式进行优化，从而求解神经网络搜索问题。本文首发于GiantPandaCV，未经允许，不得转载。

![https://arxiv.org/pdf/1806.09055v2.pdf](https://img-blog.csdnimg.cn/20210226222235337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

[TOC]

## 1. 简介

此论文之前的NAS大部分都是使用强化学习或者进化算法等在离散的搜索空间中找到最优的网络结构。而DARTS的出现，开辟了一个新的分支，将离散的搜索空间进行松弛，得到连续的搜索空间，进而可以使用梯度优化的方处理神经网络搜索问题。DARTS将NAS建模为一个两级优化问题（Bi-Level Optimization），通过使用Gradient Decent的方法进行交替优化，从而可以求解出最优的网络架构。DARTS也属于One-Shot NAS的方法，也就是先构建一个超网，然后从超网中得到最优子网络的方法。

## 2. 贡献

DARTS文章一共有三个贡献：

- 基于二级最优化方法提出了一个全新的可微分的神经网络搜索方法。
- 在CIFAR-10和PTB（NLP数据集）上都达到了非常好的结果。
- 和之前的不可微分方式的网络搜索相比，效率大幅度提升，可以在单个GPU上训练出一个满意的模型。

笔者这里补一张对比图，来自之前笔者翻译的一篇综述：<NAS的挑战和解决方案-一份全面的综述>

![ImageNet上各种方法对比，DARTS属于Gradient Optimization方法](https://img-blog.csdnimg.cn/20201114132717357.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70#pic_center)

简单一对比，DARTS开创的Gradient Optimization方法使用的GPU Days就可以看出结果非常惊人，与基于强化学习、进化算法等相比，DARTS不愧是年轻人的第一个NAS模型。

## 3. 核心

DARTS采用的是Cell-Based网络架构搜索方法，也分为Normal Cell和Reduction Cell两种，分别搜索完成以后会通过拼接的方式形成完整网络。在DARTS中假设每个Cell都有两个输入，一个输出。对于Convolution Cell来说，输入的节点是前两层的输出；对于Recurrent Cell来说，输入为当前步和上一步的隐藏状态。

DARTS核心方法可以用下面这四个图来讲解。

![DARTS Overview](https://img-blog.csdnimg.cn/20210228164207483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

(a) 图是一个有向无环图，并且每个后边的节点都会与前边的节点相连，比如节点3一定会和节点0，1，2都相连。这里的节点可以理解为特征图；边代表采用的操作，比如卷积、池化等。

引入数学标记：

**记**  节点(特征图)为： $x^{(i)}$ 代表第i个节点对应的潜在特征表示（特征图）。

**记**  边(操作)为:  $o^{(i,j)}$ 代表从第i个节点到第j个节点采用的操作。

**记**  每个节点的输入输出如下面公式表示，每个节点都会和之前的节点相连接，然后将结果通过求和的方式得到第j个节点的特征图。

$$
x^{(j)}=\sum_{i<j} o^{(i, j)}\left(x^{(i)}\right)
$$

**记**  所有的候选操作为 $\mathcal{O}$, 在DARTS中包括了3x3深度可分离卷积、5x5深度可分离卷积、3x3空洞卷积、5x5空洞卷积、3x3最大化池化、3x3平均池化，恒等，直连，共8个操作。

(b) 图是一个超网，将每个边都扩展了N个操作，通过这种方式可以将离散的搜索空间松弛化。













## 致谢

感谢我的师兄提供的资料，以及知乎上两位大佬，他们文章链接如下：

薰风读论文|DARTS—年轻人的第一个NAS模型 https://zhuanlan.zhihu.com/p/156832334

【论文笔记】DARTS公式推导 https://zhuanlan.zhihu.com/p/73037439











