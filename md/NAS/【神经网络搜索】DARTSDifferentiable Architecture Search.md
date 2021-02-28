# 【神经网络搜索】DARTS: Differentiable Architecture Search

【GiantPandaCV】DARTS将离散的搜索空间松弛，从而可以用梯度的方式进行优化，从而求解神经网络搜索问题。本文首发于GiantPandaCV，未经允许，不得转载。

![https://arxiv.org/pdf/1806.09055v2.pdf](https://img-blog.csdnimg.cn/20210226222235337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

[TOC]

## 1. 简介

此论文之前的NAS大部分都是使用强化学习或者进化算法等在离散的搜索空间中找到最优的网络结构。而DARTS的出现，开辟了一个新的分支，将离散的搜索空间进行松弛，得到连续的搜索空间，进而可以使用梯度优化的方处理神经网络搜索问题。DARTS将NAS建模为一个两级优化问题（Bi-Level Optimization），通过使用Gradient Decent的方法进行交替优化，从而可以求解出最优的网络架构。DARTS也属于One-Shot NAS的方法，也就是先构建一个超网，然后从超网中得到最优子网络的方法。



## 2. 贡献



























