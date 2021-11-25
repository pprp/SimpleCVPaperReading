# Bag of Tricks for Neural Architecture Search

【GiantPandaCV导语】相比于普通的分类网络，基于超网的NAS更加难以训练，会出现收敛效果较差甚至不收敛的情况。并且，基于超网的NAS还需要额外关注子网的排序一致性等问题，训练策略的选择也极为重要。AutoSlim, BigNAS等文章都花费了大量篇幅来讲解超网的训练技巧。本文是CVPR2021 Workshop中的一篇短文，作者单位是de bosch，介绍了NAS中常用的trick。

![](https://img-blog.csdnimg.cn/2021062319333084.png)

## 1. 介绍

NAS在很多问题和benchmark上都取得了SOTA的成就，比如图像分类、语义分割、目标检测等领域。但是NAS往往诟病于其训练的不稳定性，并且除了比较优化的架构，还往往添加了不透明、不公平的trick来提高性能。这样模型性能的提高很难判断是由于trick带来的还是模型结构带来的。

尤其是one-shot模型，对超参数非常敏感，并且不比随机搜索的结果更好。

很多文章虽然涉及到了训练的技巧，但是往往并不全面，甚至有些文章可能忽略了对训练技巧的说明。

所以这篇短文会讨论：

- 如何提升模型训练的稳定性。
- 如何提升模型训练的高效性。
- 如何提升模型训练的整体性能。

## 2. 基于梯度的NAS的稳定&训练one-shot模型的方法

### 2.1 weights warm-up

Gradient-based NAS（最经典的DARTS）通常是将离散的搜索空间进行连续化，使用网络架构参数α作为权重来决定各个op的重要性。

通常使用的是bi-level optimization的方法进行优化。但是这种方式可能会带来问题，即搜索空间的过早收敛。

过早收敛问题的一个**通用的Trick** 是：

- 一开始只优化网络的权重
- 在整个搜索过程经历一半以后再进行优化网络架构参数。

Sampling-based NAS也会有类似的weights warm-up的方法。One-Shot（Bender）一开始训练整个one-shot网络模型，然后逐步提高path dropout rate。TuNAS中打开一个one-shot模型中的全部候选op，然后让选择全部op的概率下降到0。

还可以将两个过程完全解耦：

- 先对one-shot模型进行完全的训练
- 然后在进行搜索过程。

### 2.2 正则化和Loss Landscape Smoothing

通过使用强大的正则化方法来平滑loss landscape可以有效稳定网络搜索的过程。

通常采用的方法有：

- drop path
- weight decay
- data augmentation
- robust loss functions (SAM谷歌的)
  - Stabilizing differentiable archi- tecture search via perturbation-based regularization


- implicitly smoothing the loss function via auxiliary connections.

  - Robustly stepping out of performance collapse without indicators.


### 2.3 Normalization 方法

在NAS中Normalization方法的选择非常关键，对于Gradient-based NAS比如DARTS来说，使用传统的Batch Norm, Layer Norm, Instance Norm, Group Norm都存在比较大的问题。因为BN中的可学习参数会导致网络架构参数的放缩，这样网络架构参数就变得没有意义了。因此在DARTS-Like的NAS算法中，往往禁用BN中的可学习参数。

甚至有一些文章表明BN对NAS结果产生负面影响，因此在网络构建过程中不使用BN。Batch Norm在与one-shot NAS算法结合的时候往往会存在问题，因为one-shot 模型往往全部保存在内存中，由于内存的限制，batch size往往比较小。

Batch Norm和基于采样的方法结合也会存在问题，因为归一化统计量会随着不同的采样路径而变化。在训练one-shot nas的过程中，一开始可能会出现训练不稳定的情况，可以以下几种方法来克服：

- 使用验证阶段的batch统计量(又称BN Calibration)
- 使用Ghost Batch Normalization
- 使用synchronized Batch Normalization完成跨GPU的BN
- NAS-FCOS使用了Group Normalization来取代BN。

## 3. NAS训练过程加速

### 3.1 Proxy代理任务

常见的加速NAS训练过程的方法就是低保真度（low fidelity）,比如搜索过程中使用更少的filter，训练更少的epoch，减少训练样本数量、降低图像分辨率、减少网络层数等方法。其中Econas研究了低保真度的影响，并评估了如何结合不同的代理来实现最佳速度，同时保持高度的排序一致性。

### 3.2 缓存功能

NAS在被用到目标检测、语义分割等领域的时候，一般可以将网络划分为几个部分，如stem、head等。在搜索过程中，如果主干网络在搜索过程中是固定的，那么其输出就可以被提前计算出来，避免不必要的计算，从而加速搜索过程。

### 3.3 Sequential搜索

不同时优化网络架构的几个组件，而是采用顺序优化的方式。最经典的当属once for all，按照分辨率-kernel size-depth-width的顺序依次优化目标。

对于目标检测的搜索问题，首先可以搜索多尺度特征提取器，然后搜索检测头。

### 3.4 预先优化搜索空间

借鉴人类的先验知识进行搜索可以帮助构建搜索空间，比如搜索空间常常基于inverted residual block构建，本质上就是优化这些block中的超参数，比如kernel size，dilation rate，expansion rate等。

但是缺点也很明显，这种预先定义的搜索空间很难发现全新的体系结构，比如transformer等。

## 4. 提升模型表现

### 4.1 在搜索过程中找到最优架构

如何在搜索过程中找到最优架构是至关重要的：

- 由于几乎所有的方法都采用低保真度估计，代理任务上的rank排名可能和真实任务上的rank排名并不一致。
- 目前还没有很好的理解权重共享机制是如何影响架构的排序。

为了减少 co-adaptation问题，Few-Shot neural architecture Search提出了使用Sub-one-shot模型的方法，每个子模型负责覆盖一部分的搜索空间。

对于Darts这种直接根据网络架构参数最大值，认为其对应的就是网络最优架构的方法最近也有很多质疑，rethink archtecture selection in differentiable NAS 中认为这种先验并没有理论依据，也未得到证实。提出了一种不同的方案来离散化网络架构，在移除op的时候，最小化性能下降的影响。

### 4.2 超参数、数据增强和其他微调

网络的性能受很多因素影响：

- data augmentation: cutout,mixup, autoaugmentation
- stochastic regularization: shake-shake等
- activation functions: Search for activation functions
- learning rate: SGDR

ICLR20一篇文章 NAS is trustratingly hard 进行了详尽的消融实验，证明了训练流程甚至要比网络的架构更加重要。

此外，搜索的超参数对one-shot NAS方法额外重要。ICLR20另一篇Nas-bench-1shot1优化了各种one-shot网络算法的超参数，找到的解决方案可以优于黑盒NAS优化器。

## 后记

NAS工程实现以及优化方式与传统普通的CNN构建方式有所不同，对工程要求更高，使用不同的Trick，不同的优化方式对超网的训练都有比较大的影响。本文刚好收集了一批这样的Trick，在工程实践方面有很大的参考价值。

ps: 近期笔者在CIFAR10数据集上测评了常见的模型，同时收集了一批Trick和数据增强方法。如果有遗漏的，欢迎在Issue中补充。

https://github.com/pprp/pytorch-cifar-tricks




