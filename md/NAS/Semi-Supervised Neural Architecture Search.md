# Semi-Supervised Neural Architecture Search

 【GiantPandaCV导语】本文介绍了一篇发表于NeuIPS20发表的半监督神经网络结构搜索算法，通过在训练预测器的过程中引入半监督算法，一定程度上提升了预测器的准确率。

## 1. Info

Title: Semi-Supervised Neural Architecture Search

Author: Renqian Luo

Link: [https://arxiv.org/abs/2002.10389](https://arxiv.org/abs/2002.10389)

Date: NIPS2020

Code: [https://github.com/renqianluo/SemiNAS](https://github.com/renqianluo/SemiNAS)

## 2. Motivation

基于预测器的方法需要获取成对的网络结构-精度数据，这对资源要求非常高，因为需要将每个网络结构充分训练才能准确获取其精度。而目前出现的one-shot nas方法引入超网，通过权重共享的策略来降低计算成本，但是有一些研究发现了超网的排序关系和真实的排序关系相关性较差，即排序一致性较弱。这表示提供给控制器的训练数据的质量会比较差，网络结构对应的精度信息是不准确的。

所以计算成本和精度的准确程度存在一个权衡。为了使用有限的准确标注，半监督算法被引入来加强训练的准确率。在NAS场景下，网络结构的生成是无需任何代价的，比较容易获取。

## 3. Method

本文提出除了半监督NAS（SemiNAS）来利用无标签网络结构帮助学习的方法，具体包括：

（1）训练一些网络结构，得到其准确的精度。

（2）用训练好的精度预测器来预测大量无标签网络结构的精度

（3）将生成的网络结构-精度数据添加到原始数据中，提高精度预测器性能。

SemiNAS可以与很多算法相结合，本文采用了NAO（Neural Architecture Optimization）作为基础，增加半监督算法作为指导。

简单了解一下NAO，NAO归属于SMBO（基于顺序模型优化）类别的方法，提出了简单高效的方法使用连续优化Continuous Optimization来自动化网络架构设计。其包含三个核心部件：

- encoder将网络结构映射到连续的空间中。

- predictor将网络的连续特征作为输入，预测网络的准确率。

- decoder将连续的表征映射回网络架构本身。

![](https://img-blog.csdnimg.cn/90f9c42991734bb5835df6ac4c106786.png)

所以SemiNAS可以对NAO中的Predictor部分进行改进，通过这种计算开销和精度，通过利用大量无标签网络结构来减少实际需要训练的网络结构数量，通过训练好的精度预测器来预测无标签网络结构的精度作为为标签，然后将其与有标签网络结构结合起来，接着进一步训练以提高其准确性。

具体来讲半监督学习包括以下三个步骤：

- 生成N个网络结构并训练得到其精度，用N个有标签网络训练精度预测器。

- 生成M个无标签的网络结构，使用训练好的编码器和预测器来预测精度。

- 同时使用N个有标签和M个伪标签的网络结构进行训练精度预测其来提升性能。

详细伪代码：

![](https://img-blog.csdnimg.cn/c7db294e7f20425aa44787d13c7fed32.png)

## 4. Experiment

**NASBench101上实验结果：** 

下图展示了不同神经网络结构搜索算法在NASBench101上的性能，query戴白哦从数据集中查询的网络结构精度的数量。


![](https://img-blog.csdnimg.cn/197ccf4d24af436190559ff2ebaf73dd.png)

可以看到SemiNAS在使用有标签网络结构数量（2000）的一半就能达到93.97%的测试精度，使用全部2000个情况下达到了94.03%，这说明在NAS中引入半监督学习来加快搜索速度有很大的潜力。

**ImageNet上实验结果** 

![](https://img-blog.csdnimg.cn/55a5087f8f6e454a895638daf3e05036.png)

## 5. Revisiting

SemiNAS抓住在训练NAS中预测器过程中，很多网络结构是无标签的特征，而网络结构的获取是非常容易地，所以结合了无监督学习中最简单的方法，提升了预测器训练的准确率，可以让模型用更少的标签达到更高的准确率。

## 6. Reference

【1】高效神经网络结构搜索算法及应用 罗人千

【2】Semi-Supervised Neural Architecture Search NIPS20

