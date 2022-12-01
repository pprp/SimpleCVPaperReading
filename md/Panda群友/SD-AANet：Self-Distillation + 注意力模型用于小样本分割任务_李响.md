
## 文章目录

- 前言
- 概述
- SD-AANet 整体架构
- Self-distillation Guided Prototype Generating 模块
- Supervised Affinity Attention 机制
- 阶段总结
- 实验和可视化分析
- 结论
- 参考链接
- 同系列的文章

## 前言

本文是小样本语义分割系列的第四篇解读，每一篇的方法都具有代表性且不同，同系列的文章链接也在文末给出了（分别是 CWT-for-FSS、GFS-Seg 和 CD-FSS）。很多读者在刚学习小样本时可能觉的真的只需要少量样本就可以完成全部的学习过程，这不完全正确，实际上在训练的过程中我们仍然需要大量的样本，只不过我们在测试的时候，我们可以对未曾在训练集中出现过的测试图像类只用几张甚至一张 Support 图像（或者理解为在推理过程中用到的训练图像）来达到对所谓的 unseen 类的分割。而传统的图像分割网络是需要在训练集中也包含了测试集的类才能对测试图像进行分割，比如我想从一张有狗的测试图像里分割出狗，那么在训练集中也需要有分割狗的任务才行，小样本却不需要。

小样本分割任务有两个通用的可以去解决的问题，首先，Support 和 Query 图像之间存在特征差异导致知识迁移难，从而降低分割性能（有点废话的嫌疑）。其次，Support 图像样本少，导致 Support 特征不具有代表性，难以指导高质量的 Query 图像分割。那么如何捕获代表性的特征，并巧妙的减小 Query 图像的特征差异就是 A Self-Distillation Embedded Supervised Affinity Attention Model for Few-Shot Segmentation 这篇文章的动机。同时，这篇解读开始有些概念有些晦涩，但是后面分解开来看就很容易理解了。

## 概述

文章提出了 Self-Distillation 引导的原型模块（SDPM，self-distillation guided prototype module），也就是通过 Support 和 Query 图像之间的 Self-Distillation 来提取内在原型（intrinsic prototype），以捕获代表性的特征，来解决我们前言中提到的问题。受监督的亲和力注意模块（SAAM，supervised affinity attention module）用 Support 的金标准来指导产生高质量的 Query 注意力图，既可以学习亲和力信息来关注 Query 目标的整个区域。这里我的初步理解是：结合 Support 的交叉熵监督信息和可训练的亲和力特征提取器，有助于形成高质量的注意力 map，告诉解码器关注哪里。

这里有些云里雾里是没关系的，后面我们会对 SDPM 和 SAAM 详细介绍，现在先来整体看下 SD-AANet 的思路，如下图所示。SD-AANet 包含四个处理：分别是编码器；SDPM；SAAM 和解码器。编码器是一个 CNN backbone，用于提取输入的特征。SDPM 通过引入 Self-Distillation 的方法获得内在原型，SAAM 学习 Query 注意力。最后，将内在原型、 Query 特征和 Query 注意力的融合向量输入到解码器，输出 Query 图像的分割结果。

![请添加图片描述](https://img-blog.csdnimg.cn/89a056fe90834b59b5510b63a5e0bc2a.png)


## SD-AANet 整体架构

经过上一部分的概述，再来看方法中更多的细节，下图是 SD-AANet 的整体架构（概述图的细化），在 SD-AANet 中，将 CNN backbone 提取 Support 和 Query 图像的特征，和 Support 图像标签的中间层特征一起被输入到 SDPM 和 SAAM。SDPM 使用 Support 原型来实现对 Support 和 Query 特征的通道重新赋权，输出下图的 Query Reweighing Feature 。然后 SDPM 中的 Self-Distillation 方法产生内在的 Support 原型。而 SAAM 将 Support 标签监督的引入可学习金字塔特征提取器，通过 CE loss 不断调整，可以产生高质量的 Query 注意力图（下图的 Affinity Attention Map）。最后，内在原型、查询特征（Query Reweighing Feature）和 Query 注意力图做融合，通过解码器来预测分割结果，实现小样本分割。

![请添加图片描述](https://img-blog.csdnimg.cn/2033039ee27a4b43a297ab7e6b45a4da.png)


在对 SD-AANet 整体架构有了初步的认识之后（如何实现 Few-shot 分割），下面两部分是我们需要重点研究的地方，首先关注 SDPM 的实现。

## Self-distillation Guided Prototype Generating 模块

先来想一个问题，一个对象往往存在两类特征，一类是普遍存在于该类所有对象中的内在特征，另一类是在不同对象中可能存在的独特特征。以飞机为例，所有的飞机都是由金属制成的，并且有翅膀，这些存在于所有飞机中的特征可以被看作是内在的特征。由于拍摄角度和光线条件的不同，飞机的形状和颜色也会不同，所以这些都是独特的特征。在小样本分割中，我们需要提取 Support 图像和含有丰富内在特征的 Query 图像的代表性特征。

Self-distillation Guided Prototype Generating 模块的设计思路如下图，SDPM 首先应用金标准（还是这个说法看起来更易懂）的 GAP（全局平均池化） 生成 Support 原型，原型一般指由嵌入特征图的金标准引导的全球平均池化计算得到的权重向量，这种跨特征通道的矢量压缩鉴别信息用于指导 Support 图像和 Query 图像之间的特征比较，以进行语义分割。然后使用 Support 原型生成通道重权向量，对应下图的 Channel Reweighting 操作。Support 特征和 Query 特征的通道都通过上述的重权向量做重新赋权。之后，新的 Support 原型和 Query 原型分别再由金标准的 GAP 生成，然后在两个原型之间使用 Self-distillation 方法来生成内在支持原型。为了提升模型的学习能力，Self-distillation 方法中的教师向量是 Support 原型和 Query 原型的平均值，如下图的蓝色虚线框所示。最终，SDPM 的输出是 Query 通道的重复权特征和 Support 原型，如下图红色虚线框所示。

![请添加图片描述](https://img-blog.csdnimg.cn/c85345eede1b4ed0ae3cc18f0500d1eb.png)


上面的叙述中，最让人疑惑的地方是 Channel Reweighting 操作，下图是 Support 引导的挤压和激发（SSE）模块（只是借鉴 SE 的方式），下图（a）在 Support 特征上应用金标准的 GAP 来生成 Support 原型，然后 Support 原型经过 FC 层来产生通道重赋权向量，同时包含 ReLU 和 Sigmoid。在下图（b）中，用这个向量同时对 Support 和 Query 特征的通道进行重新加权，然后将重新加权的特征和原始特征的平均值作为最终结果，注意 Query Feature 是不输入到 SSE Block 中的。这样做，也可以减小 Support 和 Query 图像之间存在特征差异，并且 Support Feature 对 Query Feature 有指导作用。

![请添加图片描述](https://img-blog.csdnimg.cn/9504763a4656445e85c8dcba066c8440.png)


分析过 Channel Reweighting 后，怎么自蒸馏的也是值得注意的。如下图是在 K-shot 任务中 SDPM 的两种策略：（a）使用 K 个复权向量的平均值和金标准的 GAP 来产生教师原型到最下面的 Query，指导 Support 的训练（KD Loss），图中没有给出 Support 的金标准图示。如果（a）是整体的，那么（b）的自蒸馏则是分散的，既为每个 Support 原型产生一个专属教师原型，这个看图示会更清晰。
![请添加图片描述](https://img-blog.csdnimg.cn/a531f08afaf84567a4f86c78e972839a.png)



## Supervised Affinity Attention 机制

这个设计的动机很简单，注意力机制可以有效地捕捉物体的位置。现在很多注意力机制是免训练的，比如，PFENet 使用 Support 图像和 Query 图像的高级特征来生成 Query 注意力图，通过采用 ImageNet 的预训练模型作为骨干并固定其权重。这篇文章的思路则是有监督的注意力。首先还是利用金标准的 GAP 来获得 Support 原型，并将其扩展到与 Support 特征相同的空间形状，那么下图 Query Feature 的空间形状自然也相同了，于是可以就可以把扩展后的原型与 Support 特征和 Query 特征相连接，假设将其结果分别定义为 FC,s 和 FC,q。随后，FC,s 和 FC,q 分别被输入到金字塔特征提取器，关于金字塔特征提取器这里就不过多介绍了，既 Pyramid Pooling Module（PPM）。在 PPM 的头部，有两个 1×1 的卷积层，分别生成 Support 预测和 Query 注意力图。通过不断的与 Support Mask 的训练学习（CE loss，这里说明注意力机制是有监督的），来调整 Affinity Attention Map。Support 预测是由 1×1 卷积层产生的，有两个输出通道。用于生成 Query 注意力图的 1×1 卷积层只有一个输出通道，在代码里有说明。

![请添加图片描述](https://img-blog.csdnimg.cn/1add5be022f848d1944634b39c8f3d3c.png)


## 阶段总结

以上就是 Self-distillation Guided Prototype Generating 模块和 Supervised Affinity Attention 机制的实现，这部分简单的总结下，避免懵逼。来关注下 Self-Distillation 如何嵌入到 SAAM 中，也就是在 SDPM 和 SAAM 的基础上，提出的自蒸馏嵌入式监督亲和力网络（SD-AANet）。使用应用通道重赋权和自蒸馏方法来提取内在的原型，通道重赋权有助于抑制背景特征的通道，增强前景特征的通道。自蒸馏法采用 Query 原型来指导 Support 原型（比较符合直觉），这大大缓解了 Support 特征和 Query 特征之间的差距。同时，生成的内在原型可以实现更高质量的 Query 分割。SAAM 将 Support 预测监督引入到基于可学习的 CNN 架构中，根据 Support 和 Query 之间的亲和力获得 Query 注意力图。最后对上面这些有输出的结构的结果（既内在原型、查询特征（Query Reweighing Feature）和 Query 注意力图），做了Fusion，再经过解码器，得到分割图。

关于损失函数，整个模型的损失函数用 CE loss，SDPM 用的 KD（knowledge-distillation loss），SAAM 也用的 CE，三者平衡之后求和即可。

## 实验和可视化分析

数据集使用 PASCAL-5i 和 COCO-20i（有数据集的疑问可以参考：https://blog.csdn.net/qq_38932073/article/details/115054262），消融实验在 PASCAL-5i 上：
![请添加图片描述](https://img-blog.csdnimg.cn/5f77537c72794ebf8dc94424616c94a8.png)


下图是 Support 原型的可视化消融比较研究，每张图包含了从相同的 5000 个图像对中产生的 5000 个 Support 原型：

![请添加图片描述](https://img-blog.csdnimg.cn/cecf617a422547ae97d660944a153631.png)


下图是在 COCO-20i 上的与 SOTA 的比较，† 表示对图像标签做了缩放：
![请添加图片描述](https://img-blog.csdnimg.cn/f65d1d760909454aa76d30278ffe2e0e.png)


## 结论

一句话总结：这篇提出了一种新的小样本分割方法：SD-AANet。思路与现有的方法有很大不同，最重点的是实现了 Self-Distillation 学习和小样本分割的新颖组合，进一步构建了一个有监督的亲和注意力模块（SAAM），用于生成高质量的 Query 图像的先验注意图。

## 参考链接

- https://arxiv.org/abs/2108.06600
- https://github.com/cv516Buaa/SD-AANet

## 同系列的文章

- https://mp.weixin.qq.com/s/YVg8aupmAxiu5lGTYrhpCg（CWT-for-FSS）
- https://mp.weixin.qq.com/s/ztmOZkdD1LySTqZd2RngwQ（GFS-Seg）
- https://zhuanlan.zhihu.com/p/580733255（CD-FSS）
