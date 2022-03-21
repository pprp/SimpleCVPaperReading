# Compact-Transformer：缓解数据不足带来的问题

【GiantPandaCV导语】本文致力于解决ViT在小型数据集上性能不够好的问题，这个问题非常实际，现实情况下如果确实没有大量数据集，同时也没有合适的预训练模型需要从头训练的时候，ViT架构性能是不如CNN架构的。这篇文章实际上并没有引入大量的卷积操作，通过修改patch size，以及使用SeqPool的方法就可以取得不错的成绩。

![](https://img-blog.csdnimg.cn/f71b9131763843938ca3d60ea70c4b7d.png)

## 引言

ViT不适用于小数据集，但是由于很多领域中数据量大小是非常有限的，为了打破ViT数据匮乏下性能不好，只能应用于大数据集的问题。本文提出使用正确的尺寸以及tokenization方法，可以让Transformer在小型数据集上达到SOTA。从头开始训练CIFAR10可以达到98%准确率。

本文中首先引入了ViT-Lite，这是一种小型结构更紧密的ViT，在其基础上添加合适大小的Patch sizeing得到Compact Vision Transformer.

> 这部分与Early Convolution Stem那篇异曲同工。


本文核心贡献如下：

- 通过引入ViT-Lite能够有效从头开始在小型数据集上实现更高精度，打破Transformer需要大量数据的神话。
- 引入新型序列池化策略（sequence pooling)的CVT（Compact Vision Transformer），从而让Transformer无需class token
- 引入CCT（Compact Convolutional Transformer）来提升模型性能，同时可以让图片输入尺寸更加灵活。



## 方法

提出了三种模型ViT-Lite, CVT, CCT。



### ViT-Lite

![](https://img-blog.csdnimg.cn/564163bc9cab42c0a5cd316942caafa3.png)

该模型与原始模型几乎相同，但是使用更小的patch size。小型数据集本身就很小，因此patch size比较重要。

### CVT

![](https://img-blog.csdnimg.cn/ef9dc5241f4f444695ffc4a050eb3234.png)

引入了**Sequential Pooling**方法，SeqPool消除了额外的Classification Token， 这个变换用T表示：$\mathbb{R}^{b \times n \times d} \mapsto \mathbb{R}^{\hat{b} \times d}$, 输入为$\mathbf{x}_{L}=\mathrm{f}\left(\mathbf{x}_{0}\right) \in \mathbb{R}^{b \times n \times d}$, 其中b表示batch size， n表示sequence length， d是embedding 维度，g代表linear layer，$\mathrm{g}\left(\mathbf{x}_{L}\right) \in \mathbb{R}^{d \times 1}$(ps：感觉有点类似channel attention)

$$
\mathbf{z}=\mathbf{x}_{L}^{\prime} \mathbf{x}_{L}=\operatorname{softmax}\left(\mathrm{g}\left(\mathbf{x}_{L}\right)^{T}\right) \times \mathbf{x}_{L} \in \mathbb{R}^{b \times 1 \times d}
$$


代码实现如下：

```
self.attention_pool = Linear(self.embedding_dim, 1)
x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
```


### CCT

![](https://img-blog.csdnimg.cn/70c450644da34a0996c4ad661fa1c93a.png)

直接引入卷积作为主干，保留了局部信息，并且能够编码patch之间的关系。

CCT-12/7x2代表：Transformer encoder有12层，使用了2个Convolution block，其中用的是7x7 大小的卷积核。

![](https://img-blog.csdnimg.cn/4c59e043382a4f6f847b699ad33d9158.png)



## 实验

小型数据集上结果：

![](https://img-blog.csdnimg.cn/8cd2fcd491e84027b88177cd9de4f611.png)

这个地方其实有点不太符合直觉，直觉上来看，卷积层数越多在小数据集上性能应该越好，但是这里发现使用一个卷积要比使用两个更好。

ImageNet结果：

![](https://img-blog.csdnimg.cn/baf1c69aabeb40fbbd923e76056463db.png)

CIFAR10上的Trade off:

![](https://img-blog.csdnimg.cn/74e3443f6c5a411da6c9605704422f38.png)

消融实验：（关注SP即Seqence Pooling操作有效性）

![](https://img-blog.csdnimg.cn/a3902f57cffd4f4eb4f3687326382965.png)

Positional Embedding的影响：

![](https://img-blog.csdnimg.cn/367dbe6c5a24432dbcdf04594bfa2f5a.png)

## 参考

[https://arxiv.org/abs/2104.05704](https://arxiv.org/abs/2104.05704)

[https://github.com/SHI-Labs/Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers)

[https://zhuanlan.zhihu.com/p/364589899](https://zhuanlan.zhihu.com/p/364589899)



