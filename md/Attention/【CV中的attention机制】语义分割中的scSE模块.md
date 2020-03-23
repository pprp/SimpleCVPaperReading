
> 前言: 本文介绍了一个用于语义分割领域的attention模块scSE。scSE模块与之前介绍的BAM模块很类似，不过在这里scSE模块只在语义分割中进行应用和测试，对语义分割准确率带来的提升比较大。



提出scSE模块论文的全称是：《**Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks** 》。这篇文章对SE模块进行了改进，提出了SE模块的三个变体cSE、sSE、scSE，并通过实验证明了了这样的模块可以增强有意义的特征，抑制无用特征。实验是基于两个医学上的数据集MALC Dataset和Visceral Dataset进行实验的。



语义分割模型大部分都是类似于U-Net这样的encoder-decoder的形式，先进行下采样，然后进行上采样到与原图一样的尺寸。其添加SE模块可以添加在每个卷积层之后，用于对feature map信息的提炼。具体方案如下图所示：



![](https://img-blog.csdnimg.cn/20200106214302699.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

然后开始分别介绍由SE改进的三个模块，首先说明一下图例:

![](https://img-blog.csdnimg.cn/20200106214518416.png)

1. cSE模块：

![](https://img-blog.csdnimg.cn/20200106214506323.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这个模块类似之前BAM模块里的Channel attention模块，通过观察这个图就很容易理解其实现方法，具体流程如下:

- 将feature map通过global average pooling方法从[C, H, W]变为[C, 1, 1]

- 然后使用两个1×1×1卷积进行信息的处理，最终得到C维的向量
- 然后使用sigmoid函数进行归一化，得到对应的mask
- 最后通过channel-wise相乘，得到经过信息校准过的feature map

```python
import torch
import torch.nn as nn


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=1,
                                      bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2,
                                         in_channels,
                                         kernel_size=1,
                                         bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
        z = self.norm(z)
        return U * z.expand_as(U)


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    c_se = cSE(c)
    print("in shape:", in_tensor.shape)
    out_tensor = c_se(in_tensor)
    print("out shape:", out_tensor.shape)
```

2. sSE模块：

![](https://img-blog.csdnimg.cn/20200106221034514.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图是空间注意力机制的实现，与BAM中的实现确实有很大不同，实现过程变得很简单，具体分析如下：

- 直接对feature map使用1×1×1卷积, 从[C, H, W]变为[1, H, W]的features
- 然后使用sigmoid进行激活得到spatial attention map
- 然后直接施加到原始feature map中，完成空间的信息校准

NOTE: 这里需要注意一点，先使用1×1×1卷积，后使用sigmoid函数，这个信息无法从图中直接获取，需要理解论文。

```python
import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U) # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q # 广播机制


if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    s_se = sSE(c)
    print("in shape:", in_tensor.shape)
    out_tensor = s_se(in_tensor)
    print("out shape:", out_tensor.shape)
```

3. scSE模块：

![](https://img-blog.csdnimg.cn/20200106222528563.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

可以看出scSE是前两个模块的并联，与BAM的并联很相似，具体就是在分别通过sSE和cSE模块后，然后将两个模块相加，得到更为精准校准的feature map, 直接上代码：

```python
import torch
import torch.nn as nn


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

if __name__ == "__main__":
    bs, c, h, w = 10, 3, 64, 64
    in_tensor = torch.ones(bs, c, h, w)

    sc_se = scSE(c)
    print("in shape:",in_tensor.shape)
    out_tensor = sc_se(in_tensor)
    print("out shape:", out_tensor.shape)
```

NOTE: 没有找到官方的实现，所以就根据论文中内容，进行基于pytorch的实现。

---



这三个模块都很容易实现，可以说是仅仅比SE模块稍微复杂一点，接下来看一下实验部分：

![](https://img-blog.csdnimg.cn/20200106223359538.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

作者分别在两个数据集上使用了三个语义分割网络，以上就是结果，可以看出scSE模块可以带来2-9%的提升，相比于BAM,CBAM,SE等对分类网络带来的1%左右的提升，要好很多。

不仅如此，添加了scSE模块可以带来细粒度的语义分割提升，能够让分割边缘更加平滑，在医学图像分割领域效果很好。

![](https://img-blog.csdnimg.cn/20200106223724371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

> 后记：接触这篇文章是在知乎一个分享kaggle图像分割竞赛的文章中，拖了很长时间才开始仔细阅读这篇文章，其带来的效果确实很不错，但是实验仅限于图像分割，各位可以尝试将其添加到图像分类，目标检测等领域，对该模块进行测评。

参考文献：

论文链接：<http://arxiv.org/pdf/1803.02579v2.pdf>

核心代码：<https://github.com/pprp/SimpleCVReproduction/tree/master/attention/scSE>