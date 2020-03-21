> 前言：SKNet是SENet的加强版，是attention机制中的与SE同等地位的一个模块，可以方便地添加到现有的网络模型中，对分类问题，分割问题有一定的提升。

## 1. SKNet

SKNet是SENet的加强版，结合了SE opetator、Merge-and-Run Mappings以及attention on inception block的产物。其最终提出的也是与SE类似的一个模块，名为SK模块, 可以自适应调节自身的感受野。

据作者说，该模块对超分辨率任务有一定的提升，并且论文中的实验也证实了在分类任务上有很好的表现。

>  SK模块核心思想就是：用multiple scale feature汇总的information来channel-wise地指导如何分配侧重使用哪个kernel的表征
>
>  -- 李翔

下图就是论文中SKNet的核心实现：

![](https://img-blog.csdnimg.cn/20200105210340547.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这里重画了SK模块示意图，详见下图，主要是根据代码内容进行修改的，重画的部分分为了三个分支，而论文中只分成了两个分支。分支也是SK模块的一个可选参数，不过考虑到多分支可能增加过多的模型参数，默认设置分支个数为2。

![](https://img-blog.csdnimg.cn/20200105210505450.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



接下来对照着上图理一遍实现思路：

原始feature map X 经过kernel size分别为3$\times$3，5$\times$5, 7$\times$7, ....以此类推的卷积核进行卷积后得到U1,U2,U3三个特征图，然后相加得到了U，U中融合了多个感受野的信息。然后得到的U是形状是[C,H,W]（C代表channel,H代表height, W代表width）的特征图，然后沿着H和W维度求平均值，最终得到了关于channel的信息是一个C×1×1的一维向量，代表的是各个通道的信息的重要程度。

之后再用了一个线性变换，将原来的C维映射成Z维的信息，然后分别使用了三个线性变换，从Z维变为原来的C，这样完成了针对channel维度的信息提取，然后使用Softmax进行归一化，这时候每个channel对应一个分数，代表其channel的重要程度，这相当于一个mask。将这三个分别得到的mask分别乘以对应的U1,U2,U3，得到A1,A2,A3。然后三个模块相加，进行信息融合，得到最终模块A， 模块A相比于最初的X经过了信息的提炼，融合了多个感受野的信息。

经过以上分析，就能理解了作者的SK模块的构成了：

- 从C线性变换为Z维，再到C维度，这个部分与SE模块的实现是一致的。
- 多分支的操作借鉴自：inception。
- 整个流程类似merge-and-run mapping。

![](https://img-blog.csdnimg.cn/20200102194942760.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这就是merge-and-run mapping中提出的三个基础模块，与本文sk虽然没有直接联系，但是都是属于先进行分支，然后在合并。

## 2. 代码实现

```python
import torch.nn as nn
import torch

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))
            
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

if __name__ == "__main__":
    t = torch.ones((32, 256, 24,24))
    sk = SKConv(256,WH=1,M=2,G=1,r=2)
    out = sk(t)
    print(out.shape)
```

查看SKConv的forward函数时，可以与上图进行对应。

## 3. 参考

sknet论文地址：<https://arxiv.org/pdf/1903.06586.pdf>

merge and run mapping: <https://arxiv.org/pdf/1611.07718.pdf>

作者知乎讲解：<https://zhuanlan.zhihu.com/p/59690223>

代码源自：<https://github.com/implus/SKNet>

核心代码: <https://github.com/pprp/SimpleCVReproduction/blob/master/attention/SK/sknet.py>



