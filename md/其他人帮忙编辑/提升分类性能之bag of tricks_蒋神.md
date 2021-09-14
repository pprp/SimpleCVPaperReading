# 提升分类模型acc(二)：Bag of Tricks

【GiantPandaCV导语】本篇文章是本系列的第二篇文章，主要是介绍张航的```Bag of Tricks for Image Classification with Convolutional Neural Networks```中的一些方法以及自己实际使用的一些trick。

> 论文链接:https://arxiv.org/abs/1812.01187 
> R50-vd代码: https://github.com/FlyEgle/ResNet50vd-pytorch
知乎专栏: https://zhuanlan.zhihu.com/p/409920002

## 一、前言

如何提升业务分类模型的性能，一直是个难题，毕竟没有99.999%的性能都会带来一定程度的风险，所以很多时候只能通过控制阈值来调整准召以达到想要的效果。本系列主要探究哪些模型trick和数据的方法可以大幅度让你的分类性能更上一层楼，不过要注意一点的是，tirck不一定是适用于不同的数据场景的，但是数据处理方法是普适的。

ps: 文章比较长，不喜欢长文可以直接跳到结尾看结论。

简单的回顾一下第一篇文章的结论: 使用大的batchsize训练会略微降低acc，可以使用LARS进行一定程度的提升，但是需要进行适当的微调，对于业务来说，使用1k的batchsize比较合适。



## 二、实验配置

- 模型: ResNet50, CMT-tiny
- 数据: ImageNet1k & 业务数据
- 环境: 8xV100

ps: 简单的说明一下，由于部分实验是从实际的业务数据得到的结论，所以可能并不是完全适用于别的数据集，domain不同对应的方法也不尽相同。

本文只是建议和参考，不能盲目的跟从。imagenet数据集的场景大部分是每个图片里面都会包含一个物体，也就是有主体存在的，笔者这边的业务数据的场景很多是理解性的，更加抽象，也更难。

## 三、Bag of Tricks

### 数据增强

1. 朴素数据增强

通用且常用的数据增强有```random flip```, ```colorjitter```, ```random crop```，基本上可以适用于任意的数据集，```colorjitter```注意一点是一般不给```hue```赋值。

2. RandAug

AutoAug系列之RandAug，相比autoaug的是和否的搜索策略，randaug通过概率的方法来进行搜索，对于大数据集的增益更强，迁移能力更好。实际使用的时候，直接用搜索好的imagnet的策略即可。

3. mixup & cutmix

mixup和cutmix均在imagenet上有这不错的提升，实际使用发现，cutmix相比mixup的通用性更强，业务数据上mixup几乎没有任何的提升，cutmix会提高一点点。不过两者都会带来训练时间的开销, 因为都会导致简单的样本变难，需要更多的iter次数来update，除非0.1%的提升都很重要，不然个人觉得收益不高。在物体识别上，两者可以一起使用。公式如下：

$$
\lambda = Beta(\alpha)\\
\widetilde{x} = \lambda x_{i} + (1 - \lambda) x_{j},\\
\widetilde{y} = \lambda y_{i} + (1 - \lambda) y_{j}
$$

4. gaussianblur和gray这些方法，除非是数据集有这样的数据，不然实际意义不大，用不用都没啥影响。

实验结论:
- 20% imagenet数据集 & CMT-tiny

|模型|数据集|数据增强|训练周期|acc@top-1|
|:---:|:---:|:---:|:---:|:---:|
|CMT-tiny|imagenet-train-20%-val-all|randomcrop, randomflip|120|0.55076|
|CMT-tiny|imagenet-train-20%-val-all|randomcrop, randomflip, colorjitter, randaug|120|0.59714|
|CMT-tiny|imagenet-train-20%-val-all|randomcrop, randomflip, colorjitter, mixup|300|0.60532|
|CMT-tiny|imagenet-train-20%-val-all|randomcrop, randomflip, colorjitter, cutmix|300|0.61192|

- 业务数据上(ResNet50)
autoaug&randaug没有任何的提升(主要问题还是domain不同，搜出来的不适用)，cutmix提升很小(适用于物体而不是理解)。

### 学习率衰减

1. warmup
  深度学习更新权重的计算公式为$W_{i} = W_{i-1} - \eta \alpha \frac{loss}{W_{i-1}}$，如果bs过大，lr保持不变，会导致Weights更新的次数相对变少，最终的精度不高。
  

要调整lr随着bs线性增加而增加，但是lr变大，会导致W更新过快，最终都接近于0，出现nan。

所以需要warmup，在训练前几个epoch，按很小的概率线性增长为初始的LR后再进行LRdecay。

2. LRdecay

笔者常用的LR decay方法一般是Step Decay，按照epoch或者iter的范围来进行线性衰减，对于SGD等优化器来说，效果稳定，精度高。

进一步提升精度，可以使用CosineDecay，但是需要更长的训练周期。

![decay](https://tva1.sinaimg.cn/large/008i3skNgy1gueusoqr3uj60r20c6dgu02.jpg)

CosineDecay公式如下:

$$
\eta_{t} = \frac{1}{2}(1 + cos(\frac{t\pi}{T}))\eta
$$
如果不计较训练时间，可以使用更暴力的方法，余弦退火算法(Cosine Annealing Decay), 公式如下:

$$
\eta_{t} = \eta_{min}^{i} + \frac{1}{2}(\eta_{max}^{i} - \eta_{min}^{i})(1 + cos(\frac{T_{cur}}{T_{i}}\pi))
$$

这里的$i$表示的是重启的序号，$\eta$表示学习率，$T_{cur}$表示当前的epoch。

![](https://tva1.sinaimg.cn/large/008i3skNgy1guevn3292ej60rs0ku75z02.jpg)

退火方法常用于图像复原等用于L1损失的算法，有着不错的性能表现。

个人常用的方法就是cosinedecay，比较喜欢最后的acc曲线像一条"穿天猴", 不过要相对多训练几k个iter，cosinedecay在最后的acc上升的比较快，前期的会比较缓慢。

![](https://tva1.sinaimg.cn/large/008i3skNgy1guf11c8b5uj612k0g00uu02.jpg)

### 跨卡同步bn&梯度累加

这两个方法均是针对卡的显存比较小，batchsize小(batchszie总数小于32)的情况。

1. SyncBN

虽然笔者在训练的时候采用的是ddp，实际上就是数据并行训练，每个卡的batchnorm只会更新自己的数据，那么实际上得到的running_mean和running_std只是局部的而不是全局的。

如果bs比较大，那么可以认为局部和全局的是同分布的，如果bs比较小，那么会存在偏差。


所以需要SyncBN同步一下mean和std以及后向的更新。

2. GradAccumulate
  
  梯度累加和同步BN机制并不相同，也并不冲突，同步BN可以用于任意的bs情况，只是大的bs下没必要用。
  
  跨卡bn则是为了解决小bs的问题所带来的性能问题，通过loss.backward的累加梯度来达到增大bs的效果，由于bn的存在只能近似不是完全等价。代码如下:
  
```python
  for idx, (images, target) in enumerate(train_loader):
  images = images.cuda()
  target = target.cuda()
  outputs = model(images)
  losses = criterion(outputs, target)

loss = loss/accumulation_steps
loss.backward()
if((i+1)%accumulation_steps) == 0:
optimizer.step()
optimizer.zero_grad()
```
    ```backward```是bp以及保存梯度，```optimizer.step```是更新weights，由于accumulation_steps，所以需要增加训练的迭代次数，也就是相应的训练更多的epoch。

### 标签平滑

LabelSmooth目前应该算是最通用的技术了

优点如下:
- 可以缓解训练数据中错误标签的影响；
- 防止模型过于自信，充当正则，提升泛华性。

但是有个缺点，使用LS后，输出的概率值会偏小一些，这会使得如果需要考虑recall和precision，卡阈值需要更加精细。

代码如下:

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

## 四、ResNet50-vd

ResNet50vd是由张航等人所提出的，相比于ResNet50，改进点如下：

1. 头部的conv7x7改进为3个conv3x3,直接使用7x7会损失比较多的信息，用多个3x3来缓解。
2. 每个stage的downsample，由```(1x1 s2)->(3x3)->(1x1)```修改为```(1x1)->(3x3 s2)->(1x1)```, 同时修改shortcut从```(1x1 s2)```为```avgpool(2) + (1x1)```。

1x1+s2会造成信息损失，所以用3x3和avgpool来缓解。

实验结论:

|模型|数据|epoch|trick|acc@top-1|
|:---:|:---:|:---:|:---:|:---:|
|R50-vd|imagenet1k|300|aug+mixup+cosine+ls|78.25%|

上面的精度是笔者自己跑出来的比paper中的要低一些，不过paper里面用了蒸馏，相比于R50，提升了将近2个点，推理速度和FLOPs几乎没有影响，所以直接用这个来替换R50了，个人感觉还算不错，最近的业务模型都在用这个。

代码和权重在git上，可以自行取用，[ResNet50vd-pytorch](https://github.com/FlyEgle/ResNet50vd-pytorch)。

## 五、结论

- LabelSmooth, CosineLR都可以用做是通用trick不依赖数据场景。
- Mixup&cutmix，对数据场景有一定的依赖性，需要多次实验。
- AutoAug，如果有能力去搜的话，就不用看笔者写的了，用就vans了。不具备搜的条件的话，如果domain和imagenet相差很多，那考虑用一下randaug，如果没效果，autoaug这个系列可以放弃。
- bs比较小的情况，可以试试Sycnbn和梯度累加，要适当的增加迭代次数。

## 六、结束语

本文是**提升分类模型acc**系列的第二篇，后续会讲解一些通用的trick和数据处理的方法，尽情关注。

























