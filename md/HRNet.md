# HRNET

HRNet作者是微软亚洲研究院的王井东老师，

在ValseWebinar《物体和关键点检测》中亲自讲解了HRNet

打通图像分类、图像分割、目标检测、人脸对齐、姿态识别、风格迁移、Image Inpainting、超分、optical flow、Depth estimation、边缘检测等网络结构。

HRNet学习高分辨率的表征



## 1. 引入

![网络结构设计思路](https://img-blog.csdnimg.cn/20200419163630831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

在人体姿态识别这类的任务中，需要生成一个高分辨率的heatmap来进行关键点检测。这就与一般的网络结构比如VGGNet的要求不同，因为VGGNet最终得到的feature map分辨率很低，损失了空间结构。

![传统的解决思路](https://img-blog.csdnimg.cn/20200419164506548.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

获取高分辨率的方式大部分都是如上图所示，采用的是先降分辨率，然后再升分辨率的方法。U-Net、SegNet、DeconvNet、Hourglass本质上都是这种结构。

![虽然看上去不同，但是本质是一致的](https://img-blog.csdnimg.cn/20200419164546242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 2. 核心

普通网络都是这种结构，不同分辨率之间是进行了串联

![不断降分辨率](https://img-blog.csdnimg.cn/20200419170010126.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

王井东老师则是将不同分辨率的feature map进行并联：

![并联不同分辨率feature map](https://img-blog.csdnimg.cn/20200419170337777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

在并联的基础上，添加不同分辨率feature map之间的交互(fusion)。

![](https://img-blog.csdnimg.cn/20200419170456726.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

具体fusion的方法如下图所示：

![](https://img-blog.csdnimg.cn/20200419170643477.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

- 同分辨率的层直接复制。
- 需要升分辨率的使用bilinear upsample + 1x1卷积将channel数统一。
- 需要降分辨率的使用strided 3x3 卷积。
- 三个feature map融合的方式是相加。

> 至于为何要用strided 3x3卷积，这是因为卷积在降维的时候会出现信息损失，使用strided 3x3卷积是为了通过学习的方式，降低信息的损耗。所以这里没有用maxpool或者组合池化。

![HR示意图](https://img-blog.csdnimg.cn/20200419202327948.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

另外在读HRNet的时候会有一个问题，有四个分支的到底如何使用这几个分支呢？论文中也给出了几种方式作为最终的特征选择。

![三种特征融合方法](https://img-blog.csdnimg.cn/20200419202655732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

(a)图展示的是HRNetV1的特征选择，只使用分辨率最高的特征图。

(b)图展示的是HRNetV2的特征选择，将所有分辨率的特征图(小的特征图进行upsample)进行concate，主要用于语义分割和面部关键点检测。

(c)图展示的是HRNetV2p的特征选择，在HRNetV2的基础上，使用了一个特征金字塔，主要用于目标检测网络。

再补充一个(d)图

![HRNetV2分类网络后的特征选择](https://img-blog.csdnimg.cn/20200419211237449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

(d)图展示的也是HRNetV2，采用上图的融合方式，主要用于训练分类网络。

总结一下HRNet**创新点**：

- 将高低分辨率之间的链接由串联改为并联。
- 在整个网络结构中都保持了高分辨率的表征(最上边那个通路)。
- 在高低分辨率中引入了交互来提高模型性能。



## 3. 效果

### 3.1 姿态识别任务上的表现

![](https://img-blog.csdnimg.cn/20200419173518732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

以上的姿态识别采用的是top-down的方法。

![COCO验证集的结果](https://img-blog.csdnimg.cn/20200419173718989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

在参数和计算量不增加的情况下，要比其他同类网络效果好很多。

![COCO测试集上的结果](https://img-blog.csdnimg.cn/20200419173846105.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

在19年2月28日时的PoseTrack Leaderboard，HRNet占领两个项目的第一名。

![PoseTrack Leaderboard](https://img-blog.csdnimg.cn/2020041917420618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 3.2 消融实验

1. 对交互方法进行消融实验，证明了当前跨分辨率的融合的有效性。

![交互方法的消融实现](https://img-blog.csdnimg.cn/20200419174429487.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

2. 证明高分辨率feature map的表征能力

![](https://img-blog.csdnimg.cn/20200419174654814.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

1x代表不进行降维，2x代表分辨率变为原来一半，4x代表分辨率变为原来四分之一。W32、W48中的32、48代表卷积的宽度或者通道数。

### 3.3 语义分割任务中的表现

![CityScape验证集上的结果对比](https://img-blog.csdnimg.cn/20200419204006512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

![Cityscapes测试集上的对比](https://img-blog.csdnimg.cn/20200419204125190.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 3.4 目标检测任务中的表现

![单模型单尺度模型对比](https://img-blog.csdnimg.cn/20200419204713242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

![Mask R-CNN上结果](https://img-blog.csdnimg.cn/20200419210227313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 3.5 分类任务上的表现

![](https://img-blog.csdnimg.cn/20200419210548978.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

>  ps: 王井东老师在这部分提到，分割的网络也需要使用分类的预训练模型，否则结果会差几个点。

![图像分类任务中和ResNet进行对比](https://img-blog.csdnimg.cn/20200419210900224.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

以上是HRNet和ResNet结果对比，同一个颜色的都是参数量大体一致的模型进行的对比，在参数两差不多甚至更少的情况下，HRNet能够比ResNet达到更好的效果。

## 4. 代码

HRNet( https://github.com/HRNet )工作量非常大，构建了六个库涉及语义分割、人体姿态检测、目标检测、图片分类、面部关键点检测、Mask R-CNN等库。全部内容如下图所示：

![](https://img-blog.csdnimg.cn/20200419175411130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

笔者对HRNet代码构建非常感兴趣，所以以HRNet-Image-Classification库为例，来解析一下这部分代码。

先从简单的入手，BasicBlock

![BasicBlock结构](https://img-blog.csdnimg.cn/20200419225550106.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



```python
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

Bottleneck:

![Bottleneck结构图](https://img-blog.csdnimg.cn/20200419225744701.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

```python
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
```

HighResolutionModule,这是核心模块。

![HighResolutionModule核心实现](https://img-blog.csdnimg.cn/20200419230013844.png)



## 5. 总结

HRNet核心方法是：在模型的整个过程中，保存高分辨率表征的同时使用让不同分辨率的feature map进行特征交互。

HRNet在非常多的CV领域有广泛的应用，比如ICCV2019的东北虎关键点识别比赛中，HRNet就起到了一定的作用。并且在分类部分的实验证明了在同等参数量的情况下，可以取代ResNet进行分类。

之前看郑安坤大佬的一篇文章[CNN结构设计技巧-兼顾速度精度与工程实现]( https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247485711&idx=1&sn=955bf5dc66e5ab173babfa7025a363ac&chksm=9f80b399a8f73a8f73bf77b80304e69a800928a75b0e751cee8523ad7480de176edf236ccb89&scene=21#wechat_redirect )中提到了一点：

>  senet是hrnet的一个特例，hrnet不仅有通道注意力，同时也有空间注意力 
>
> -- akkaze-郑安坤 

![SELayer核心实现](https://img-blog.csdnimg.cn/20200101094228695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

SELayer首先通过一个全局平均池化得到一个一维向量，然后通过两个全连接层，将信息进行压缩和扩展，通过sigmoid以后得到每个通道的权值，然后用这个权值与原来的feature map相乘，进行信息上的优化。

![HRNet一个结构](https://img-blog.csdnimg.cn/20200419220312558.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

可以看到上图用红色箭头串起来的是不是和SELayer很相似。为什么说SENet是HRNet的一个特例，但从这个结构来讲，可以这么看：

- SENet没有像HRNet这样分辨率变为原来的一半，分辨率直接变为1x1，比较极端。变为1x1向量以后，SENet中使用了两个全连接网络来学习通道的特征分布；但是在HRNet中，使用了几个卷积(Residual block)来学习特征。
- SENet在主干部分(高分辨率分支)没有安排卷积进行特征的学习；HRNet中主干部分(高分辨率分支)安排了几个卷积(Residual block)来学习特征。
- 特征融合部分SENet和HRNet区分比较大，SENet使用的对应通道相乘的方法，HRNet则使用的是相加。之所以说SENet是通道注意力机制是因为通过全局平均池化后没有了空间特征，只剩通道的特征；HRNet则可以看作同时保留了空间特征和通道特征，所以说HRNet不仅有通道注意力，同时也有空间注意力。

HRNet团队有10人之多，构建了分类、分割、检测、关键点检测等库，工作量非常大，而且做了很多扎实的实验证明了这种思路的有效性。所以是否可以认为HRNet属于SENet之后又一个更优的backbone呢？还需要自己实践中使用这种想法和思路来验证。

## 6. 参考

https://arxiv.org/pdf/1908.07919

https://www.bilibili.com/video/BV1WJ41197dh?t=508 

https://github.com/HRNet