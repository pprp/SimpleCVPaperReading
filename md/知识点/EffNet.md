# EffNet: 继MobileNet和ShuffleNet之后的高效网络结构

EffNet于2018年提出，被ICIP18接收。当时MobileNetV1,MobiletNetV2, ShuffleNetV1已经被提出，随后几个月后ShuffleNetV2才被提出。所以EffNet主要**对比对象**是MobileNet和ShuffleNetV1, 针对其存在的问题进行了调整和改进，是一个非常**简单而高效**的轻量级模型，整个实现的代码量仅有**50多行**。

## 1. 介绍

与MobileNet、ShuffleNet等网络的目的类似，EffNet目标也是让模型能够在嵌入式或者移动端硬件设备上高效地运行。

本文指出了MobileNet和ShuffleNet的不足，由于基本的block和stride等超参数会对信息造成一定的损失，这种损失对于小型的网络尤其突出。EffNet提出就是为了解决以上问题，对shallow和narrow情况下的网络效果更好。

EffNet主要有两个贡献：

1. 提出了EffNet Block，将深度可分离3x3卷积改进为1x3和3x1的空间可分离卷积，在两者之间添加一维maxpooling
2. ShuffleNet和MobileNet都选择避免处理第一层，并认为第一层的计算代价已经很低。但是EffNet认为每个优化都非常重要，如果优化了除了第一层以外的其它层，那么第一层的计算量相比之下就会比较大。实验证明使用EffNet块替换第一层能够节省30%的计算量。

EffNet和MobileNet、ShuffleNet的基础模块对比如下图所示。![EffNet和MobileNet、ShuffleNet的模块对比](https://img-blog.csdnimg.cn/20200616155306163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图中ch代表通道个数，dw代表深度可分离卷积，mp代表maxpool，gc代表组卷积，shuffle代表channel shuffle操作。

（b）中MobileNet所做的工作主要是引入了深度可分离卷积，将普通的3x3卷积替换为深度可分离卷积，可以有效地降低模型的运算量。

GoogLeNet中提到的Inception结构中将普通的3x3卷积替换为连续的1x3和3x1卷积，实际完成的是卷积的空间分离操作。

（c）ShuffleNet的核心是point wise group convolution和channel shuffle两个操作，channel shuffle可以在一定程度上弥补组卷积带来的精度下降。

（a）中EffNet的基础模块中使用了空间可分离卷积、深度可分离卷积、最大化池化等来降低计算量。

## 2. 代码

具体代码如下，核心就是make_layers函数：

```python
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
     
class EffNet(nn.Module):

    def __init__(self, nb_classes=10, include_top=True, weights=None):
        super(EffNet, self).__init__()
        
        self.block1 = self.make_layers(32, 64)
        self.block2 = self.make_layers(64, 128)
        self.block3 = self.make_layers(128, 256)
        self.flatten = Flatten()
        self.linear = nn.Linear(4096, nb_classes)
        self.include_top = include_top
        self.weights = weights

    def make_layers(self, ch_in, ch_out):
        layers = [
            nn.Conv2d(3, ch_in, kernel_size=(1,1), stride=(1,1), bias=False, padding=0, dilation=(1,1)) if ch_in ==32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1,1),stride=(1,1), bias=False, padding=0, dilation=(1,1)) ,
            self.make_post(ch_in),
            # DepthWiseConvolution2D
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(1, 3),stride=(1,1), padding=(0,1), bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # DepthWiseConvolution2D
            nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1,1), padding=(1,0), bias=False, dilation=(1,1)),
            self.make_post(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, padding=(0,0), dilation=(1,1)),
            self.make_post(ch_out),
        ]
        return nn.Sequential(*layers)

    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.linear(x)
        return x
```

单独看EffNet中的layer：

```python
nn.Conv2d(3, ch_in, kernel_size=(1,1), stride=(1,1), bias=False, padding=0, dilation=(1,1)) if ch_in ==32 else nn.Conv2d(ch_in, ch_in, kernel_size=(1,1),stride=(1,1), bias=False, padding=0, dilation=(1,1)) ,
self.make_post(ch_in),
# DepthWiseConvolution2D
nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(1, 3),stride=(1,1), padding=(0,1), bias=False, dilation=(1,1)),
self.make_post(ch_in),
nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
# DepthWiseConvolution2D
nn.Conv2d(ch_in, 1 * ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1,1), padding=(1,0), bias=False, dilation=(1,1)),
self.make_post(ch_in),
nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, padding=(0,0), dilation=(1,1)),
self.make_post(ch_out),
```

make_post函数主要添加的是leaky relu和Batch Norm，所以只需要关心其中的Conv2d即可。

（1）第一个Conv2d是1x1卷积，或者Pointwise Convolution

（2）接下来是kernel size为(1,3)和(3,1)的空间分离卷积, 这两个卷积之间添加了一个一维的最大化池化层。

（3）最后一个Conv2d是kernel size为(1,2)，stride为(1,2)的卷积。

关于以上提到的卷积的综述可以看[【综述】神经网络中不同种类的卷积层](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247485284&idx=1&sn=16ecf58701053cace06a1fd631ef891d&chksm=9f80bdf2a8f734e4688ac2dc06d0bce91fdfd00cb17e10fbb422ca4b79389773c208ab8b9f44&scene=21#wechat_redirect)。

EffNet也尝试过ShuffleNet的分组卷积，虽然分组卷积带来的计算优势，但是结果表明准确度明显下降，因此EffNet也避开使用分组卷积。

## 3. 实验

EffNet并没有在ImageNet这种大型数据集上训练测试，而是选择了Cifar10, SVHN, GTSRB等较小规模的数据集进行对比。

![Cifar10上的对比结果](https://img-blog.csdnimg.cn/20200617131025174.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

![SVHN上结果对比](https://img-blog.csdnimg.cn/20200617131220754.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

![GTSRB上的结果对比](https://img-blog.csdnimg.cn/20200617131254427.png)

此外，由于这篇文章发表和MobileNetV2有一些撞车，所以单独列了一章和MobileNetV2进行对比。

![Cifar10上的结果对比](https://img-blog.csdnimg.cn/20200617131423999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 4. 评价

EffNet是在MobileNetV1,V2和ShuffleNetV1之后提出来的改进模型，文章认为MobileNet和ShuffleNet的Block设计对信息造成了一定的损耗，所以提出了一个EffNet Block的模块来弥补，并且在第一层的处理方面和MobileNet和ShuffleNet都不同，选择用EffNet Block替换掉第一层的模块。总体来看，EffNet的参考价值一般，实现的代码非常简单，知名度不够高，github上的实现只有10多个star。不过EffNet算是使用空间分离卷积比较成功的模型，不过提出的时间比较尴尬，前有MobileNetV2后有ShuffleNetV2，所以自然知名度一般了。

## 5. 参考文献

https://github.com/andrijdavid/EffNet

https://arxiv.org/abs/1801.06434