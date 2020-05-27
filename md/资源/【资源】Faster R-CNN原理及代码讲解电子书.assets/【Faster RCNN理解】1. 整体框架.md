# 【Faster R-CNN】1. 整体框架

Faster R-CNN是R-CNN系列中第三个模型，经历了2013年Girshick提出的R-CNN、2015年Girshick提出的Fast R-CNN以及2015年Ren提出的Faster R-CNN。

Faster R-CNN是目标检测中较早提出来的两阶段网络，其网络架构如下图所示：

![](https://img-blog.csdnimg.cn/20200205215726626.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

可以看出可以大体分为四个部分：

1. Conv Layers 卷积神经网络用于提取特征，得到feature map。
2. RPN网络，用于提取Region of Interests(RoI)。
3. RoI pooling, 用于综合RoI和feature map, 得到固定大小的resize后的feature。
4. classifier, 用于分类RoI属于哪个类别。





### 1. Conv Layers

在Conv Layers中，对输入的图片进行卷积和池化，用于提取图片特征，最终希望得到的是feature map。在Faster R-CNN中，先将图片Resize到固定尺寸，然后使用了VGG16中的13个卷积层、13个ReLU层、4个maxpooling层。（VGG16中进行了5次下采样，这里舍弃了第四次下采样后的部分，将剩下部分作为Conv Layer提取特征。）

与YOLOv3不同，Faster R-CNN下采样后的分辨率为原始图片分辨率的1/16（YOLOv3是变为原来的1/32）。feature map的分辨率要比YOLOv3的Backbone得到的分辨率要大，这也可以解释为何Faster R-CNN在小目标上的检测效果要优于YOLOv3。



### 2. Region Proposal Network

简称RPN网络，用于推荐候选区域（Region of Interests），接受的输入为原图片经过Conv Layer后得到的feature map。

![](https://img-blog.csdnimg.cn/20200209174933951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图参考的实现是：https://github.com/ruotianluo/pytorch-faster-rcnn

RPN网络将feature map作为输入，然后用了一个3x3卷积将filter减半为512,然后进入两个分支：

一个分支用于计算对应anchor的foreground和background的概率，目标是foreground。

一个分支用于计算对应anchor的Bounding box的偏移量，来获得其目标的定位。

通过RPN网络，我们就得到了每个anchor是否含有目标和在含有目标情况下目标的位置信息。

**对比RPN和YOLOv3:**

都说YOLOv3借鉴了RPN，这里对比一下两者：

**RPN:** 分两个分支，一个分支预测目标框，一个分支预测前景或者背景。将两个工作分开来做的，并且其中前景背景预测分支功能是判断这个anchor是否含有目标，并不会对目标进行分类。另外就是anchor的设置是通过先验得到的。

**YOLOv3:**将整个问题当做回归问题，直接就可以获取目标类别和坐标。Anchor是通过IoU聚类得到的。

区别：Anchor的设置，Ground truth和Anchor的匹配细节不一样。

联系：两个都是在最后的feature map（w/16,h/16或者w/32，h/32）上每个点都分配了多个anchor，然后进行匹配。虽然具体实现有较大的差距，但是这个想法有共同点。

### 3. ROI Pooling

这里看一个来自deepsense.ai提供的例子：

RoI Pooling输入是feature map和RoIs：

假设feature map是如下内容：

![](https://img-blog.csdnimg.cn/20200209183030343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

RPN提供的其中一个RoI为：左上角坐标（0,3)，右下角坐标（7,8）

![](https://img-blog.csdnimg.cn/20200209183241113.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

然后将RoI对应到feature map上的部分切割为2x2大小的块：

![](https://img-blog.csdnimg.cn/20200209183428731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

将每个块做类似maxpooling的操作，得到以下结果：

![](https://img-blog.csdnimg.cn/20200209183556994.png)

以上就是ROI pooling的完整操作，想一想**为何要这样做**？

在RPN阶段，我们得知了当前图片是否有目标，在有目标情况下目标的位置。现在唯一缺少的信息就是这个目标到底属于哪个类别（通过RPN只能得知这个目标属于前景，但并不能得到具体类别）。

如果想要得知这个目标属于哪个类别，最简单的想法就是将得到的框内的图片放入一个CNN进行分类，得到最终类别。这就涉及到最后一个模块：classification



### 4. Classification

ROIPooling后得到的是大小一致的feature，然后分为两个分支，靠下的一个分支去进行分类，上一个分支是用于Bounding box回归。如下图所示（来自知乎）：



![](https://img-blog.csdnimg.cn/20200209184617310.png)

分类这个分支很容易理解，用于计算到底属于哪个类别。Bounding box回归的分支用于调整RPN预测得到的Bounding box，让回归的结果更加精确。



### 5. 总结

本文大概梳理了一下四个模块，最起码可以留一个大体的印象，了解各个模块的作用。其中有很多细节，之后几天会翻译一篇来自telesens的博客，写的非常的详细，值得我们仔细阅读。



### 6. 参考内容

文章链接：<https://arxiv.org/abs/1504.08083>

博客：<http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/>

代码：https://github.com/ruotianluo/pytorch-faster-rcnn

ROI pooling:<https://deepsense.ai/region-of-interest-pooling-explained/>

Classification图示：<https://zhuanlan.zhihu.com/p/31426458>