# 一张图梳理YOLOv4论文

AlexeyAB大神继承了YOLOv3, 在其基础上进行持续开发，将其命名为YOLOv4。并且得到YOLOv3作者Joseph Redmon的承认，下面是Darknet原作者的在readme中更新的声明。

![Darknet原作者pjreddie在readme中承认了YOLOv4](https://img-blog.csdnimg.cn/20200424101249538.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

来看看YOLOv4和一些SOTA模型的对比，YOLOv4要比YOLOv3提高了近10个点。

![FPS vs AP](https://img-blog.csdnimg.cn/2020042410252679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 1. 思维导图

YOLOv4总体上可以划分为两部分，一部分是讲Bag of freebies和Bag of Specials; 另外一部分讲的是YOLOv4的创新点。

![YOLOv4的思维导图](https://img-blog.csdnimg.cn/20200424193904517.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

Bag of freebies和Bag of specials涉及到的大部分trick在GiantPandaCV公众号历史文章中都有介绍，所以不一一列举，主要讲一下YOLOv4的创新点。

## 2. 创新点

1. Mosaic数据增强方法

![镶嵌数据增强方法](https://img-blog.csdnimg.cn/20200424204700343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这个方法在解析U版YOLOv3的时候就讲过了，将4张不同的图片镶嵌到一张图中，其优点是：

- 混合四张具有不同语义信息的图片，可以让检测器检测超出常规语境的目标，增强模型的鲁棒性。
- 由于BN是从四张图片计算得到的，所以可以减少对大的mini-batch的依赖。

评价：这个方法在U版YOLOv3中很早就出现了，在自己数据集上也用过，但是感觉效果并不是很稳定。笔者数据集只有一个类，所以可能不需要这种特殊的数据增强方法，欢迎各位读者通过自己的实验来验证这个数据增强方法的有效性。

2. Self-Adversarial Training

自对抗训练也是一种新的数据增强方法，可以一定程度上抵抗对抗攻击。其包括两个阶段，每个阶段进行一次前向传播和一次反向传播。

- 第一阶段，CNN通过反向传播改变图片信息，而不是改变网络权值。通过这种方式，CNN可以进行对抗性攻击，改变原始图像，造成图像上没有目标的假象。
- 第二阶段，对修改后的图像进行正常的目标检测。

评价：笔者对对抗领域不是很熟悉，不是很理解这个部分。感觉这个部分讲解不是很详细，只是给出整个过程和描述，不是很能理解。

3. CmBN

![BN、CBN、CmBN示意图](https://img-blog.csdnimg.cn/2020042421021154.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图表达的是三种不同的BN方式，理解的时候应该从左往右看，BN是对当前mini-batch进行归一化。CBN是对当前以及当前往前数3个mini-batch的结果进行归一化。而本文提出的CmBN则是仅仅在这个Batch中进行累积。

评价：在消融实验中，CmBN要比BN高出不到一个百分点。感觉影响不是很大。

4. modified SAM

![modified SAM](https://img-blog.csdnimg.cn/20200424210743518.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

SAM实际上是之前解读的<CV中的Attention机制>系列中的CBAM, CBAM含有空间注意力机制和通道注意力机制, SAM就是其中的空间注意力机制.

![CBAM中的SAM](https://img-blog.csdnimg.cn/20191129215240121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

本文将Spatial-wise Attention变为Point-wise Attention, modified SAM中没有使用pooling, 而是直接用一个卷积得到的特征图直接使用Sigmoid进行激活, 然后对应点相乘, 所以说改进后的模型是Point-wise Attention. 

评价: 作者并没有给出**改进后的SAM**和**SAM**的实验对比,所以并不清楚这个模块的性能到底怎样. 并且在yolov4.cfg中没有发现使用SAM的痕迹, 这非常奇怪..不知道作者将SAM用到了模型的哪个部分.

5. modified PANet

![Modified PANet](https://img-blog.csdnimg.cn/20200424214821486.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

PANet融合的时候使用的方法是Addition, 详解见：[CVPR 2018 PANet]( https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247485145&idx=2&sn=dbd970411f3ec2da25bf432af8400a74&chksm=9f80bc4fa8f7355924af4aec888671a31a499684aa5e4e86b4c502b7f28f2521040d7507b980&scene=21#wechat_redirect )

这里YOLOv4将融合的方法由加法改为乘法，也没有解释详细原因，但是yolov4.cfg中用的是route来链接两部分特征。



## 3. 结构

YOLOv4的模型结构笔者读了一下yolov4.cfg文件，然后根据结构画出了大体结构。

![YOLOv4简化结构图](https://img-blog.csdnimg.cn/20200424193849660.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

其中没有详细展开backbone部分，其实backbone之前在解读CSPNet的时候就讲过了，YOLOv4使用的是CSPDarknet53作为Backbone。

## 4. 总结

读了这篇文章以后，给人留下深刻印象的不是创新点，而是Bag of freebies和Bag of specials。所以有人多人都说YOLOv4是拼凑trick得到的。YOLOv4中Bag of freebies和Bag of Specials两部分总结的确实不错，对研究目标检测有很大的参考价值，涵盖的trick非常广泛。但是感觉AB大神并没有将注意力花在创新点上，没有花更多篇幅讲解这创新性，这有些可惜。（ASFF中就比较有侧重，先提出一个由多个Trick组成的baseline,然后在此基础上提出ASFF结构等创新性试验，安排比较合理）

此外，笔者梳理了yolov4.cfg并没有发现在论文中提到的创新点比如modified SAM, 并且通过笔者整理的YOLOv4结构可以看出，整体架构方面，可以与yolov3-spp进行对比，有几个不同点：

- 换上了更好的backbone: CSDarknet53
- 将原来的FPN换成了PANet中的FPN

结构方面就这些不同，不过训练过程确实引入了很多特性比如：

- Weighted Residual Connections(论文中没有详细讲)
- CmBN
- Self-adversarial-training
- Mosaic data augmentation
- DropBlock
- CIOU loss

总体来讲，这篇文章工作量还是非常足的，涉及到非常非常多的trick, 最终的结果也很不错，要比YOLOv3高10个百分点。文章提到的Bag of freebies和Bag of specials需要好好整理，系统学习一下。

但是缺点也很明显，创新之处描述的不够，没能有力地证明这些创新点的有效性。此外，yolov4.cfg可能并没有用到以上提到的创新点，比如SAM。

一家之言，欢迎大佬在文末留言讨论。

## 5. 参考

https://arxiv.org/pdf/2004.10934.pdf

 https://github.com/AlexeyAB/darknet 