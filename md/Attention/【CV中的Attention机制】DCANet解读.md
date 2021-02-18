# 【CV中的Attention机制】DCANet解读

【GiantPandaCV导读】DCANet与其他文章不同之处在于，DCANet用增强其他Attention模块能力的方式来改进的，可以让注意力模块之间的信息流动更加充分，提升注意力学习的能力。目前文章还没有被接收。

本文首发于GiantPandaCV，未经允许，不得转载。

![](https://img-blog.csdnimg.cn/20210218110600518.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 1、摘要

自注意力机制在很多视觉任务中效果明显，但是这些注意力机制往往都是考虑当前的特征，而没有考虑和其他层特征进行融合（其实也有几个工作都在做融合的Attention比如BiSeNet、AFF等）。

本文提出的DCANet（Deep Connected Attention Network）就是用来提升attention模块能力的。主要做法是：将相邻的Attention Block互相连接，让信息在Attention模块之间流动。

## 2、思想

**注意力机制：** 注意力机制可以通过探索特征之间的依赖关系来得到更好的特征表示。自注意力机制在NLP和CV领域中的各个任务都得到了广泛应用。注意力机制可以分为三部分：通道注意力机制、空间注意力机制和self-attention，也有说法是可以将self-attention视为通道注意力和空间注意力的混合。

**残差链接：**从ResNet开始，到后来的DenseNet，残差链接已经成为了当前网络标配的设计，可以让网络变得更深，并且缓解了网络退化的情况。DCANet设计也参考了这种思想。

**连接注意力：** Attention模块之间的连接方式也已经有一部分的研究，比如RA-CNN是专门用于细粒度图像识别的网络架构，网络可以不断生成具有鉴别力的区域，从而实现从粗糙到细节的识别。GANet中，高层的attention特征会送往底层的特征来指导注意力学习。

## 3、核心

下图是总结了attention方面的设计思路，首先将他们分为三个阶段，分别是：

- Extraction: 特征提取阶段
- Transformation：转换阶段，将信息进行处理或者融合
- Fusion: 融合阶段，将得到的信息进行融合到主分支中

下图从左到右分别是SEblock、GEblock、GCblock、SKblock,他们都可以归结以上三个阶段。

![通常的范式](https://img-blog.csdnimg.cn/20210218212120915.png)

**Extraction**

这个阶段是从特征图上进行特征提取，规定特征图为：
$$
X\in R^{C\times W\times H}
$$
经过特征提取器g（wg是这个特征提取操作的参数，G是输出结果）:
$$
G=g(X,w_g)
$$
**Transformation**

这个阶段处理上一个阶段得到的聚集的信息，然后将他们转化到非线性注意力空间中。规定转换t为（wt是参数，T是这个阶段的输出结果）：
$$
T=t(G,w_t)
$$
**Fusion**

这个阶段整合上个阶段获取的特征图，可以表达为：
$$
X'_i=T_i\circledast \mathbf{X}_{i}
$$
其中$X'_i$代表最终这一层的输出结果，$\circledast$代表特征融合方式，比如dot-product，summation等操作。

**Attention Connection**

这是本文的核心，特征不仅仅要从当前层获取，还要从上一层获取，这时就是需要两层信息的融合，具体融合方式提出了两种：

- Direct Connection: 通过相加的方式进行相连。

![Add连接](https://img-blog.csdnimg.cn/20210218213902876.png)

- weighted Connection: 加权的方式进行相连。

![weighted连接](https://img-blog.csdnimg.cn/20210218213958458.png)

> 为何这样融合，笔者也不是很清楚，欢迎一起讨论。

![DCANet示意图](https://img-blog.csdnimg.cn/20210218214828666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)

可以看到，上边通路是两个通道注意力模块，两个模块之间会有一个Attention Connection让两者相连，让后边的模块能利用上之前模块的信息。下边通路是两个空间注意力模块，原理同上，空间注意力模块可以有也可以没有。

## 4、 实验

分类ImageNet结果，主要和上边提到的几个模块进行对比，大约有一个点的提升：

![ImageNet验证集上的结果](https://img-blog.csdnimg.cn/20210218215136107.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_1,color_FFFFFF,t_70)

ImageNet12上的几组消融实验结果：

![消融实验](https://img-blog.csdnimg.cn/20210218215430334.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)

结果可视化：

![激活特征图可视化结果](https://img-blog.csdnimg.cn/20210218220504641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)

第一行是ResNet50、第二行是CBAM-ResNet50、第三行是DCA-CBAM-ResNet50的结果。

目标检测数据集MSCOCO上的结果对比。

![MSCOCO数据集上结果展示](https://img-blog.csdnimg.cn/20210218220625744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



## 5、评价

笔者在知乎上了解到，这篇文章投ECCV被拒稿了，理由是创新点不够，提升不够大。提升暂且不提，这个创新点确实比较少，虽然实验做的很多，分析也不错，但是总体看下来核心就是前后两个模块特征相加，这种方法确实可以提升，但是单纯这样确实是不够发表顶会的，可以考虑类似ASFF、Effective Fusion Factor in FPN for Tiny Object Detection等文章的创新点。

文章连接：https://arxiv.org/pdf/2007.05099.pdf

代码实现：https://github.com/13952522076/DCANet

题外话：

笔者维护了一个有关Attention机制和其他即插即用模块的库，欢迎在Github上进行PR或者Issue。

https://github.com/pprp/awesome-attention-mechanism-in-cv



