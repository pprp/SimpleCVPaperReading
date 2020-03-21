图像预处理+网络结构组织

翻译1：



本文依据的Faster R-CNN具体实现为：https://github.com/ruotianluo/pytorch-faster-rcnn

**文章组织：**

**第1节：** **图像预处理** 主要包括减去平均像素值并缩放图像。训练和测试过程对图像的是的预处理步骤必须完全相同。

**第2节：** **网络组织** 主要描述网络的三个主要的组成部分，Head、RPN和Classification。

**第3节：** **训练细节** 详细描述了训练R-CNN网络所涉及到的步骤。

**第4节：** **测试细节** 中将介绍测试过程中涉及的步骤，使用经过训练的R-CNN来进行目标检测。

**附录：** 介绍R-CNN中常用的算法细节，比如非极大抑制和ResNet50架构细节。

### 1. 图像预处理

在图片进入CNN前，应该实施一下预处理步骤。在训练和测试的过程中图片必须经过相同的预处理过程。其中减均值中的均值不是当前处理图片的均值，而是所有训练和测试图片的均值。

![](https://img-blog.csdnimg.cn/20200211194253807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

targetSize和maxSize两个参数的默认值分别是600和1000。

**预处理描述：**

1. 读取当前图片

2. 将当前图片减去全体训练和测试图片的均值

3. 进行图片缩放

    1. targetSize限制的是图片最小边的边长
    2. maxSize限制的是图片最大边边长
    3. 具体算法如下所示：

```python
# 具体算法
w = image.width
h = image.height
minDim = min(w,h)
maxDim = max(w,h)
scale = targetSize/minDim
if scale * maxDim > maxSize:
    scale = maxSize/maxDim
im = rescale(img, scale)
```

通过以上步骤，就可以将图片的长和宽都限制在[targetSize, maxSize]之间。

### 2. 网络组织

R-CNN使用CNN来解决两个主要问题：

- 识别输入图像中可能包含前景对象的区域（感兴趣区域– ROI）
- 计算每个ROI的对象类别概率分布–即计算ROI包含某个类别的对象的概率。然后，可以选择概率最高的对象类别作为分类结果。

R-CNN由三种主要类型的网络组成：

1. Head
2. 区域建议网络（RPN）
3. Classification

R-CNN使用诸如ResNet50之类的预训练网络的前几层来从输入图像中得到feature map。由于CNN可以进行迁移学习 ，因此可以使用在一个数据集上针对不同问题训练的网络（进行finetune）。网络的前几层学习检测常规特征，例如边缘和颜色斑点，这些特征可以很好地区分许多不同问题。后面几层网络学习的功能是更高级别的，更多与问题相关的功能。可以去除这些层，或者可以在反向传播期间微调这些层的权重。

从预训练网络初始化的前几层构成“Head”网络。然后，将由Head网络生成的feature map通过RPN，该RPN使用一系列卷积和完全连接的层来产生可能包含前景对象的ROI。这些ROI之后用于从头部网络生成的特征图中裁剪出相应的区域。这称为“Crop Pooling”(RoI Pooling)。然后，将RoI pooling产生的区域通过分类网络，该网络将对每个ROI中包含的对象进行分类。

**网络架构：**

下图显示了上述三种网络类型的各个组件。图中显示了每个网络层的输入和输出的维度，这有助于理解网络的每个层如何转换和处理数据。w和h代表输入图像的宽度和高度（经过预处理）。

![](https://img-blog.csdnimg.cn/20200211194310898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图真的是见过最详细的Faster R-CNN的细节图了，主要分为3个部分，梳理一下：

1. Head网络，在左下角，进行4次下采样，得到feature map，shape=[w/16,h/16,1024]。
2. RPN网络，在左上角，经过一个卷积、relu后通道变为512，然后经过两个分支，靠上的分支负责预测对应anchor属于前景还是背景；靠下的分支负责预测对应anchor的位置，四个值。
3. Classification网络，在右下角，输入为Head网络得到的feature map和RPN提供的RoIs，通过RoI Pooling后得到Classification的输入。RoI pooling中有Pooling Size参数来控制具体Pooling的尺寸。然后经过卷积和average pooling操作，得到形状为[n, 2048]大小的张量，然后分别送入两个分支：
    1. cls_score_net用于分类
    2. bbx_pred_net用于定位回归

### 3. 训练细节

在本节中将详细描述训练R-CNN所涉及的步骤。一旦了解了训练的工作原理，就可以轻松理解测试过程，因为它仅使用了训练中涉及的部分步骤。训练的目的是调整RPN和分类网络中的权重，并微调Head网络的权重（这些权重是从诸如ResNet之类的预训练网络中初始化的）。

回想一下，RPN网络的工作是产生ROI，而分类网络的工作是为每个ROI分配对象类别得分。因此，为了训练这些网络，我们需要相应的Ground Truth，即图像中存在的对象周围的边界框的坐标以及这些对象的类别。

这个Ground Truth来自免费使用的数据集，数据集中每张图片都带有一个注释文件。该标注文件包含图像中存在的每个对象的边界框的坐标和对象类别标签（这些对象类来自预定义的对象类列表）。这些图像数据库已用于支持各种对象分类和检测挑战。两种常用的数据库是：

![](https://img-blog.csdnimg.cn/20200211194321753.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

- PASCAL VOC：**VOC 2007**数据库包含9963个训练/验证/测试图像，其中包含针对20个对象类的24,640个标注文件。

    - **人：**  人
    - **动物：**  鸟，猫，牛，狗，马，绵羊
    - **车辆：**  飞机，自行车，轮船，公共汽车，汽车，摩托车，火车
    - **室内：**  瓶子，椅子，餐桌，盆栽，沙发，电视/显示器
- COCO（常见目标）：COCO数据集要大得多。它包含具有90个对象类别的> 200K标记图像。

本文将较小的PASCAL VOC2007数据集用于训练。R-CNN能够在同一步骤中训练RPN和Classification网络。

让我们花一点时间来探讨在本文的其余部分中广泛使用的“**边界框回归系数**”和“**边界框重叠**”的概念。



**边界框回归系数**（也称为“回归系数”和“回归目标”）：R-CNN的目标之一是得到与对象边界紧密匹配的边界框。R-CNN通过采用给定的边界框（由左上角的坐标，宽度和高度定义）并通过应用一组“回归系数”来调整其左上角，宽度和高度来生成这些边界框。

分别将目标和原始边界框的左上角的x，y坐标分别表示为$T_x,T_y,O_x,O_y$。目标和原始边界框的宽度/高度分别表示为$T_w,T_h,O_w,O_h$。然后，回归目标（将原始边界框转换为目标框的函数的系数）给出为：

$$
t_x = \frac {(T_x-O_x)} {O_w}，t_y = \frac {(T_y-O_y)} {O_h}，t_w = log(\frac {T_w} {O_w})，t_h = log(\frac {T_h} {O_h})
$$

此函数很容易求逆，即，给定左上角的回归系数和坐标 以及 原始边界框的宽度和高度，可以轻松计算目标框的左上角和宽度及高度。注意，回归系数对于没有剪切的仿射变换是不变的。这一点很重要，因为在计算分类损失时，目标回归系数**以原始纵横比**计算，而分类网络输出回归系数则是在正方形特征图（纵横比为1:1）上的ROI合并步骤之后计算的。当我们在下面讨论分类损失时，这将变得更加清楚。

![](https://img-blog.csdnimg.cn/2020021123162831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

**NOTE:**稍微解读一下这个图，蓝色的框是Ground Truth，红色的框是RPN得到的Bounding Box。

![](https://img-blog.csdnimg.cn/20200213093239363.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

虽然看上去这个图比较高深，其实想表达的意思是使用了tx,ty,tw,th这组边界框回归系数，然后进行**缩放和平移操作以后，这组系数依然是保持不变**的。所以也就能保证两个框之间的相对关系是不变的。上图以tx为例，进行举例，可以看出经过平移和缩放以后，tx保持不变。

**交并比：**  我们需要某种程度的度量，以使给定的边界框与另一个边界框有多接近，而该边界框与用来测量边界框尺寸的单位（像素等）无关。此度量应直观（两个重合的边界框应具有1的重叠，而两个不重叠的框应具有0的重叠）并且快速且易于计算。常用的重叠度量是“交并比”，计算方法如下所示。

![](https://img-blog.csdnimg.cn/2020021123170940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200211194331580.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

有了这些初步准备，现在让我们深入了解实施细节，以训练R-CNN。在实现中，R-CNN执行分为以下几层。每一层封装了一系列逻辑步骤，这些步骤可能涉及通过神经网络某些部分运行数据或其他步骤，如比较边界框之间的重叠，执行非最大值抑制等。

![](https://img-blog.csdnimg.cn/20200211194342409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



- **Anchor Generation Layer：**通过首先生成9个不同比例和长宽比的Anchor，然后通过在输入图像的均匀间隔的网格点上平移它们来复制这些Anchor，从而生成固定数量的“Anchor”（Bounding box）。
- **Proposal Layer：**根据边界框回归系数转换Anchor，以生成转换后的Anchor。然后，通过使用Anchor为前景区域的概率进行非极大抑制来减少Anchor的数量。（非极大抑制算法在附录中进行解释）
- **Anchor Target Layer**：目标是生成一组合适的Anchor以及相应的前景/背景标签和目标回归系数，以训练RPN。该层的输出仅用于训练RPN网络，分类层不使用。给定一组Anchor（由Anchor Generation Layer生成，Anchor Target Layer标识有前景的前景和背景Anchor。有希望的前景Anchor是那些与某个Ground Truth重叠的阈值大于某个阈值的那些框。Ground Truth低于阈值Anchor定目标层还输出一组边界框回归量，即每个Anchor定目标距最近的边界框的距离的度量。
- **RPN Loss**： RPN损失函数是在优化过程中最小化的指标，以训练RPN网络。损失函数是以下各项的组合：
    - 由RPN产生的边界框被正确分类为前景/背景的比例
    - 预测回归系数和目标回归系数之间的一些距离度量。
- **Proposal Target Layer：** 目标是减少提案层产生的Anchor，并生成特定于类别的边界框回归目标，这些目标可用于训练分类层以生成良好的类别标签和回归目标
- **ROI Pooling Layer：** 根据提议目标层生成的ROI的边界框坐标对输入特征图进行采样。这些坐标通常不会位于整数边界上，因此需要基于插值的采样。
- **Classification Layer：** 获取由ROI合并层生成的输出特征图，并将它们传递给一系列卷积层。输出通过两个全连接层。第一层为每个ROI生成类别概率分布，第二层为一组特定类别的边界框回归变量。
- **Classification Loss:** 类似于RPN损失，分类损失是在优化过程中训练分类网络时最小化的指标。在反向传播期间，误差梯度也会流到RPN网络，因此训练分类层也会修改RPN网络的权重。关于这一点，我们将在以后再说。分类损失是以下各项的组合：
    - 由RPN产生并正确分类（作为正确的对象类）的Bouding Box的比例
    - 预测回归系数和目标回归系数之间的一些距离度量。

下面，我们将详细介绍所有这些层。

### Anchor Generation Layer

Anchor Generation Layer会生成一组大小和纵横比都分布在整个输入图像上的边界框（称为“Anchor”）。这些边界框对于所有图像都是相同的，即，它们与图像的内容无关。其中一些边界框将包含前景对象，而大多数边界框则只有背景。RPN网络的目标是学习识别其中哪些框是好框，即可能包含前景对象并产生目标回归系数，将其应用于锚框时会将锚框变成更好的边界框（使封闭的前景对象更紧密地适合）。

下图演示了如何生成这些锚框

![](https://img-blog.csdnimg.cn/20200211194352355.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

解释：Faster R-CNN中进行16倍下采样，输入图片size=(800,600)的情况下，得到的feature map size=(800/16,600/16), 相当于有1900个grid。要在每个grid上生成9个anchor,如左上角所示，所以总共生成了17100个bounding box。9个anchor是由3个形状，3个长宽比组成的。最后生成的Bounding Box可以完全覆盖整张图片。



### Region Proposal Layer

目标检测方法需要输入“区域建议系统”，该系统会生成一组稀疏特征（例如，Selective Search 、Dense等） 。R-CNN系统的第一个版本使用Selective Search方法来生成区域建议。在 Faster R-CNN中，基于划窗的技术（在上一节中进行了描述）用于生成一组密集的候选区域，然后使用神经网络驱动的RPN根据包含前景对象的区域的概率对区域建议进行排序。RPN有两个目标：

- 从Anchor列表中，确定背景Anchor和前景Anchor
- 通过应用一组“回归系数”来修改Anchor的位置，宽度和高度，以提高Anchor的质量（例如，使Anchor更好地适合对象的边界）

Region Proposal Layer由RPN和三层（提议层，锚定目标层和提议目标层）组成。以下各节将详细介绍这三层。

![](https://img-blog.csdnimg.cn/20200211194404153.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

ROI层通过卷积层（在代码中称为rpn_net）和RELU来运行由头网络生成的特征图。rpn_net的输出通过两（1,1）个内核卷积层来产生背景/前景类分数和概率以及相应的边界框回归系数。头部网络的步幅长度与生成Anchor时使用的步幅相匹配，因此Anchor框的数量与区域建议网络生成的信息成1-1对应（Anchor框的数量=类别分数的数量=得分的数量）边界框回归系数= ![\ frac {w} {16} \ times \ frac {h} {16} \ times9](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-cafeb41929226bc13d570966070b1313_l3.svg)）



### 提案层

提议层获取由锚生成层生成的锚框，并通过基于前景分数应用非最大抑制来修剪框的数量（有关详细信息，请参见附录）。它还通过将RPN生成的回归系数应用于相应的锚框来生成变换后的边界框。

![](https://img-blog.csdnimg.cn/20200211194415719.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 锚定目标层

Anchor目标层的目标是选择有前途的Anchor，这些Anchor可用于训练RPN网络以：

1. 区分前景和背景区域
2. 为前景框生成良好的边界框回归系数。

首先查看RPN损耗的计算方式很有用。这将揭示计算RPN损耗所需的信息，从而使跟踪锚定目标层的操作变得容易。

#### 计算RPN损失

请记住，RPN层的目标是生成良好的边界框。为此，RPN层必须学习将锚定框分类为背景还是前景，并计算回归系数以修改前景锚定框的位置，宽度和高度，使其成为“更好”的前景。框（更适合前景对象）。RPN损失的制定方式旨在鼓励网络学习这种行为。

RPN损失是分类损失和边界框回归损失的总和。分类损失使用交叉熵损失对不正确分类的框进行惩罚，回归损失使用真实回归系数（使用前景Anchor框的最接近匹配Ground Truth计算）与网络预测的回归系数之间的距离的函数（请参阅RPN网络体系结构图中的rpn_bbx_pred_net）。

![RPN损失= \ text {分类损失} + \ text {边界框回归损失}](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-142d5b70256748a64605bfc6e2f30ea9_l3.svg)

**分类损失：**

cross_entropy（预测的_class，实际的_class）

**边界框回归损失：**

![L_ {loc} = \ sum_ {u \ in {\ text {所有前景Anchor}}}} l_u](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-79e8cbe4b5682f6abc719c54d768a4ae_l3.svg)

对所有前景Anchor的回归损失求和。对背景Anchor执行此操作没有意义，因为背景Anchor没有相关的地面真相框

![l_u = \ sum_ {i \ in {x，y，w，h}} smooth_ {L1}（u_i（预测的）-u_i（目标）） ](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-f26b9d082be79d08e06cdbeb5cfc1e3a_l3.svg)

这显示了如何计算给定前景Anchor的回归损失。我们采用预测（通过RPN）和目标（使用与锚框最接近的Ground Truth计算）回归系数之间的差异。有四个部分–对应于左上角的坐标和边界框的宽度/高度。平滑的L1函数定义如下：

![smooth_ {L1}（x）= \ begin {cases} \ frac {\ sigma ^ 2x ^ 2} {2}和\ lVert x \ rVert <\ frac {1} {\ sigma ^ 2} \\ \ lVert x \ rVert-\ frac {0.5} {\ sigma ^ 2}及其他\ end {cases}](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-dae64c7ea8affa572e4b38b84688e1fd_l3.svg)

这里![\ sigma](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-1c9cc40f96a1492e298e7da85a2c1692_l3.svg)是任意选择的（在我的代码中设置为3）。请注意，在python实现中，用于前景Anchor的掩码数组（称为“ bbox_inside_weights”）用于将损失作为矢量运算来计算，并避免了for-if循环。

因此，要计算损失，我们需要计算以下数量：

1. 类别标签（背景或前景）和锚框得分
2. 前景Anchor框的目标回归系数

现在，我们将遵循锚定目标层的实现，以了解如何计算这些数量。我们首先选择图像范围内的Anchor框。然后，通过首先计算所有锚定框（在图像内）与所有Ground Truth的IoU（联合交集）重叠来选择好的前景框。使用此重叠信息，将两种类型的框标记为前景：

1. **类型A：**对于每个地面真相框，所有具有最大IoU的前景框都与地面真相框重叠
2. **类型B：**  与某些Ground Truth的最大重叠量超过阈值的Anchor框

这些框如下图所示：

![](https://img-blog.csdnimg.cn/20200211194427990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

注意，仅将与某些Ground Truth的重叠量超过阈值的Anchor框选择为前景框。这样做是为了避免向RPN提供“无希望学习任务”，即学习距离最佳匹配Ground Truth太远的框的回归系数。类似地，重叠小于负阈值的框被标记为背景框。并非所有不是前景框的框都标记为背景。既不是前景也不是背景的框都标记为“无关”。这些框不包括在RPN损失的计算中。



还有两个其他阈值与我们要实现的背景框和前景框的总数以及应作为前景的分数相关。如果通过测试的前台框数量超过了阈值，我们将多余的前台框随机标记为“无关”。类似的逻辑应用于背景框。

接下来，我们计算前景框与相应的地面真实框之间具有最大重叠的边界框回归系数。这很容易，只需遵循以下公式即可计算回归系数。

至此，我们对锚定目标层的讨论结束了。回顾一下，让我们列出该层的参数和输入/输出：

**参数：**

- TRAIN.RPN_POSITIVE_OVERLAP：用于选择Anchor框是否为好前景框的阈值（默认值：0.7）
- TRAIN.RPN_NEGATIVE_OVERLAP：如果来自地面真理框的Anchor的最大重叠量小于此阈值，则将其标记为背景。重叠大于RPN_NEGATIVE_OVERLAP但小于RPN_POSITIVE_OVERLAP的框被标记为“无关”。（默认值：0.3）
- TRAIN.RPN_BATCHSIZE：背景和前景Anchor的总数（默认值：256）
- TRAIN.RPN_FG_FRACTION：作为前景Anchor的批次大小的一部分（默认值：0.5）。如果找到的前景Anchor数量大于TRAIN.RPN_BATCHSIZE ![\ times](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-3e2a3b7b9d8913e71519bf7df9eb51b3_l3.svg) TRAIN.RPN_FG_FRACTION，则多余的（索引是随机选择的）标记为“无关”。

**输入：**

- RPN网络输出（预测的前景/背景类别标签，回归系数）
- 锚框（由锚生成层生成）
- 地面真相箱

**输出量**

- 好的前景/背景框和相关的类标签
- 目标回归系数

其他层（投标目标层，ROI合并层和分类层）用于生成计算分类损失所需的信息。就像我们对锚定目标层所做的一样，让我们首先来看一下如何计算分类损失以及计算所需的信息

### 计算分类层损失

与RPN损失类似，分类层损失有两个组成部分-分类损失和边界框回归损失

![\ text {分类层损失} = \ text {分类层损失} + \ text {边界框回归损失}](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-a01bdc80cc44f16d0971e77943f816b7_l3.svg)

RPN层和分类层之间的主要区别在于，虽然RPN层仅处理两个类-前景和背景，但是分类层处理我们的网络正在训练用于分类的所有对象类（加上背景）。

分类损失是将真实物体类别和预测类别得分作为参数的交叉熵损失。如下所示进行计算。



![](https://img-blog.csdnimg.cn/20200211194437311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

边界框回归损失的计算也类似于RPN，只是现在回归系数是特定于类别的。网络为每个对象类别计算回归系数。目标回归系数显然仅适用于正确的类别，该类别是与给定锚定框具有最大重叠的地面真相边界框的对象类别。在计算损耗时，将使用标记每个锚框正确对象类别的掩码数组。错误对象类别的回归系数将被忽略。该掩码阵列允许将损耗的计算作为矩阵乘法，而不是需要逐循环。

因此，需要以下数量来计算分类层损耗：

1. 预测的类别标签和边界框回归系数（这些是分类网络的输出）
2. 每个锚框的类标签
3. 目标边界框回归系数

现在让我们看看如何在投标目标和分类层中计算这些数量。

### 提案目标层

提案目标层的目标是从提案层输出的ROI列表中选择有前途的ROI。这些有前途的ROI将用于从头层生成的特征图执行作物合并，并传递到网络的其余部分（head_to_tail），该网络计算预测的类分数和盒回归系数。

类似于锚定目标层，重要的是要选择好的建议（那些与gt框明显重叠的建议）传递到分类层。否则，我们将要求分类层学习“无希望学习任务”。

提案目标层始于提案层计算出的ROI。使用每个ROI与所有基本真值框的最大重叠量，将ROI分为背景和前景ROI。前景ROI是最大重叠超过阈值的阈值（TRAIN.FG_THRESH，默认值：0.5）。背景ROI是最大重叠在TRAIN.BG_THRESH_LO和TRAIN.BG_THRESH_HI之间的ROI（分别为默认值0.1、0.5）。这是“硬否定挖掘”的一个示例，用于向分类器提供困难的背景示例。

还有一些其他逻辑试图确保前景和背景区域的总数恒定。如果发现背景区域太少，它将尝试通过随机重复一些背景索引来弥补该不足，以填充批次。

接下来，在每个ROI和最接近的匹配地面实况框之间计算边界框目标回归目标（这也包括背景ROI，因为对于这些ROI也存在重叠的地面实况框）。如下图所示，这些回归目标针对所有类别进行了扩展。

![](https://img-blog.csdnimg.cn/20200213100301656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

bbox_inside_weights数组用作掩码。仅对于每个前景ROI的正确类别为1。对于背景ROI也为零。因此，在计算分类层损失的边界框回归分量时，仅考虑前景区域的回归系数。对于分类损失不是这种情况–包括背景ROI，因为它们属于“背景”类别。

**输入：**

- 提案层产生的投资回报率
- 基本事实信息

**输出：**

- 满足重叠条件的选定前景和背景ROI。
- 针对ROI的类别特定目标回归系数

**参数：**

- TRAIN.FG_THRESH ：（默认值：0.5）用于选择前景ROI。与地面实况框的最大重叠超过FG_THRESH的ROI被标记为前景
- TRAIN.BG_THRESH_HI ：（默认值为0.5）
- TRAIN.BG_THRESH_LO ：（默认值为0.1）这两个阈值用于选择背景ROI。最大重叠在BG_THRESH_HI和BG_THRESH_LO之间的ROI被标记为背景
- TRAIN.BATCH_SIZE ：（默认值为128）已选择的前景框和背景框的最大数量。
- TRAIN.FG_FRACTION ：（默认值0.25）。前台框的数量不能超过BATCH_SIZE * FG_FRACTION



### 作物汇集

提案目标层为我们提供了有希望的投资回报率，以供我们进行分类以及训练期间使用的相关类别标签和回归系数。下一步是从头部网络生成的卷积特征图中提取与这些ROI对应的区域。然后，将提取的特征图遍历网络的其余部分（如上图所示，在网络图中的“尾部”），以生成每个ROI的对象类别概率分布和回归系数。作物合并层的工作是从卷积特征图中执行区域提取。

作物汇集背后的关键思想在“空间转化网络” [（Anon。2016）](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/#ITEM-1455-7)[ *中进行了描述](https://arxiv.org/pdf/1506.02025.pdf)。目的是将变形函数（由![2 \次3](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-40990cb7ba6e0228790c93ed0df1681e_l3.svg)仿射变换矩阵描述）应用于输入特征图，以输出变形特征图。如下图所示

![img](http://www.telesens.co/wp-content/uploads/2018/03/img_5aa402baba3a1-1.png)

作物合并涉及两个步骤：

1. 对于一组目标坐标，应用给定的仿射变换来生成源坐标的网格。
    ![\ begin {bmatrix} x_i ^ s \\ y_i ^ s \ end {bmatrix} = \ begin {bmatrix} \ theta_ {11}＆\ theta_ {12}＆\ theta_ {13} \\ \ theta_ {21}＆\ theta_ {22}和\ theta_ {23} \ end {bmatrix} \ begin {bmatrix} x_i ^ t \\ y_i ^ t \\ 1 \ end {bmatrix} ](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-f60f960a880c0385d4d51cae39215333_l3.svg)。这![x_i ^ s，y_i ^ s，x_i ^ t，y_i ^ t](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-d71ed008c2b7407162e68fb434a5ecf3_l3.svg)是高度/宽度归一化坐标（类似于图形中使用的纹理坐标），因此![-1 \ leq x_i ^ s，y_i ^ s，x_i ^ t，y_i ^ t \ leq 1](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-60ee2d9b96282537c4daa3d77c382017_l3.svg)。
2. 第二步，在源坐标处对输入（源）图进行采样以生成输出（目标）图。在此步骤中，每个![（x_i ^ s，y_i ^ s）](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-733fbd7b04e31ff38cd738e8f44d6a9d_l3.svg)坐标都定义了输入中的空间位置，在该位置上应用了采样核（例如双线性采样核）以获取输出特征图中特定像素处的值。

空间变换中描述的采样方法提供了一种可微分的采样机制，允许损耗梯度流回到输入要素图和采样网格坐标。

幸运的是，在PyTorch中实现了农作物合并，API包含两个与这两个步骤相似的函数。 torch.nn.functional.affine_grid采用仿射变换矩阵并生成一组采样坐标，然后torch.nn.functional.grid_sample  在这些坐标处对网格进行采样  。pyTorch自动处理后退步骤中的后向传播渐变。

要使用作物池，我们需要执行以下操作：

1. 用ROI坐标除以“头部”网络的步长。由提议目标层产生的ROI的坐标在原始图像空间中（！800600 ![\ times](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-3e2a3b7b9d8913e71519bf7df9eb51b3_l3.svg)）。为了将这些坐标带入由“ head”生成的输出特征图的空间中，我们必须将它们除以步幅长度（当前实现中为16）。
2. 要使用上面显示的API，我们需要仿射变换矩阵。仿射变换矩阵的计算如下
3. 我们还需要目标特征图上![X](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-ede05c264bba0eda080918aaa09c4658_l3.svg)和![ÿ](http://www.telesens.co/wp-content/ql-cache/quicklatex.com-0af556714940c351c933bba8cf840796_l3.svg)维度中的点数。这由配置参数cfg.POOLING_SIZE提供（默认值为7）。因此，在作物合并期间，非方形ROI用于从卷积特征图中裁剪出扭曲到恒定大小的方形窗口的区域。当作物合并的输出传递到需要固定尺寸输入的其他卷积层和完全连接层时，必须进行此变形。


![](https://img-blog.csdnimg.cn/20200213102612505.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 分类层

作物合并层采用提案目标层输出的ROI框和“头部”网络输出的卷积特征图，并输出正方形特征图。然后将特征图穿过ResNet的第4层，然后沿着空间维度进行平均池化。结果（在代码中称为“ fc7”）是每个ROI的一维特征向量。该过程如下所示。

![](https://img-blog.csdnimg.cn/2020021119445387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



然后，特征向量通过两个完全连接的层传递-bbox_pred_net和cls_score_net。cls_score_net层为每个边界框生成类分数（可以通过应用softmax将其转换为概率）。bbox_pred_net层生成特定于类的Bouding Box回归系数，该系数与提案目标层生成的原始Bouding Box坐标相结合，以生成最终的Bouding Box。这些步骤如下所示。



![](https://img-blog.csdnimg.cn/20200211194503520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)





最好回忆一下两组边界框回归系数之间的差异，其中一组由RPN网络生成，第二组由分类网络生成。第一组用于训练RPN层以产生良好的前景边界框（更紧密地围绕对象边界放置）。目标回归系数，即将ROI框与其最接近的匹配地面真相边界框对齐所需的系数，是由锚定目标层生成的。很难精确地确定这种学习是如何发生的，但是我想像RPN卷积和完全连接的层将学习如何将神经网络生成的各种图像特征解释为解密良好的对象边界框。当我们在下一部分中考虑推理时，

第二组边界框系数由分类层生成。这些系数是特定于类别的，即，为每个ROI框的每个对象类别生成一组系数。这些目标回归系数由提案目标层生成。注意，分类网络在正方形特征图上操作，该正方形特征图是应用于头部网络输出的仿射变换（如上所述）的结果。但是，由于回归系数对于没有剪切的仿射变换是不变的，因此可以将建议目标层计算的目标回归系数与分类网络生成的目标回归系数进行比较，并作为有效的学习信号。事后看来，这一点似乎很明显，但是我花了一些时间来理解。

有趣的是，在训练分类层时，误差梯度也会传播到RPN网络。这是因为在作物合并期间使用的ROI框坐标本身就是网络输出，因为它们是将RPN网络生成的回归系数应用于锚框的结果。在反向传播期间，误差梯度将通过作物合并层传播回RPN层。计算和应用这些梯度将非常棘手，但是值得庆幸的是，PyTorch提供了作物合并API作为内置模块，并且内部处理了计算和应用梯度的详细信息。在Faster RCNN论文的第3.2（iii）节[（Ren等人，2015年）](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/#ITEM-1455-2)[ *中](https://arxiv.org/abs/1506.01497)讨论了这一点   。



## 实现细节：推论

推断过程中执行的步骤如下所示

![](https://img-blog.csdnimg.cn/20200213102732471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

不使用定位目标和建议目标层。RPN网络应该已经学会了如何将锚框分为背景框和前景框，以及如何生成良好的边界框系数。建议层仅将边界框系数应用于排名最高的锚框，然后执行NMS来消除具有大量重叠的框。为了更清楚起见，这些步骤的输出如下所示。将结果框发送到分类层，在其中生成类分数和类特定的边界框回归系数。



![](https://img-blog.csdnimg.cn/20200213102819343.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

红色框显示按得分排名的前6位锚。应用RPN网络计算的回归参数后，绿色框显示锚框。绿色框似乎更紧密地适合基础对象。请注意，在应用回归参数之后，矩形仍为矩形，即没有剪切。还要注意矩形之间的明显重叠。通过应用非最大值抑制来解决此冗余问题

![](https://img-blog.csdnimg.cn/20200213102827650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

红色框显示NMS之前的前5个边界框，绿色框显示NMS之后的前5个边界框。通过抑制重叠的框，其他框（分数列表中的较低）有机会向上移动



![](https://img-blog.csdnimg.cn/20200213102833369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

从最终分类分数数组（dim：n，21）中，我们选择与某个前景对象（例如car）相对应的列。然后，我们选择与该数组中最大分数相对应的行。该行对应于最有可能是汽车的提案。令该行的索引为car_score_max_idx现在，令最终边界框坐标的数组（应用回归系数之后）为bboxes（dim：n，21 * 4）。从该数组中，选择与car_score_max_idx对应的行。我们希望与“汽车”列相对应的边界框应比其他边界框（对应于错误的对象类别）更好地适合汽车在测试图像中的位置。确实是这样。在红色框对应于原提案箱，蓝箱是汽车类别的计算出的边界框，白色框对应于其他（不正确）前景类别。可以看出，蓝色框比其他框更适合实际的汽车。

为了显示最终的分类结果，我们应用了另一轮NMS并将目标检测阈值应用于类分数。然后，我们绘制与满足检测阈值的ROI对应的所有变换后的边界框。结果如下所示。



## 附录

### ResNet 50网络架构

![](https://img-blog.csdnimg.cn/20200213102849487.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/20200211194514288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



### 非最大抑制（NMS）

非最大抑制是一种用于通过消除重叠量大于阈值的框来减少候选框数量的技术。首先按照某些条件对这些框进行排序（通常是右下角的y坐标）。然后，我们遍历框的列表，并抑制其IoU与正在考虑的框重叠的框超过阈值。通过y坐标对框进行排序会导致保留一组重叠框中的最低框。这可能并不总是理想的结果。R-CNN中使用的NMS按前景分数对框进行排序。这导致保留了一组重叠框中得分最高的框。下图显示了两种方法之间的区别。黑色数字是每个框的前景得分。右图显示了将NMS应用于左图的结果。第一个数字使用标准NMS（方框按右下角的y坐标排序）。这导致保留较低分数的框。第二个数字使用修改后的NMS（框按前景得分排序）。这导致保留了具有最高前景得分的框，这是更可取的。在这两种情况下，均假定框之间的重叠度高于NMS重叠阈值。这是更可取的。在这两种情况下，均假定框之间的重叠度高于NMS重叠阈值。这是更可取的。在这两种情况下，均假定框之间的重叠度高于NMS重叠阈值。



![](https://img-blog.csdnimg.cn/20200211194525703.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20200211194538562.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)







