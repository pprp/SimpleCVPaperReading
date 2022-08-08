# 动态稠密SLAM的自监督场景运动分解

## 0. 引言

场景运动估计的任务是获取动态场景的三维结构和三维运动，在论文"DeFlowSLAM: Self-Supervised Scene Motion Decomposition for Dynamic Dense SLAM"，作者提出了一种基于双流的运动估计算法，并且不需要对对象检测进行显式监督，更好地模拟了人类看待世界的方式。此外，该方法可以支持单目、双目和RGB-D等数据格式，算法即将开源。

## 1. 论文信息

标题：DeFlowSLAM: Self-Supervised Scene Motion Decomposition for Dynamic Dense SLAM

作者：Weicai Ye, Xingyuan Yu, Xinyue Lan, Yuhang Ming, Jinyu Li, Hujun Bao, Zhaopeng Cui, Guofeng Zhang

原文链接：<https://arxiv.org/abs/2207.08794>

代码链接：https://zju3dv.github.io/deflowslam

## 2. 摘要

我们提出了一种新的场景运动的双流表示法，将光流分解为由相机运动引起的静态流场和由场景中物体运动引起的动态流场。基于这种表示，我们提出了一种动态SLAM，称为DeFlowSLAM，它利用图像中的静态和动态像素来求解相机姿态，而不是像其他动态SLAM系统那样简单地使用静态背景像素。我们提出了一个动态更新模块，以自监督的方式训练我们的DeFlowSLAM，其中稠密BA层接受估计的静态流场和由动态掩模控制的权重，并输出优化的静态流场、相机姿态和反演深度的残差。通过将当前图像扭曲到相邻图像来估计静态和动态流场，并且通过将两个场相加可以获得光流。大量实验表明，DeFlowSLAM对静态和动态场景都有很好的通用性，因为它在静态和动态场景中表现出与最先进的DROID-SLAM相当的性能，而在高度动态环境中明显优于DROID-SLAM。

## 3. 算法分析

如图1所示是作者提出的基于双流的SLAM方法DeFlowSLAM的概述，该系统将一系列图像作为输入，提取特征构建相关体，并将其与初始静态流、光流、动态掩膜一起馈入动态更新模块，迭代优化姿态残差、逆深度、静态光流和动态光流，最终输出相机姿态和3D结构的估计。优化过程是通过创建一个共视图并更新现有的共视图来执行的。

![](https://img-blog.csdnimg.cn/f18f4e10a0154152bf0b4664a1fa80e4.png)

图1 DeFlowSLAM概述

DeFlowSLAM具有端到端的差异化架构，结合了经典方法和深层网络的优势。由于双流表示、迭代动态更新模块和基于帧间共视的因子图优化，它可以鲁棒地处理动态场景等具有挑战性的场景。

就像DROID-SLAM优化姿势和深度一样，静态场很容易优化，而对于动态场，可以通过将当前帧变换到相邻帧来获得一致的亮度。基于这种思路，作者为DeFlowSLAM提出了一个动态更新模块，其中巧妙地嵌入了双流表示，在动态更新模块中，作者引入了一个名为Mask-Agg的动态掩码聚集算子来消除不正确的对齐效应。具体来说，Mask-Agg算子通过卷积门控递归单元(ConvGRU)迭代更新动态掩码的残差。通过将聚集的动态掩码残差与原始掩码相加，可以获得最终的动态掩码。所获得的动态掩模将与所估计的权重相结合，并被馈送到稠密束调整(DBA)层，以优化姿态和深度的残差。

综上所述，作者所做工作的主要贡献如下：

\(1\) 提出了一种新颖的双流场景运动表示方法，将光流分解为静态流场和动态流场。

\(2\) 构建了一个动态稠密SLAM，即DeFlow-SLAM，它在动态场景中的性能优于最先进的方法。

\(3\) 提出了一种自我监督的训练方法来代替DROID-SLAM中的强监督。

### 3.1 双流表示法

作者所提出的动态SLAM网络的核心概念是双流表示和自监督训练方案。DROID-SLAM使用光流作为中间运动表示，而DeFlowSLAM通过将光流分解为由相机运动引起的静态流和由动态对象本身的运动引起的动态流，因此得出了一种新颖的场景运动表示，其原理如图2所示。这种表示法可以区分静态和动态的物体运动，因此具有更好的可解释性，并使网络在培训过程中可追溯。

<img src="https://img-blog.csdnimg.cn/19153fa8db254ffc8c496ebc26218121.png" style="zoom: 67%;" />

图2 双流表示法

与DROID-SLAM类似，DeFlowSLAM也使用了共视图。在静态流场更新每次姿态或深度后，可以使用新的共视关系进行更新。当相机返回到以前访问过的地方时，将向图形中添加一个远程连接，并进行回环检测。

### 3.2 动态更新模块

图3所示是DeFlowSLAM的动态更新模块结构，该模块是3×3的ConvGRU，并带有一个隐藏状态。与DROID-SLAM中直接处理修正光流的更新模块不同，DeFlowSLAM的动态更新模块分别处理分解后的静态流场和动态流场。首先以类似于DROID-SLAM的方式更新修正的静态流场，而对于动态流场，DeFlowSLAM会将其添加到静态流场中以获得光流，并在下一次迭代中作为一个新的优化项输入流编码器。每个应用程序都会更新隐藏状态，并另外生成姿态增量、深度增量、动态掩模增量和动态流。姿态增量通过在SE3流形上的缩回应用于当前姿态，而深度和动态掩模增量分别添加到当前深度和动态掩模中。

![](https://img-blog.csdnimg.cn/34bfc3a4c91247dd892b55b6edc21a07.png)

图3 动态更新模块

### 3.3 损失函数

DeFlowSLAM的损失函数有三个组成部分，分别是几何光度损失、光流光度损失以及人工掩膜损失。其中几何光度损失为SLAM中的重投影误差：

<img src="https://img-blog.csdnimg.cn/2019943edabb427bb65718965a9d08a5.png" style="zoom:67%;" />

其中$M_{\text{d}_{\text{i}}}^{\text{A}\text{gg}}$表示几何掩模聚合算子，用以滤掉错误的像素匹配，避免由于物体的运动而导致像素不匹配使精度降低的问题，$M_{\text{d}_{\text{i}}}^{\text{A}\text{gg}}$原理如图4所示。表1展示了Mask-Agg模块有助于过滤掉自监督训练中的模糊匹配，从而获得更好的结果。

![](https://img-blog.csdnimg.cn/7c668580fe1942d98ef27788e0052eb2.png)

图4 Mask-Agg举例

光流光度损失用来监控整个场景的运动，包括相机的运动和物体的运动。通过更新模块，可以通过添加静态流和动态流来获得光流结果。然后使用双线性采样从源图像中进行采样，评估它们的光度误差：

<img src="https://img-blog.csdnimg.cn/dc7d0705c42143b2991881f547f43fa9.png" style="zoom:67%;" />

人工掩模损失可以在动态掩模标签不可用时发挥效果：

<img src="https://img-blog.csdnimg.cn/d2d327cd909c41bc98cef635ba37ffc3.png" style="zoom:70%;" />

最终损失为三者的叠加：

<img src="https://img-blog.csdnimg.cn/1f8f01769aac43729e7d57b11412d45e.png" style="zoom:80%;" />

## 4. 实验

作者首先在VKITTI2的高度动态场景中验证方法的有效性，并进行消融实验。此外，在数据集TartanAir上使用相同的策略从头开始训练DeFlowSLAM，并在VKITTI2等不同的动态数据集上测试该方法的泛化能力。还测试了单目或双目数据集，如TUM-RGBD的静态场景和EuRoc数据集。评估指标为绝对轨迹误差(ATE)。

### 4.1 消融实验

作者使用VKITTI2数据集用于消融实验，图5表明，与DROID-SLAM相比，DeFlowSLAM使用动态像素来解决相机姿态。表1证明了双流表示方法优于粗糙的单流表示方法，并且Mask-Agg有助于改进姿态估计，其中SS表示自监督，SM表示半监督，SF表示单流，DF表示双流。此外，表1证明了自监督方法可以达到与监督方法相当或更好的准确性。

<img src="https://img-blog.csdnimg.cn/de5a669565704c61be49fb271adcb5a7.png" style="zoom: 50%;" />

图5 DROID-SLAM与DeFlowSLAM定性对比

表1 在VKITTI2数据集上训练和测试的DeFlowSLAM消融研究

<img src="https://img-blog.csdnimg.cn/b6147a222bba417f9a2dc9957d5c8c31.png" style="zoom: 70%;" />

表2展示了动态阈值的消融实验结果，结果显示阈值设置为0.5时性能达到最优。

表2 在VKITTI2上训练和测试的DeFlow-SLAM的动态阈值消融研究

<img src="https://img-blog.csdnimg.cn/bf1bcbb3cb55423583a629e33f650d8e.png" style="zoom:67%;" />

### 4.2 通用化

作者在TartanAir数据集上训练了DeFlowSLAM并在其他主流SLAM数据集上测试，如VKITTI2、具有动态车辆的KITTI、具有强烈运动和显著光照变化的EuRoc，带有运动模糊和剧烈旋转的TUM RGB-D。图6所示是DeFlowSLAM的定性结果，结果表明DeFlowSLAM很好地推广到不同的数据集。

<img src="https://img-blog.csdnimg.cn/1686f514c3c842ba81048563a4aa0b22.png" style="zoom: 80%;" />

图6 DeFlowSLAM的定性结果

### 4.3 动态SLAM

作者在KITTI数据集的序列09和10上测试了DeFlowSLAM的性能，ATE结果如表3所示。与使用Mask-RCNN进行动态环境感知的DynaSLAM和DROID-SLAM相比，DeFlowSLAM在动态场景中更加准确和鲁棒。表4展示了不同动态比例的TUM RGB-D动态序列评估结果，结果表明DeFlowSLAM具有竞争力，甚至是最好的性能。注意结果中其他算法使用RGB-D数据，而DeFlowSLAM和DROID-SLAM只使用单目数据。

表3 KITTI (K)和VKITTI2 (VK)数据集上的动态SLAM结果

<img src="https://img-blog.csdnimg.cn/fd7e4a0d6b3c48e7aea6295704ada766.png" style="zoom:80%;" />

表4 动态SLAM在TUM动态序列上的结果

<img src="https://img-blog.csdnimg.cn/d5637bc7287b486dad287fd017e88423.png" style="zoom:80%;" />

在单目实验中，作者在TartanAir测试集、EuRoC和TUMRGB-D数据集上测试训练过的DeFlowSLAM，结果如表5所示，DeFlowSLAM取得了最好的效果。表6和表7表明，DeFlowSLAM在大多数序列上取得了比SOTA监督方法更好的结果。这也证明了DeFlowSLAM比经典的SLAM算法更鲁棒，具体来说DeFlowSLAM在EuRoC数据集上实现的平均ATE为0.136 m，在TUM-RGBD静态序列上实现的平均ATE为0.114m，优于大多数监督方法。

表5 单目SLAM在TartanAir单目基准测试中的结果

<img src="https://img-blog.csdnimg.cn/f09a91bb04f64172a39d9c18d5203d79.png" style="zoom:67%;" />

表6 单目SLAM在EuRoc数据集上的结果

<img src="https://img-blog.csdnimg.cn/a8ff84e4c8c6460a8894185fc1828201.png" style="zoom: 67%;" />

表7 单目SLAM在TUM-RGBD数据集上的结果

<img src="https://img-blog.csdnimg.cn/185049c47d664d3b9e703ad428c9cd2a.png" style="zoom: 67%;" />

在双目数据下，DeFlowSLAM也在TartanAir测试数据集和EuRoc双目数据集上进行了测试。表8说明了DeFlowSLAM取得了与DROID-SLAM相当的结果，平均ATE为1.02m，优于TartanVO。表9显示DeFlowSLAM优于大多数监督方法和传统的SLAM算法。

表8 TartanAir双目基准测试结果

<img src="https://img-blog.csdnimg.cn/dea52c5813e7403d8fd62bf3e6ebdfd0.png" style="zoom:67%;" />

表9 EUROC数据集上的双目SLAM结果

<img src="https://img-blog.csdnimg.cn/934740e95a09450394638e983ecb809a.png" style="zoom: 67%;" />

如图7所示，作者还对AR应用进行了实验，以证明DeFlowSLAM的鲁棒性。其中，作者用虚拟树、汽车和路灯来增强原始视频。结果显示，DeFlowSLAM可以很好地处理场景中的动态对象，而DROID-SLAM则显示出显著的漂移。

<img src="https://img-blog.csdnimg.cn/146882e797d14c369792e8588e16a1a5.png" style="zoom: 50%;" />

图7 AR应用

### 4.4 运动分割

最后，作者提出虽然DeFlowSLAM主要应用于SLAM系统，但双流表示可以很好地应用于运动分割。如图8和表10所示，只需为运动设置一个阈值，将大于这个阈值的动态场的像素点可视化，就可以得到运动分割的结果，

<img src="https://img-blog.csdnimg.cn/53967edc21d54dc4bcea8177eaaa4f72.png" style="zoom: 50%;" />

图8 运动分割可视化结果

表10 运动分割定量结果

<img src="https://img-blog.csdnimg.cn/a03833548d4d42d3b328442e755bc779.png" style="zoom: 80%;" />

## 5. 结论

在论文"DeFlowSLAM: Self-Supervised Scene Motion Decomposition for Dynamic Dense SLAM"中，作者提出了一种新颖的双流表示法，将光流分解为由相机姿态引起的静态流场和由动态物体运动引起的动态流场，该系统在高动态场景中的性能优于DROID-SLAM，在静态和轻微动态场景中的性能相当。此外，作者还提出了几个DeFlowSLAM算法的不足，感兴趣的读者可以尝试进行深入研究：

\(1\) 在某些场景中DeFlowSLAM的表现比DROID-SLAM稍弱，可能是因为使用了某个固定的动态阈值。因此可以探索动态阈值估计方法来应对不同场景的挑战。

\(2\) 与DROID-SLAM一样，DeFlowSLAM对更长的序列和更大的场景有很高的内存要求，轻量高效的SLAM系统是一个潜在的研究方向。

\(3\) DeFlowSLAM更侧重于求解相机姿态，获得的深度和光流只有原始图像大小的1/8，对于深度估计和光流估计这样的任务并不理想。
