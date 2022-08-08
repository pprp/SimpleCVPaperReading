# VOS: Learning What You Don't Know by Virtual Outlier Synthesis                                                 ICLR 2022

### 1. 论文信息

标题：VOS: Learning What You Don't Know by Virtual Outlier Synthesis

作者：Xuefeng Du, Zhaoning Wang, Mu Cai, Yixuan Li

原文链接：https://github.com/deeplearning-wisc/vos

代码链接：https://arxiv.org/abs/2202.01197

### 2. 引言

现代深度神经网络在已知的环境中获得了前所未有的成功，但它们常常难以处理未知的环境。特别是，神经网络已被证明对out- distribution (OOD)测试输入产生高后验概率，即会把一些OOD的类别定义为In-distribution的类别。以自动驾驶汽车为例，训练一个对象检测模型来识别分布内的对象(如汽车、停车标志)，可以对驼鹿的看不见的物体产生高置信度预测，预测为了行人。这样的故障情况会影响模型的可靠性。更糟糕的是，当部署在安全应用的关键程序程序中时，可能会导致灾难。

![image-20220807223432701](rg9005y61.hn-bkt.clouddn.com/image-20220807223432701.png)

![image-20220805154335297](https://cdn.jsdelivr.net/gh/Taly-1119/blogImage@main/img/image-20220805154335297.png)

无法感知出OOD的输入，很大成都上是因为在训练期间缺乏对ID数据边界的建模造成的。特别是，神经网络通常只对分布内(ID)数据进行优化，而缺少对OOD数据的感知。由此产生的决策边界，尽管在分类等ID任务中很有用，但对于OOD检测来说可能是很难正确完成的。如上图所示，ID数据(灰色)由三个类条件高斯组成，在此基础上训练一个三向softmax分类器。得到的分类器对于远离ID数据的区域产生over-confidence的现象(见图I(b)中的红色阴影部分)，影响了OOD的检测。理想情况下，模型应该学习一个更紧凑的决策边界，该边界为ID数据产生较低的不确定性，而在其他地方产生较高的OOD不确定性(例如，图[1(c))。然而，由于缺乏未知的监督信号，实现这一目标并非易事。

所以论文提出了一个问题：

> Can we synthesize virtual outliers for effective model regularization？

是否可以生成一些outlier来辅助规范模型的边界，从而让模型可以很好地确定ID类别和OOD类别的边界？

本文提出了一种新的未知感知学习框架VOS (Virtual Outlier Synthesis)，该框架优化了ID任务和OOD检测性能的双重目标。简而言之，VOS由三部分组成，用于解决异常点综合和有效的模型正则化问题。为了综合异常值，我们估计了特征空间中的类条件分布，并从ID类的低似然区域中采样异常值。论文阐述了在特征空间中采样比在高维像素空间中合成图像更容易处理。

### 3. 方法框架

![image-20220807223509212](rg9005y61.hn-bkt.clouddn.com/image-20220807223509212.png)

![image-20220805154929065](https://cdn.jsdelivr.net/gh/Taly-1119/blogImage@main/img/image-20220805154929065.png)

提出的新颖的未知感知学习框架如上图所示。 论文提出的框架包括 三种新组分，针对以下问题:

1. 如何合成虚拟的outliers
2. 如何利用合成的outliers进行有效的模型正则化  
3. 如何在推断阶段执行OOD检测

#### 3.1 如何合成outliers

首先，论文把每个ID类别建模为混合多元高斯分布，通过UAMP降维也可以观察到确实和高斯分布比较类似：
$$
p_{\theta}(h(\mathbf{x}, \mathbf{b}) \mid y=k)=\mathcal{N}\left(\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}\right),
$$
where $\boldsymbol{\mu}_{k}$ is the Gaussian mean of class $k \in\{1,2, . ., \mathrm{K}\}, \boldsymbol{\Sigma}$ is the tied covariance matrix, and $h(\mathbf{x}, \mathbf{b}) \in \mathbb{R}^{m}$ is the latent representation of an object instance $(\mathbf{x}, \mathbf{b})$. To extract the latent representation, we use the penultimate layer of the neural network. The dimensionality $m$ is significantly smaller than the input dimension $d$.

![image-20220807223649172](rg9005y61.hn-bkt.clouddn.com/image-20220807223649172.png)

然后，再依靠各个ID类别在特征空间的分布，估计出各个类别的均值和协方差：
$$
\begin{aligned}
\widehat{\boldsymbol{\mu}}_{k} &=\frac{1}{N_{k}} \sum_{i: y_{i}=k} h\left(\mathbf{x}_{i}, \mathbf{b}_{i}\right) \\
\widehat{\boldsymbol{\Sigma}} &=\frac{1}{N} \sum_{k} \sum_{i: y_{i}=k}\left(h\left(\mathbf{x}_{i}, \mathbf{b}_{i}\right)-\widehat{\boldsymbol{\mu}}_{k}\right)\left(h\left(\mathbf{x}_{i}, \mathbf{b}_{i}\right)-\widehat{\mu}_{k}\right)^{\top},
\end{aligned}
$$
然后根据这些混合高斯分布的参数在boundary附近进行采样：
$$
\mathcal{V}_{k}=\left\{\mathbf{v}_{k} \mid \frac{1}{(2 \pi)^{m / 2}|\widehat{\boldsymbol{\Sigma}}|^{1 / 2}} \exp \left(-\frac{1}{2}\left(\mathbf{v}_{k}-\widehat{\boldsymbol{\mu}}_{k}\right)^{\top} \widehat{\boldsymbol{\Sigma}}^{-1}\left(\mathbf{v}_{k}-\widehat{\boldsymbol{\mu}}_{k}\right)\right)<\epsilon\right\}
$$
where $\mathbf{v}_{k} \sim \mathcal{N}\left(\widehat{\boldsymbol{\mu}}_{k}, \widehat{\boldsymbol{\Sigma}}\right)$ denotes the sampled virtual outliers for class $k$, which are in the sublevel set based on the likelihood. $\epsilon$ is sufficiently small so that the sampled outliers are near class boundary.

设定一个阈值，若与boundary的距离小于预先设定的一个阈值则定义该类别为合成的outliers。（根据代码可以发现采样阈值的实现是通过取的数量来决定的，可以参考代码）
$$
f(\mathbf{v} ; \theta)=W_{\mathrm{cls}}^{\top} \mathbf{v}
$$
where $W_{\text {cls }} \in \mathbb{R}^{m \times K}$ is the weight of the last fully connected layer. We proceed with describing how to regularize the output of virtual outliers for improved OOD detection.

然后就是根据一个全连接层计算出输出的logit。

![image-20220807223729739](rg9005y61.hn-bkt.clouddn.com/image-20220807223729739.png)

![image-20220805160703904](https://cdn.jsdelivr.net/gh/Taly-1119/blogImage@main/img/image-20220805160703904.png)

#### 3.2 利用outlier训练的方式

在OOD检测中，经常采用energy score来作为判断为ID或者OOD：
$$
p(y \mid \mathbf{x})=\frac{p(\mathbf{x}, y)}{p(\mathbf{x})}=\frac{e^{f_{y}(\mathbf{x} ; \theta)}}{\sum_{k=1}^{K} e^{f_{k}(\mathbf{x} ; \theta)}}
$$
由于传统的energy score把所有类别都采用相同的权重，少了很多灵活度，所以加入非线性层来增加网络的灵活度。再采用BCELoss来作为二分类的损失函数来训练。
$$
\mathcal{L}_{\text {uncertainty }}=\mathbb{E}_{\mathbf{v} \sim \mathcal{V}}\left[-\log \frac{1}{1+\exp ^{-\phi(E(\mathbf{v} ; \theta))}}\right]+\mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\left[-\log \frac{\exp ^{-\phi(E(\mathbf{x} ; \theta))}}{1+\exp ^{-\phi(E(\mathbf{x} ; \theta))}}\right]
$$
但同时由于目标检测的数据集非常的不平衡，因此引入一个权重值来对不同类别进行加权：
$$
E(\mathbf{x}, \mathbf{b} ; \theta)=-\log \sum_{k=1}^{K} w_{k} \cdot \exp ^{f_{k}((\mathbf{x}, \mathbf{b}) ; \theta)}
$$
训练方式就非常直接了，就是利用energy的方式来进行二分类的训练。

#### 3.3 推理阶段识别OOD

![image-20220805163231986](https://cdn.jsdelivr.net/gh/Taly-1119/blogImage@main/img/image-20220805163231986.png)

测试也就非常直接，利用uncertainty来提前设定阈值来判断是OOD还是ID。

### 4. 实验结果

![](rg9005y61.hn-bkt.clouddn.com/image-20220807223741384.png)

跟其他的OOD检测的方法比，自然是性能优越不少，同时可以看到，采用相同的网络，对于ID数据的识别也是有提升的，mAP也比较高。

![image-20220805163513052](https://cdn.jsdelivr.net/gh/Taly-1119/blogImage@main/img/image-20220805163513052.png)

相较于其他的生成outlier的方式，也具有一定的优势。

![](rg9005y61.hn-bkt.clouddn.com/image-20220807223807224.png)

虽然文章主要实验在使用目标检测进行判断，但在CIFAR-10这一分类数据集上，也有明显的效果。

![image-20220807223918037](rg9005y61.hn-bkt.clouddn.com/image-20220807223918037.png)

通过对energy权重的分析，也可以发现权重和物体的数量呈现非常明显的正相关。

![](rg9005y61.hn-bkt.clouddn.com/image-20220807223937730.png)

发现采用这种outlier的合成方式，通过降维可视化也的确符合预期，是在ID数据的boundary上进行采样。

### 5. 结论

在本文中，提出了vos，一个新的未知感知的OOD检测训练框架。与需要真实离群数据的方法不同，vos在训练过程中从类条件分布的低似然区域采样虚拟离群数据，自适应地合成离群数据。该方法有效地改善了ID数据与OOD数据之间的决策边界，在保持ID任务性能的同时，提高了OOD的检测性能。VOS是有效的，适用于目标检测和分类任务。希望该工作能够启发未来在现实环境中对未知感知深度学习的研究。