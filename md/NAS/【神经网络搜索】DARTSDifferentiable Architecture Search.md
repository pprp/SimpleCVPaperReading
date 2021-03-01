# 【神经网络搜索】DARTS: Differentiable Architecture Search

【GiantPandaCV】DARTS将离散的搜索空间松弛，从而可以用梯度的方式进行优化，从而求解神经网络搜索问题。本文首发于GiantPandaCV，未经允许，不得转载。

![https://arxiv.org/pdf/1806.09055v2.pdf](https://img-blog.csdnimg.cn/20210226222235337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

[TOC]

## 1. 简介

此论文之前的NAS大部分都是使用强化学习或者进化算法等在离散的搜索空间中找到最优的网络结构。而DARTS的出现，开辟了一个新的分支，将离散的搜索空间进行松弛，得到连续的搜索空间，进而可以使用梯度优化的方处理神经网络搜索问题。DARTS将NAS建模为一个两级优化问题（Bi-Level Optimization），通过使用Gradient Decent的方法进行交替优化，从而可以求解出最优的网络架构。DARTS也属于One-Shot NAS的方法，也就是先构建一个超网，然后从超网中得到最优子网络的方法。

## 2. 贡献

DARTS文章一共有三个贡献：

- 基于二级最优化方法提出了一个全新的可微分的神经网络搜索方法。
- 在CIFAR-10和PTB（NLP数据集）上都达到了非常好的结果。
- 和之前的不可微分方式的网络搜索相比，效率大幅度提升，可以在单个GPU上训练出一个满意的模型。

笔者这里补一张对比图，来自之前笔者翻译的一篇综述：<NAS的挑战和解决方案-一份全面的综述>

![ImageNet上各种方法对比，DARTS属于Gradient Optimization方法](https://img-blog.csdnimg.cn/20201114132717357.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70#pic_center)

简单一对比，DARTS开创的Gradient Optimization方法使用的GPU Days就可以看出结果非常惊人，与基于强化学习、进化算法等相比，DARTS不愧是年轻人的第一个NAS模型。

## 3. 方法

DARTS采用的是Cell-Based网络架构搜索方法，也分为Normal Cell和Reduction Cell两种，分别搜索完成以后会通过拼接的方式形成完整网络。在DARTS中假设每个Cell都有两个输入，一个输出。对于Convolution Cell来说，输入的节点是前两层的输出；对于Recurrent Cell来说，输入为当前步和上一步的隐藏状态。

DARTS核心方法可以用下面这四个图来讲解。

![DARTS Overview](https://img-blog.csdnimg.cn/20210228164207483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

**(a) 图**是一个有向无环图，并且每个后边的节点都会与前边的节点相连，比如节点3一定会和节点0，1，2都相连。这里的节点可以理解为特征图；边代表采用的操作，比如卷积、池化等。

引入数学标记：

**记**  节点(特征图)为： $x^{(i)}$ 代表第i个节点对应的潜在特征表示（特征图）。

**记**  边(操作)为:  $o^{(i,j)}$ 代表从第i个节点到第j个节点采用的操作。

**记**  每个节点的输入输出如下面公式表示，每个节点都会和之前的节点相连接，然后将结果通过求和的方式得到第j个节点的特征图。

$$
x^{(j)}=\sum_{i<j} o^{(i, j)}\left(x^{(i)}\right)
$$

**记**  所有的候选操作为 $\mathcal{O}$, 在DARTS中包括了3x3深度可分离卷积、5x5深度可分离卷积、3x3空洞卷积、5x5空洞卷积、3x3最大化池化、3x3平均池化，恒等，直连，共8个操作。

**(b) 图**是一个超网，将每个边都扩展了8个操作，通过这种方式可以将离散的搜索空间松弛化。具体的操作根据如下公式：
$$
\bar{o}^{(i, j)}(x)=\sum_{o \in \mathcal{O}} \frac{\exp \left(\alpha_{o}^{(i, j)}\right)}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left(\alpha_{o^{\prime}}^{(i, j)}\right)} o(x)
$$

这个可以分为两个部分理解，一个是$o(x)$代表操作，一个代表选择概率 $\frac{\exp \left(\alpha_{o}^{(i, j)}\right)}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left(\alpha_{o^{\prime}}^{(i, j)}\right)}$，这是一个softmax构成的概率，其中$\alpha_o^{(i,j)}$表示 **第i个节点到第j个节点之间操作的权重**，这也是之后需要搜索的网络结构参数，会影响该操作的概率。即以下公式：
$$
softmax(\alpha)\times operation_{w}(x)
$$
左侧代表当前操作的概率，右侧代表当前操作的参数。

**(c)和(d)图** 是保留的边，训练完成以后，从所有的边中找到概率最大的边，即以下公式：
$$
o^{(i, j)}=\operatorname{argmax}_{o \in \mathcal{O}} \alpha_{o}^{(i, j)}
$$

## 4. 数学推导

DARTS将NAS问题看作二级最优化问题，具体定义如下：

$$
\begin{aligned} \min _{\alpha} & \mathcal{L}_{v a l}\left(w^{*}(\alpha), \alpha\right) \\ \text { s.t. } & w^{*}(\alpha)=\operatorname{argmin}_{w} \mathcal{L}_{\text {train }}(w, \alpha) \end{aligned}
$$

$w*(\alpha)$ 代表当前网络结构参数$\alpha$的情况下，训练获得的最优的网络结构参数。

第一行代表：在验证数据集中，在特定网络操作参数w下，通过训练获得最优的网络结构参数$\alpha$。

第二行表示：在训练数据集中，在特定网络结构参数$\alpha$下，通过训练获得最优的网络操作参数$w$。

> 条件：在结构确定的情况下，获得最优的网络操作权重
>
> ​           ----- 结构确定，训练好卷积核
>
> 目标：在网络操作权重确定的情况下，获得最优的结构
>
> ​           ----- 卷积核不动，选择更好的结构

最简单的方法是通过交替优化参数$w$和参数$\alpha$, 来得到最优的结果，伪代码如下：

![DARTS伪代码](https://img-blog.csdnimg.cn/20210301092133238.png)

交替优化的复杂度非常高，是$O(|\alpha||w|)$, 这种复杂度不可能投入使用，所以要将复杂度进行优化，用复杂度低的公式近似目标函数。
$$
\nabla_{\alpha} \mathcal{L}_{\text {val }}\left(w^{*}(\alpha), \alpha\right) \approx \nabla_{\alpha} \mathcal{L}_{v a l}\left(w-\xi \nabla_{w} \mathcal{L}_{t r a i n}(w, \alpha), \alpha\right)
$$
这种近似方法在Meta Learning中经常用到，详见《Model-agnostic meta-learning for fast adaptation of deep networks》，也就是通过使用单个step的训练调整w，让这个结果来近似$w*(\alpha)$。

然后对右侧公式进行推导，得到梯度优化以后的表达式：

![师兄提供](https://img-blog.csdnimg.cn/20210301101236181.png)

---

这里求梯度使用的是链式法则，回顾一下：
$$
z=f(g1(x),g2(x))
$$

则梯度计算为：
$$
\frac{\partial z}{\partial x}=\frac{\partial g1}{\partial x} \times \frac{\partial z}{\partial g1} + \frac{\partial g2}{\partial x}\times\frac{\partial z}{\partial g2}
$$

或者

![师兄提供](https://img-blog.csdnimg.cn/20210301101617433.png)

上述公式中Di代表对$f(g1(\alpha),g2(\alpha))$的第i项的偏导。

---

![手敲公式太痛苦了](https://img-blog.csdnimg.cn/20210301105453255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

整理以后结果就是：

![计算结果](https://img-blog.csdnimg.cn/20210301105536805.png)

减号后边的是二次梯度，权重的梯度求解很麻烦，这里使用泰勒公式将二阶转为一阶（h是一个很小的值）。

![泰勒公式复习](https://img-blog.csdnimg.cn/20210301110026762.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

利用最右下角的公式：

令$A=\nabla_{\omega^{\prime}} \mathcal{L}_{v a l}\left(\omega^{\prime}, \alpha\right)$,$h=\epsilon$, $x_0=w$, $f=\nabla_{\alpha} \mathcal{L}_{\text {train }}(\cdot, \cdot)$, 代入可得
$$
\nabla_{\alpha, \omega}^{2} \mathcal{L}_{\text {train }}(\omega, \alpha) \cdot \nabla_{\omega^{\prime}} \mathcal{L}_{\text {val }}\left(\omega^{\prime}, \alpha\right) \approx \frac{\nabla_{\alpha} \mathcal{L}_{\text {train }}\left(\omega^{+}, \alpha\right)-\nabla_{\alpha} \mathcal{L}_{\text {train }}\left(\omega^{-}, \alpha\right)}{2 \epsilon}
$$

其中

$$
\omega^{\pm}=\omega \pm \epsilon \nabla_{\omega^{\prime}} \mathcal{L}_{v a l}\left(\omega^{\prime}, \alpha\right)
$$

这样就可以将二次梯度转化为多个一次梯度。到这里复杂度从$O(|\alpha||w|)$优化到$O(|\alpha|+|w|)$












## 致谢

感谢师兄提供的资料，以及知乎上两位大佬，他们文章链接如下：

薰风读论文|DARTS—年轻人的第一个NAS模型 https://zhuanlan.zhihu.com/p/156832334

【论文笔记】DARTS公式推导 https://zhuanlan.zhihu.com/p/73037439











