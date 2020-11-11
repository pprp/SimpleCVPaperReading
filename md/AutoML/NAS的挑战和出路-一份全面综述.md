# NAS的挑战和解决方案-一份全面的综述

【GiantPandaCV导读】笔者在这篇文章中对《A Comprehensive Survey of Nerual Architecture Search: Challenges and Solutions》这篇进行解读总结，这是2020年刚刚发到arxiv上的有关NAS的综述，内容比较多，30页152篇参考文献。

[TOC]



## 1. 背景

深度学习在很多领域都取得了巨大的突破和进展。这是由于深度学习具有的强大的自动化特征提取的能力。而网络结构的设计对数据的特征的表征和最终模型的表现起到了至关重要的作用。

为了获取数据的更好的特征表示，研究人员设计了多种多样的复杂的网络架构，而网络结构的设计是严重依赖于研究人员的先验知识和经验。同时网络结构的设计也很难跳出原有思考模式并设计出一个最优的网络。萌新很难根据自己的实际任务和需求对网络结构进行合理的修改。

一个很自然的想法就是尽可能减少人工的干预，让算法能够自动寻找最合适的网络架构，这就是网络搜索NAS提出的背景。

近些年很多有关NAS的优秀的工作层出不穷，分别从不同的角度来提升NAS算法。为了让初学者更好的进行NAS相关的研究，一个全面而系统的有关NAS的综述很重要。

**文章组织方式**：先前的关于NAS的综述是从NAS的不同组件的角度进行分析的，但是对初学者不太友好。本篇文章将按照以下方式组织：

- 早期NAS算法的特点。
- 总结早期NAS算法中存在的问题。

- 给出随后的NAS算法对以上问题提出的解决方案。
- 对以上算法继续分析、对比、总结。
- 最后，给出NAS未来可能的发展方向。

## 2. NAS介绍

NAS-神经网络架构搜索，其作用就是取代人工的网络架构设计，在这个过程中尽可能减少人工的干预。

人工设计的网络如ResNet、DenseNet、VGG等结构，实际上也是采用了以下组件组装而成的：

- identity 
- 卷积层（3x3、5x5、7x7）

- 深度可分离卷积
- 空洞卷积
- 组卷积
- 池化层
- Global Average Pooling

- 其他

如果对网络结构的设计进行建模的话，希望通过搜索的方法得到最优的网络结构，这时就需要NAS了。

NAS需要在有限的计算资源达到最好的效果，同时尽可能将更多的步骤自动化。在设计NAS算法的过程通常要考虑几个因素：

- search space 如何定义搜索空间
- search strategy 搜索的策略
- evaluation strategy 评估的策略

## 3. 早期NAS的特征

早期的NAS的结构如下图所示：

![NAS的大体架构](https://img-blog.csdnimg.cn/20201109203458812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

- 找一些预定义的操作集合（eg 卷积、池化等）这些集合构成了Search Space搜索空间。
- 采用一定的搜索策略来获取大量的候选网络结构。
- 在训练集训练这些网络，并且在验证集测试得到这些候选网络的准确率。
- 这些候选网络的准确率会对搜索策略进行反馈，从而可以调整搜索策略来获得新一轮的候选网络。重复这个过程。
- 当终止条件达到（eg:准确率达到某个阈值），搜索就会停下来，这样就可以找到准确率最高对应的网络架构。
- 在测试集上测试最好的网络架构的准确率。

下面介绍的几个网络都是遵从以上的结构进行设计的：

**1. NAS-RL**

NAS-RL发现神经网络的结构可以用一个变长字符串来描述，这样的话就可以使用RNN来作为一个控制器生成一个这样的字符串，然后使用**强化学习算法**来优化控制器，通过这种方法得到最终最优的网络架构。

**2. MetaQNN**

MetaQNN将选择网络架构的过程看作**马尔可夫决策过程**，用**Q-learning**来记录得到的奖励，通过这样的方法得到最优的网络架构。

**3. Large-scale Evolution**

Large-scale Evolution目标是使用**进化算法**(Evolutionary Algorithms)来学习一个最优的网络架构。使用一个最简单的网络结构来初始化种群，通过交叉、突变等方法来选择最优种群。

**4. GeNet**

GeNet也采用了**进化算法**，提出了一个新的神经网络架构编码机制，用定长的二进制串来表征一个网络的结构。GeNet随机初始化一组个体，使用预选定义好的基因操作（将二进制串看作一个个体的基因）来修改二进制串进而产生新的个体，从而最终选择出具有竞争力的个体作为最终的网络架构。

总结一下早期的NAS的特征：

- **全局搜索策略**：早期NAS采用的策略是搜索整个网络的全局，这就意味着NAS需要在非常大的搜搜空间中搜索出一个最优的网络结构。搜索空间越大，计算的代价也就越大。
- **离散的搜索空间**：早期NAS的搜索空间都是离散的，不管是用变长字符串也好，还是用二进制串来表示，他们的搜索空间都是离散的，如果无法连续，那就意味着无法计算梯度，也无法利用梯度策略来调整网络模型架构。
- **从头开始搜索**：每个模型都是从头训练的，这样将无法充分利用现存的网络模型的结构和已经训练得到的参数。

### 3.1 全局搜索

![](https://img-blog.csdnimg.cn/20201109225817417.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

全局搜索一般采用了以上两种常见的搜索形式，左侧图是最简单的一个链式结构；右侧图在其之上添加了skip connection。**MNASNet**吸取了人工设计的经验，希望搜索多个按顺序连接的**段（segment）**组成的网络结构，其中每个段都具有各自重复的结构。

> **注记**：跳转连接往往可以采用多种方式进行特征融合，常见的有add, concate, attention等。作者在文中提到了实验证明，add操作要比concate操作更加有效（原文：the sum operation is better than the merge operation）所以在NAS中，通常采用Add的方法进行特征融合操作。

### 3.2 从头搜索

早期NAS中，从头开始搜索也是一个较为常见的搜索策略。NAS-RL将网络架构表达为一个可变长的字符串，采用RNN作为控制器来生趁这个可变长字符串，根据该串可以得到一个相对应的神经网络的架构，采用强化学习作为搜索的策略对网络搜索方式进行调整。

![](https://img-blog.csdnimg.cn/20201110093132587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

MetaQNN考虑训练一个Agent/Actor来让其在搜索空间中按照顺序选择神经网络的架构。MetaQNN将这个选择过程看作马尔可夫决策过程，使用Q-learning作为搜索策略进而调整Agent/Actor所决定执行的动作。

> https://bowenbaker.github.io/metaqnn/

![](https://img-blog.csdnimg.cn/20201110093724416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

GeNet也采用了将网络架构编码的方式，提出了用二进制编码来表达网络的架构。这个二进制编码串被视为这个网络个体的DNA，这样可以通过交叉、编译等操作，产生新的网络架构，使用进化算法得到最优个体，也就是最好的网络架构。

![](https://img-blog.csdnimg.cn/20201110094136138.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

Large-scale Evolution只用了一个单层的、无卷积操作的模型作为初始的进化个体，然后使用进化学习方法来进化种群，如下图所示：

![Large-scal Evolution 图示](https://img-blog.csdnimg.cn/20201110092913564.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

可以看到，最初的个体仅仅使用了一个全局池化来构建模型，随后的个体中，可以再加入卷积、BN、激活函数等组件，构成下一代的种群。

## 4. 优化策略

以上是早期NAS中的操作，这部分主要从模块搜索策略、连续搜索空间、网络的循环利用和不完全训练这几点分别展开。

### 4.1 模块搜索策略

搜索空间的设计非常重要，不仅决定了搜索空间的大小，还决定了模型表现的上限。搜索完整的模型的搜索空间过大，并且存在泛化性能不强的问题。也就是说在一个数据集上通过NAS得到的最优模型架构，在另外一个数据集上往往不是最优的。

所以引入了模块化的搜索策略，基于单元模块进行搜索可以**减少NAS搜索空间的复杂度**。只需要搜索单元，然后就可以通过重复堆叠这些单元来构成最终的网络架构。基于单元的模块搜索方法的**扩展性**要强于全局搜索，不仅如此，模型的表现也更加出色。与全局搜索空间相比，基于单元的搜索空间更加**紧凑、灵活**。NASNet是最先开始探索单元化搜索的，其中提出了两种模块：Normal Cell和Reduction Cell。

- Normal Cell也就是正常模块，用于提取特征，但是这个单元不能改变特征图的空间分辨率。
- Reduction Cell和池化层类似，用于减少特征图的空间分辨率。

整体网络架构就应该是Normal Cell+Reduction Cell+Normal Cell...这样的组合，如下图所示：

![](https://img-blog.csdnimg.cn/20201110124511445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center) 

左侧展示了两种cell,有图展示的是其中一个ReductionCell搜索得到的模块内结构。

在ENAS中，继承了NASNet这种基于单元搜索的搜索空间，并通过实验证明了这种空间是非常有效的。

之后的改进有：

- 使用单元操作来取代reduction cell（reduction cell往往比较简单，没有必要搜索，采用下采样的单元操作即可，如下图Dpp-Net的结构）下图Dpp-Net中采用了密集连接。

![](https://img-blog.csdnimg.cn/20201110125139625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center) 

- Block-QNN中直接采用池化操作来取代Reduction cell
- Hierarchical-EAS中使用了3x3的kernel size, 以2为stride的卷积来替代Reduction Cell。
- Dpp-Net采用平均池化操作取代Reduction Cell。同时采取了Dense连接+考虑了多目标优化问题。
- 在视频任务：SNAS使用了Lx3x3,stride=1,2,2的maxpooling取代Reduction Cell。
- 在分割任务：AutoDispNet提出了一个自动化的架构搜索技术来优化大规模U-Net类似的encoder-decoder的架构。所以需要搜索三种：normal、reduction、upsampling。

> **注记：**通过研究这些搜索得到的单元模块，可以得到以下他们的**共性**：由于现存的连接模式，**宽而浅的单元**（采用channel个数多，但层数不多）在训练过程中更容易**收敛**，并且更容易搜索，但是缺点是**泛化性能很差**。

### 4.2 连续的搜索空间

**NASNet**使用强化学习算法需要2000个GPU days才能在CIFAR-10数据集上获得最好的网络架构。AmoebaNet使用进化算法需要3150个GPU days才能完成。这些主流的搜索算法可以是基于RL、EA（进化算法）、贝叶斯优化、SMBO（基于顺序模型的优化）、MCTS（蒙特卡洛搜索树）的。这些算法将神经网络搜索看作一个在**离散空间中的黑盒优化问题**。

**DAS**将离散的网络架构空间变为连续可微分的形式，然后使用梯度优化技术来搜索网络架构。其主要集中于搜索卷积层的超参数如：filter size，通道个数和卷积分组情况。

**MaskConnect**发现现有的基于单元的网络架构通常是在两个模块之间采用了预定义好连接方法，比如，一个模块要和这个模块之前的所有模块都进行连接。MaskConnect认为这种方法可能并不是最优的，采用了一种梯度的方法来控制模块之间的连接。

但是以上方法仅限于微调特定的网络架构。

为了解决以上问题，DARTS将离散的搜索空间松弛到了连续的空间，这样就可以使用梯度的方法来有效的优化搜索空间。DARTS 也采用了和NASNet一致的基于单元的搜索空间，并进一步做了一个归一化操作。这些单元就可以看成一个有向不循环图。每个单元有两个输入节点和一个输出节点。对于卷积的单元来说，输入单元就是前两个单元的输出。

TODO

规定一组符号来表达DARTS网络：

- 中间节点$x^{(j)}$代表潜在的特征表达，并且和每个前的节点$x^{(i)}$都通过一个有向边操作$o^{(i,j)}$。对一个离散的空间来说，每个中继节点可以这样表达：

$$
x^{(j)}=\sum_{i\lt j}o(i,j)(x^{(i)})
$$



- 在DARTS中，通过一个类似softmax来松弛所有可能的操作，这样就将离散搜索空间转化为连续的搜索空间问题。

$$
\bar{o}^{(i,j)}(x)=\sum_{o\in O}\frac{e^{(\alpha_o^{(i,j)})}}{\sum_{o'\in O}e^{\alpha_{o'}^{(i,j)}}}
$$

$O$代表的是一系列候选操作；$\alpha_o^{(i,j)}$代表的是有向边$e^{(i,j)}$上的操作$o$的参数；

通过以上过程就可以将搜索网络架构的问题转化为一个对连续变量$\alpha=\{\alpha^{(i,j)}\}$进行优化的过程。搜索完成之后，需要选择最有可能的操作$e^{(i,j)}$ 边上的$o^{(i,j)}$操作，而其他操作将会被丢弃：
$$
o^{(i,j)}=argmax_{o\in O}\alpha_o^{(i,j)}
$$
这样DARTS就相当于在求解一个二次优化问题，在优化网络权重的同时还需要优化混合操作（也就是$\alpha$）
$$
\begin{array}{cl}\min _{\alpha} & \mathcal{L}_{v a l}\left(w^{*}(\alpha), \alpha\right) \\ \text { s.t. } & w^{*}(\alpha)=\operatorname{argmin}_{w} \mathcal{L}_{\text {train}}(w, \alpha)\end{array}
$$

$\mathcal{L}_{val}$代表验证集上的Loss, $\mathcal{L}_{train}$代表训练集上的Loss。其中$\alpha$是高层变量，$\mathcal{w}$是底层变量，通过同时优化这个网络，将会获得最优的$\alpha$, 然后最终网络结构的获取是通过离散化操作完成的的。具体流程可以结合下图理解：

![DARTS算法的松弛过程](https://img-blog.csdnimg.cn/20201110191932416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)



**NAO**选择将整个网络结构进行编码，将原先离散的网络架构映射为连续的嵌入编码(encoder过程)。然后使用梯度优化方法去优化**模型表现预测器(performance predictor)**的输出，从而获得最优的嵌入编码。最后用一个解码器来离散化最优的嵌入编码，变成一个最优的网络架构(decoder过程)。









## 评价





## 参考文献

https://www.cc.gatech.edu/classes/AY2021/cs7643_fall/slides/L22_nas.pdf

https://www.media.mit.edu/projects/architecture-selection-for-deep-neural-networks/overview/



