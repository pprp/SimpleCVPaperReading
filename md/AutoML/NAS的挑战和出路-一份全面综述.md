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

DARTS和NAO是同一段时期提出来的算法，都存在一个问题：训练完成以后，将已经收敛的**父代网络**使用argmax进行离散化，得到**派生的子代网络**，这个离散化的过程带来的**偏差bias会影响模型的表现**，所以还需要对子代网络进行重新训练。由于父代网络已经在验证集中进行了测试，是最优模型。因此父代网络和派生的子代网络表现相差越少越好，这样最终得到的子代网络表现才能尽可能更好。

为解决以上问题，**SNAS**首先使用了强化学习中的延迟奖励，并且分析了为什么延迟奖励会导致收敛速度变慢的问题。然后SNAS提出重建NAS的方法来理论上绕开延迟奖励的问题，并且同时采用了连续化网络参数的方法，这样网络操作（network operation）的参数和网络架构的参数就可以使用梯度方法同时进行优化。

在SNAS、DARTS等算法中，搜索的过程中，所有可行的路径都存在**耦合**关系。尽管SNAS减少了派生的子网络和父代网络之间的差异，SNAS在验证阶段依然需要选择其中一条路径。

为此，**DATA**开发了一个EGS估计器（Ensemble Gumbel-Softmax）,其功能是将不同路径的关系进行**解耦**，然后就可以让不同路径之间梯度的无缝迁移。

**I-DARTS**指出了基于Softmax方法的松弛方法可能导致DARTS变成一个"局部模型"。当前DARTS设计是：每个节点都和之前的所有节点连接，在离散化以后得到的模型，每两个节点之间必有一条边连接。**I-DARTS认为这种设计并没有理论依据并且限制了DARTS的搜索空间。**

I-DARTS解决方案是同时考虑所有的输入节点，看下面示意图：

![I-Darts示意图-和DARTS进行对比](https://img-blog.csdnimg.cn/20201112102408914.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

可以看到，DARTS的softmax是操作在每两个节点之间，I-DARTS做出的改进对一个节点所有的输入都是用一个Softmax来统一，这样最终结果就不是每两个节点之间都需要有一个边。这样可以让模型具有更强的灵活性。

**P-DARTS**发现DARTS存在搜索和验证的网络架构存在巨大的gap：DARTS本身会占用很大的计算资源，所以在**搜索阶段**只能搜索浅网络。在**验证阶段**，DARTS会将浅层的网络堆叠起来从而形成一个深层网络。两个阶段训练的网络存在gap，所以P-DARTS提出了一个采用**渐进的搜索策略**，在训练过程中逐步增加网络的深度，同时根据混合操作的权重逐渐减少候选操作集合的大小来防止计算量爆炸。为了解决这个过程中出现的搜索不稳定的问题，P-DARTS引入了正则化防止算法过于偏向skip-connect。

**GDAS（百度CVPR19）**发现同时优化不同的子操作会导致训练不稳定，有些子操作可能是竞争关系，直接相加会抵消。所以GDAS提出了采样器得到subgraph，每次迭代的更新只对subgraph进行即可。通过这个操作可以让识别率基本不变的情况下，搜索速度加快5倍。

![GDAS示意图](https://img-blog.csdnimg.cn/20201112110409403.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center) 

上图是GDAS的示意图，只有黑色实线代表这个路径被sample到了，在这个迭代中，只需要训练这个subgraph即可。

**PC-DARTS**致力于减少显存占用、提高搜索效率，进行快速的大batch搜索。设计了基于channel的采样机制，每次只有一部分channel会用来进行训练。但是这样会带来不稳定的问题，于是提出edge normalization,搜索过程中通过学习edge-level超参数来减少不确定性。

除了以上工作，还有很多工作都是基于DARTS进行改进的，DARTS由于其简单而优雅的结构，相关研究非常丰富，组成了NAS的一个重要研究方向。

### 4.3 网络架构重复利用

早期的NAS通常采用的方法是**从头开始训练**，这种方法从某种角度来讲，可以增加模型的搜索的自由度，很有可能能够设计出一个效果很棒的网络架构。但是，从头训练也会带来让搜索的时间复杂度上升，因为其并没有利用到前线已经设计好的网络架构权重，未充分利用这部分先验知识。

一个新的想法是将已有的人工设计的网络架构作为起始点，然后以此为基础使用NAS方法进行搜索，这样可以以更小的计算代价获得一个更有希望的模型架构。

以上想法就可以看作是Network Transform或者Knowledge Transform（知识迁移）。

**Net2Net**对知识迁移基础进行了详尽的研究，提出了一个“功能保留转换”（Function-perserving Transformation）的方法来实现对模型参数的重复利用，可以极大地加速训练新的、更大的网络结构。

![Net2Net的思想](https://img-blog.csdnimg.cn/20201112190059886.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

其中主要有Net2WiderNet和Net2DeeperNet，两个变换可以让模型变得更宽更深。

![Net2Wider](https://img-blog.csdnimg.cn/20201112190249127.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

Wider就是随即从已有节点中选择一个节点复制其权重，如上图右侧的h3选择复制了h2的参数。对于输出节点来说，需要把我们选择的节点的值都除以2，这样就完成了全连接层的恒等替换。（卷积层类似）

![Net2Deeper](https://img-blog.csdnimg.cn/20201112190838829.png#pic_center)

Deeper就是加深网络，对全连接层来说，利用一个单位矩阵做权值，添加一个和上一个全连接层维度完全相同的全连接层，把前一个全连接层的权重复制过来，得到更深的网络。

基于Net2Net，Efficient Architecture Search(EAS)进行了改进，使用强化学习的Agent作为元控制器(meta-controller),其作用是通过“功能保留转换”增加深度或者宽度。

![EAS示意图](https://img-blog.csdnimg.cn/20201112192413504.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

Bi-LSTM就是mata-controller，用来学习底层表达的特征，输出结果被送到Net2Wider Actor 和 Net2Deeper Actor用于判断对模型进行加深或者加宽的操作。

![Net2Wider Actor](https://img-blog.csdnimg.cn/20201112192605826.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

Net2Wider Actor使用共享的Sigmoid分类器基于Encoder输出结果来决定是否去加宽每个层。

![Net2Deeper Actor](https://img-blog.csdnimg.cn/20201112192713260.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

Net2Deeper Actor使用循环神经网络来顺序的决定是否添加一个新的层和对应的网络参数。

**N2N Learning**中没有使用加宽或者加深的操作，而是考虑通过移除层或者缩小层来压缩**教师网络**。利用强化学习对网络进行裁剪，从Layer Removal和Layer Shrinkage两个维度进行裁剪，第一个代表是否进行裁剪，第二个是对一层中的参数进行裁剪。

![N2N learning](https://img-blog.csdnimg.cn/20201112223440890.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

- 首先使用layer removal操作
- 然后使用layer shrinkage操作
- 然受使用强化学习来探索搜索空间
- 然后使用只是蒸馏的方法训练每个生成得到的网络架构。
- 最终得到一个局部最优的学生网络。

该方法可以达到10倍数的压缩率。

**Path-level EAS**实现了Path level（一个模块之内的路径path）的网络转换。这种想法主要是来源于很多人工设计的多分支网络架构的成功，比如ResNet、Inception系列网络都使用到了多分支架构。Path-level EAS通过用使用多分枝操作替换单个层来完成路径级别的转换，其中主要有分配策略和合并策略。

分配策略包括Replication和Split:

![PATH-level EAS的分配策略](https://img-blog.csdnimg.cn/20201113154400257.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

- Replication就是将输入x复制两份，分别操作以后将得到的结果除以2再相加得到输出。
- Split就是将x按照维度切成两份，分别操作以后，将得到的结果concate到一起。

合并策略包括：add和concatenation

![Path-level EAS示意图](https://img-blog.csdnimg.cn/20201113154632387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center) 

上图描述的是Path-level EAS的示意图：

（a）使用了Replication策略

（b）使用了Split策略

（c）将一个恒等映射替换为可分离3x3卷积

（d）表达了(c)中的网络结构的树状结构。

另一个类似的工作，**NASH-Net**基于Net2Net更进一步提出了四种网络变形的方法。NASH-Net可以同一个预训练的模型，使用四种网络形变方法来生成一系列子网络，对这些子网络进行一段时间的训练以后，找到最好的子网络。然后从这个子网络开始，使用基于爬山的神经网络架构搜索方法（Neural Architecture Search by Hill-Climbing）来得到最好的网络架构。

之前的网络通常研究的是图像分类的骨干网络，针对分割或者检测问题的网络一般无法直接使用分类的骨干，需要针对任务类型进行专门设计。尽管已经有一些方法用于探索分割和检测的骨干网络了，比如Auto-Deeplab、DetNas等，但是这些方法依然需要预训练，并且计算代价很高。

**FNA**(Fast Neural Network Adaptation)提出了一个可以以近乎0代价，将网络的架构和参数迁移到一个新的任务中。FNA首先需要挑选一个人工设计的网络作为种子网络，在其操作集合中将这个种子网络扩展成一个超网络，然后使用NAS方法（如DARTS,ENAS,AmoebaNet-A）来调整网络架构得到目标网络架构。然后使用种子网络将参数映射到超网络和目标网络进行参数的初始化。最终目标网络是在目标任务上进行微调后的结果。整个流程如下图所示：

![FNA流程](https://img-blog.csdnimg.cn/20201113161913558.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

> ps: 分类backbone和其他任务是有一定gap的,FNA认为通过微调神经网络带来的收益不如调整网络结构带来的收益）

### 4.4 不完全训练

NAS的核心一句话来概括就是使用搜索策略来比较一系列候选网络结构的模型表现找到其中最好的网络架构。所以如何评判候选网络也是一个非常重要的问题。

早期的NAS方法就是将候选网络进行完全的训练，然后在验证集上测试候选网络架构的表现。由于候选网络的数目非常庞大，这种方法耗时太长。随后有一些方法被提了出来：

- NAS-RL采用了并行和异步的方法来加速候选网络的训练
- MetaQNN在第一个epoch训练王以后就使用预测器来决定是否需要减少learning rate并重新训练。
- Large-scale Evolution方法让突变的子网络尽可能继承父代网络，对于突变的结构变化较大的子网络来说，就很难继承父代的参数，就需要强制重新训练。

#### 4.4.1 权重共享

上面的方法中，上一代已经训练好的网络的参数直接被废弃掉了，没有随后的网络充分利用，**ENAS**首次提出了参数共享的方法，ENAS认为NAS中的候选网络可以被认为是一个从**超网络结构**中抽取得到的**有向无环子图**

![ENAS Parameter Sharing](https://img-blog.csdnimg.cn/20201113181327482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

超网络结构就如上图绿色框部分所示，是节点和此节点之前所有节点进行连接，然后形成的网络。两个红色框就是从超网络结构中采样得到的一个单向无环图，这两个子网络共享的权重也就很容易找到$w_{21},w_{42},w_{63}$。这样ENAS就可以强制所有子网络都共享相同的参数。

然后ENAS使用LSTM作为一个控制器在超网络结构中找到最优子网络结构。通过这种方法，可以避免让每个子网络从头开始训练，可以更好地提高网络的搜索效率。

**CAS**(Continual and Multi-Task Architecture Search)基于ENAS探索了多任务网络架构搜索问题，可以扩展NAS在不同的数据集见进行迁移学习的能力。CAS引入了一个新的**连续架构搜索**方法来解决**连续学习过程**中的遗忘问题，从而可以继承上个任务中的经验，这对于多任务学习来说非常有帮助（感觉可以一定程度上避免过拟合）。

**AutoGAN**首先将GAN的思想引入NAS,并且使用了Inception Score作为强化学习的奖励值，使用ENAS中的参数共享和动态重设来加速搜索过程。训练过程中引入了Progressive GAN的技巧，逐渐的实现NAS。

**OFA**（Once for all）使用了一个弹性Kernel机制来满足多平台部署的应用需求和不同平台的视觉需求的多样性。小的kernel会和大的kernel共享权重。在网络层面，OFA优先训练大的网络，然后小的网络会共享大网络的权重，这样可以加速训练的效率。

此外，基于one-shot的方法也使用到了权重共享的思想。**SMASH**提出训练一个辅助的HyperNet，然后用它来为其他候选网络架构来生成权重。此外，SMASH对利用上了训练早期的模型表现，为排序候选网络结构提供了有意义的指导建议。

One-Shot Models这篇文章讨论了SMASH中的HyperNetwork和ENAS中的RL Controller的必要性，并认为不需要以上两者就可以得到一个很好的结果。

**Graph HyperNetwork**(GHN)推荐使用计算图来表征网络结构，然后使用**GNN**来完成网络架构搜索。GHN可以通过图模型来预测所有的自由权重，因此GHN要比SMASH效果更好，预测精度更加准确。

典型的one-shot NAS需要从HyperNet中通过权重共享的方式采样得到一系列候选网络，然后进行评估找到最好的网络架构。**STEN**提出从这些采样得到的候选网络中很难找到最好的架构，这是因为共享的权重与可学习的网络架构的参数紧密耦合。这些偏差会导致可学习的网络架构的参数偏向于简单的网络，并且导致候选网络的良好率很低。因为简单的网络收敛速度要比复杂网络要快，所以会导致HyperNet中的参数是偏向于完成简单的网络架构。

STEN提出了使用一个均匀的随机训练策略来平等的对待每一个候选网络，这样他们可以被充分训练来得到更加准确的验证集表现。此外，STEN还使用了评价器来**学习候选网络具有较低验证损失的概率**，这样极大地提升了候选网络的优秀率。

《Evaluating the search phase of neural architecture search》也认为ENAS中的权重共享策略会导致NAS很难搜索得到最优的网络架构。此外，FairNAS的研究和《 Improving One-shot NAS by Suppressing the Posterior Fading》中显示基于参数共享方法的网络结构很难被充分训练，会导致候选网络架构不准确的排序。

在DARTS、FBNet、ProxyLessNas这种同时优化超网络权重和网络参数的方法，会在子模型之间引入偏差。为了这个目的，**DNA**( Blockwisely Supervised Neural Architecture Search with Knowledge Distillation)提出将神经网络结构划分为互不影响的block，利用蒸馏的思想，引入教师模型来引导网络搜索的方向。通过网络搜索空间独立分块的权重共享训练，降低了共享权重带来的表征偏移的问题。

**GDAS-NSAS**也是希望提升one-shot Nas的权重共享机制，提出了一个NSAS的损失函数来解决多模型遗忘（当使用权重共享机制训练的时候，在训练一个新的网络架构的时候，上一个网络架构的表现会变差）的问题。

可微分的NAS使用了类似的权重共享策略，比如DARTS选择训练一个超网络，然后选择最好的子网络。ENAS这类方法则是训练从超网络中采样得到的子网络

#### 4.4.2 训练至收敛

在NAS中，是否有必要将每个候选网络都训练至收敛呢？答案是否定的。

为了更快的分析当前模型的有效性，研究院可以根据学习曲线来判断当前模型是否有继续训练下去的价值。如果被判定为没有继续训练的价值，就应该及早终止，尽可能节约计算资源和减少训练时间。

那么NAS也应该采取相似的策略，对于没有潜力的模型，应及早停止训练；对于有希望的网络结构则应该让其进行充分的训练。

《 Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves》一文中就提出了使用概率性方法来模拟网络的学习曲线。但是这种方法需要长时间的前期学习才能准确的模拟和预测学习曲线。

《Learning curve prediction with Bayesian neural networks》改进了上述方法，学习曲线的概率模型可以跨超参数设置，采用成熟的学习曲线提高**贝叶斯神经网络**的性能。

以上方法是基于部分观测到的早期性能进行预测学习曲线，然后设计对应的机器学习模型来完成这个任务。为了更好的模拟人类专家，《Accelerating neural architecture search using performance prediction.》首先将NAS和学习曲线预测结合到了一起，建立了一个标准的频率回归模型，获从网络结构、超参数和早期学习曲线得到对应的最简单的特征。利用这些特征对频率回归模型进行训练，然后结合早期训练经验**预测**网络架构最终验证集的性能。

**PNAS**中也用到了性能预测。为了避免训练和验证所有的子网络，PNAS提出了一个预测器函数，基于前期表现进行学习，然后使用预测器来评估所有的候选模型，选择其中topk个模型，然后重复以上过程直到获取到了足够数量的模型。

![基于前期结果进行预测的示意图](https://img-blog.csdnimg.cn/20201113233613487.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

上图就是一个示意图，在早停点停止以后，预测器可以预测网络随后的状态，得到预测的学习曲线。

**NAO**











## 评价





## 参考文献

https://www.cc.gatech.edu/classes/AY2021/cs7643_fall/slides/L22_nas.pdf

https://www.media.mit.edu/projects/architecture-selection-for-deep-neural-networks/overview/

https://blog.csdn.net/cFarmerReally/article/details/80927981

https://cloud.tencent.com/developer/article/1470080

Net2Net: http://xxx.itp.ac.cn/pdf/1511.05641

EAS：https://arxiv.org/abs/1707.04873

E2E learning: https://blog.csdn.net/weixin_30602505/article/details/98228471

Path-level EAS: https://blog.csdn.net/cFarmerReally/article/details/80887271

FNA：https://zhuanlan.zhihu.com/p/219774377

AmoebaNet: https://blog.csdn.net/vectorquantity/article/details/108625172

ENAS:https://zhuanlan.zhihu.com/p/35339663