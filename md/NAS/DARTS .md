# DARTS 可微分神经网络结构搜索开创者

【GiantPandaCV导读】在DARTS提出之前，所存在的基于强化学习或进化算法的架构搜索虽然能够搜索出较为优秀的网络架构，即使存在一些提速的方法，但搜索效率仍是低下，因为在离散的搜索空间上对网络架构进行搜索，需要进行大量的架构评估，而且由于待搜索参数是不可微分的，也就不能采用梯度下降算法进行优化，所以便有研究人员考虑放宽搜索空间，使其连续化，从而达到能够使用梯度下降来训练的目的。首先在神经网络架构搜索上提出使用可微方法搜索一种完整的网络架构的是发表在ICLR 2019的《DARTS: Differentiable Architecture Search》一文，基于梯度优化使得DARTS可以在使用很少的计算资源前提下，能够搜索出能与当前使用其他方法的自动神经网络架构搜索得到的网络架构相媲美。

## 1. NAS背景介绍

在神经网络架构搜索方面的开创性工作主要是2016年发表的两篇文章，分别是MIT的《Designing Neural NetworkArchitectures using Reinforcement Learning》和Google的《NeuralArchitecture Search with Reinforcement Learning》，值得一提的是，这两篇文章中所采用的方法均是采用基于强化学习的搜索策略，在形式上比较简单，并且取得了可观的效果，在图像分类任务上以及语言建模任务上超越了在此之前手工设计的网络。但是，由于搜索空间是离散的，在搜索过程中就需要不断重复地去挑选结构，然后再去评估模型的好坏，是非常消耗计算资源的。例如Google所提出的策略仅仅在CIFAR-10这么一个小的数据集上去搜索架构就需要800张GPU计算4周的时间。由此看来，直接在ImageNet这种大规模数据集上搜索显然是很难做到的。于是在2017年，Google发表一篇名为《Learning TransferableArchitectures for Scalable Image Recognition》，提出NASNet架构，搜索策略同样是基于强化学习，用于大规模的图像分类和识别。其实相比于上一篇NAS来说，无非是做了一部分改进，借鉴了当时主流的像ResNet这样优秀网络架构的重复堆叠想法，搜索模式从原来的直接搜索整个网络架构变为先搜索小的Cell然后在拼接成完整的网络架构。另外还实现了从CIFAR-10上搜索出来的最好架构转移到ImageNet或COCO这种大规模数据集上。虽然NASNet解决了NAS应用到大规模数据集上的困难，但是仍然还是很消耗计算资源，在相同的硬件基础上，NASNet仅比先前的NAS搜索速度快7倍，如此昂贵的搜索代价使得普通研究者们在该研究领域止步不前，直到ENAS的出现。在2018年，Google再次发布与NAS相关的论文，其名为《Efficient NeuralArchitecture Search via Parameter Sharing》，顾名思义，目的就是为了使NAS搜索更为高效，是第一篇将权重共享思想引入到NAS中来。此前的NAS方法均是每次在强化学习选择出子网络后，重新训练一篇网络模型后验证模型精度，ENAS是在NAS的基础上提出通过使用网络参数共享，即选择出子网络后，在已训练的模型参数上继续训练，从而达到极大地缩短搜索时间。从文章的结果可以得知，ENAS搜索出来的模型精度相比于NAS来说稍逊一些，但是在训练效率上，远远优于NAS和NASNet，在一张GTX1080Ti上仅仅需要使用10个小时的时间就可以完成在CIFAR-10数据集上的搜索，也就说在一天内，ENAS可以通过自动训练得到一个优于人工设计的模型，进一步促进了自动神经网络架构搜索的发展。当然ENAS本身也存在着一定的缺陷性，因为是通过抽样方式选择参数，搜索到的结构可能不是最优解。

与此同时，基于进化算法的神经网络架构搜索也在不断地发展着。2017年初Google的《Large-Scale Evolution of ImageClassifiers》首次尝试将进化算法结合到神经网络架构搜索中来之后，基于进化算法逐渐便成了神经网络架构搜索的另一个重要分支。然后在这一篇文章中，由于初始条件太差，这意味着，如果想要得到可观的结果，就需要采用大规模种群来扩大搜索范围，因此为了找到局部最优解，算法运行所需时间太长，计算成本较为昂贵。于是在2018年，针对进化算法在神经网络架构搜索中所存在的缺陷，Google在文章《RegularizedEvolution for Image Classifier Architecture Search》中，采用和NASNet相同的搜索空间，但是对进化算法做出了改进，引入年龄属性，将锦标赛选择法变为基于年龄的选择，很大程度上提升了搜索效率和搜索出来的模型精度。

除强化学习和进化算法外，不乏还有其他的搜索策略，例如基于可微的搜索方法，基于随机搜索策略、贝叶斯优化等搜索方法，而DARTS便是首次提出基于可微来进行整体架构搜索。

## 2. 搜索空间

在DARTS中，同样采用了NASNet中所提出的Cell结构，然后在后续步骤中逐步把Cell结构堆叠成卷积网络或者递归网络。但是与NASNet中的Cell不同之处在于，DARTS中的Cell是由 个有序节点构成的有向无环图，把节点与节点之间的有向边表示成某种操作，通过公式来训练出最合适的Cell结构。具体内容如下：

以$x^{(i)}$表示有向无环图中的节点，$o^{(i,j)}$表示从$x^{(i)}$到$x^{(j)}$ 的有向边，用来表示某种操作。图的起始节点作为输入，终止节点表示输出，图的中间节点则表示计算出来的特征，对于卷积单元和递归单元来说，其输入节点的定义是不同的。由于中间节点需要根据其所有的上一个节点计算得到，所以规定每个中间节点的表达式为


$$
x^{(j)}=\sum_{i\lt j} o^{(i, j)}\left(x^{(i)}\right)
$$

另外，为了表示两个节点之间没有任何连接，引入了一个特殊操作，即零操作。


## 3. 连续松弛和优化

在定义好搜索空间后，为了使搜索空间变得连续，这里将操作的选择简化成所有可能操作的SoftMax，公式如下：

$$
\bar{o}^{(i, j)}(x)=\sum_{(o \in \mathrm{O})} \frac{\exp \left(\alpha_{o}^{(i, j)}\right)}{\Sigma\left(o^{\prime} \in \mathrm{O}\right)\left(\alpha_{0}^{\prime(i, j)}\right)}
$$


其中$O$ 表示搜索空间内所有可能操作的集合，$\alpha $是一个长度为$|O|$的向量， $\alpha_o $ 则表示对应于操作$o$的权重。在搜索完成后，用 $O$ 中能使 $\alpha$ 中分量最大的那个操作$o$即 $o^{(i, j)}=\underset{o \in \mathrm{O}}{\operatorname{argmax}} \alpha_{o}^{(i, j)}$来代替混合操作 $\bar{o}^{(i,j)}$，作为最终两个节点之间的连接。

另外，不仅需要学习网络架构$\alpha $，还需要学习权重参数$\omega $，所以在DARTS中存在着双层优化问题，每当确定某个$\alpha $后，需要在训练集上得到最优的 $\omega $，并在验证集上评估。文中用$L_{train}$表示训练损失，用$L_{val}$表示验证损失，整个DARTS搜索过程便抽象写成优化问题，即描述为:

$$
\begin{aligned}
&\min _{\alpha} \mathrm{L}_{v a l}\left(w^{*}(\alpha), \alpha\right) \\
&\text { s.t. } w^{*}(\alpha)=\operatorname{argmin}_{w} \mathrm{~L}_{\text {train }}(w, \alpha)
\end{aligned}
$$


由于这样的双层优化执行起来耗时太长，实现起来极其地不容易，文章中便提出一种近似方案：

$$
\nabla_{\alpha} \mathrm{L}_{\text {val }}\left(w^{*}(\alpha), \alpha\right) \approx \nabla_{\alpha} \mathrm{L}_{v a l}\left(w-\xi \nabla_{w} \mathrm{~L}_{\text {tran }}(w, \alpha), \alpha\right)
$$


其中，$\xi$ 表示$\mathrm{~L}_{\text {tran }}(w, \alpha)$使用的学习速率。也就是说，在计算$\alpha$梯度时，并没有等到$\omega$完全收敛，而是按照当前$\alpha$计算出的$\omega$的梯度后，用$\xi$对$\omega$进行一次更新。将上述公式展开后得到:

$$
\nabla_{\alpha} \mathrm{L}_{v a l}\left(w^{\prime}, \alpha\right)-\xi \nabla_{\alpha, w}^{2} \mathrm{~L}_{\mathrm{train}}(w, \alpha) \nabla_{\dot{w}} \mathrm{~L}_{v a l}\left(w^{\prime}, \alpha\right)
$$


文章中提出，当$\xi = 0$时，称之为一阶近似，计算速度比较快，但是会降低性能， $\xi \gt  0$，称之为二阶近似，从搜索结果可以得知，二阶近似的搜索结果要优于一阶近似。

## 4. 小结

整个DARTS的工作过程可分为4部分，首先是初始化节点，节点间的操作是未知的，其次，通过在每个边上的混合操作生成一个连续的搜索空间，然后通过对混合概率以及网络权重参数进行优化，最终从混合操作概率中得到网络结构。

通过DARTS算法搜索神经网络架构，相比于之前的基于强化学习和进化算法来说，开辟了一个新的研究方向，通过可微的方法进行神经网络的架构搜索，这在很大程度上缩短了搜索时间，提高了搜索效率。然而DARTS中仍然存在着一些问题，诸如网络迁移问题，还有双层优化导致参数间的竞争问题等。


## 5. 参考文献

[1] Barret Zoph, Vijay Vasudevan,Jonathon Shlens, et al. Learning transferable architectures for scalable imagerecognition. In Conference on Computer Vision and Pattern Recognition, 2018.

[2] Liang-ChiehChen, Maxwell Collins, Yukun Zhu, et al. Searching for efficient multi-scalearchitectures for dense image prediction. In S. Bengio, H. Wallach, H.Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances inNeural Information Processing Systems 31, pages 8713–8724. Curran Associates,Inc., 2018. URL http://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf.

[3] Thomas Elsken, JanHendrik Metzen, Frank Hutter. Neural Architecture Search: A Survey[J]. arXivpreprint arXiv:1808.05377v3, 2019.

[4] BAKER B, GUPTAO, NAIK N, et al. Designing neural network architectures using reinforcementlearning [J]. arXiv preprint arXiv:1611. 02167, 2016.

[5] Barret Zoph,Quoc V. Le. Neural Architecture Search with Reinforcement Learning [J]. arXivpreprint arXiv:1611. 01578, 2016.

[6] Barret Zoph,Vijay Vasudevan, Jonathon Shlens, et al. Learning Transferable Architecturesfor Scalable Image Recognition[J]. arXiv preprint arXiv:1707.07012, 2017.

[7] Hieu Pham,Melody Y. Guan, Barret Zoph, et al. Efficient Neural Architecture Search viaParameter Sharing[J]. arXiv preprint arXiv:1802.03268, 2018.

[8] REAL E, MOORE S,SELLE A et al. Large-scale evolution of image classifiers [C]//Proceedings ofthe 34th International Conference on Machine Learning-Volume 70. JMLR. Org, 2017:2902－2911.

