# PC-DARTS 部分通道链接



## 1. 引言

尽管P-DARTS的渐进式搜索能够使得搜索时的Cell逐步增加到接近验证时的Cell数目，但以牺牲了一部分候选操作的数量为代价，因此还算不上一个完美的解决方案。因此华为在ICLR 2020上发表一篇名为《PC-DARTS: Partial Channel Connections for Memory-Efficient Differentiable Architecture Search》的文章，意在大规模地节省计算量以及内存，从而使得在搜索过程中能做到更快更好。

## 2. PC-DARTS构成

![PC-DARTS架构](https://img-blog.csdnimg.cn/31edb6a829174113a6e80bfc1e9236cc.png)

PC-DARTS在DARTS的基础上做了进一步的改进，具体过程如上图所示，文中所提出策略将网络提取的特征在通道维度上进行1/K采样，把采样的通道发送到|O|操作中进行混合计算，然后再对处理后的特征与剩余的特征进行拼接（concat），在实验中K取值为4，计算公式如下：
$$
f_{i, j}^{\mathrm{PC}}\left(\mathbf{x}_{i} ; \mathbf{S}_{i, j}\right)=\sum_{o \in \mathrm{O}} \frac{\exp \left\{\alpha_{i, j}^{\circ}\right\}}{\sum_{0} \in \mathrm{exp}\left\{\alpha_{i, j}^{o^{\prime}}\right\}} \cdot o\left(\mathbf{S}_{i, j} * \mathbf{x}_{i}\right)+\left(1-\mathbf{S}_{i, j}\right) * \mathbf{x}_{i}
$$


通过部分通道连接这种策略来进行操作选择，其优点显而易见，每次只有1/K通道的节点来进行操作选择，这样减少了1-1/K的内存，在搜索过程中，可以把bitchsize设置为原来的K倍，有益于架构搜索的稳定性。



## 3. 边归一化

虽然通道采样的做法有着较为积极的影响，但是仍然存在着一定的缺陷。由于每一个节点$x_j$的输出需要从$\left\{x_{0}, x_{1}, \ldots, x_{j-1}\right\}$中选取两个输入节点，权重分别是$\left\{\max _{o} \alpha_{0, j}^{o}, \max _{o} \alpha_{1, j}^{o}, \ldots, \max _{o} \alpha_{j-1, j}^{o}\right\}$，但是因为通道采样的随机性，所确定的最优连通性可能是不稳定的，从而导致最终的架构变得不稳定。

作者为解决这个缺陷，提出边归一化，即对每条边进行加权处理，公式如下:
$$
\mathbf{x}_{j}^{\mathrm{PC}}=\sum_{i \lt j} \frac{\exp \left\{\beta_{i, j}\right\}}{\sum_{i \lt j} \exp \left\{\beta_{i, j}\right\}} \cdot f_{i, j}\left(\mathbf{x}_{i}\right)
$$
其中，$\beta_{i,j}$ 表示对（i,j）边的加权。

这样，架构搜索完成后，边之间的连接变成了由$\alpha^o$和$\beta$共同决定，所以让归一化系数即：
$$
\frac{\exp \left\{\beta_{i, j}\right\}}{\sum_{i \lt j} \exp \left\{\beta_{i, j}\right\}} \times \frac{exp\{\alpha_{i,j}^{o}\}}{\sum_{o \in O} \exp \left\{\alpha_{i, j}^{o}\right\}}
$$
然后就像DARTS中一样通过找大的边权值来选择边。由于$\beta $在整个训练过程中是共享的，因此学习到的网络架构对迭代时的采样通道不敏感，从而使架构搜索更加稳定，并且归一化操作的计算开销可以忽略不计。









