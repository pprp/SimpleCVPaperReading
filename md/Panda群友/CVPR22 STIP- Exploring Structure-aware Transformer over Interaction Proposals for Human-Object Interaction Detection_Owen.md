# [CVPR22] STIP: Exploring Structure-aware Transformer over Interaction Proposals for Human-Object Interaction Detection
* Paper title:[Exploring Structure-aware Transformer over Interaction Proposals for Human-Object Interaction Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Exploring_Structure-Aware_Transformer_Over_Interaction_Proposals_for_Human-Object_Interaction_Detection_CVPR_2022_paper.pdf)
* Task: Human-Object Interaction Detection
* Code: [https://github.com/zyong812/STIP.](https://github.com/zyong812/STIP)




## Human-Object Interaction (HOI) Detection Background and Motivations
HOI Detection要求定位出存在交互的人、物并给出两者之间的动作关系，即最终要求给出三元组$<human,object,interaction>$。实际的HOI系统执行以人为中心的场景理解，因此有着广泛的用途，例如监控事件监测和机器人模仿学习。传统的HOI范式倾向于以多阶段的方式来解决这个有挑战性的复杂问题，即先执行object detection，再执行动作关系的分类，这种范式需要繁重的后处理（post-processing），例如启发式匹配来完成任务，这导致其无法以端到端的方式进行training，导致了次优的性能。最近sota的一系列HOI方法往往受启发于DETR，将HOI Detection视为集合预测问题来克服这一问题，实现end-to-end的训练优化，这类方法的基本流程如下图（a）中所示，可以看出，这类方法往往利用transformer来将可学习的queries映射为HOI的预测集合，从而实现one-stage的HOI detection，然而，这些HOI检测范式中的parametric interaction queries（参数化的交互queries）往往是随机初始化的，这就导致范式中的queris和输出HOI 预测之间的对应关系是**动态的**，其中对应于每个目标HOI三元组的query，例如$<human,bat,hold>$，在预测开始时往往是**未知的**，这将严重影响模型去探索先验知识，即inter-interaction 或 intra-interaction structure，即交互间的结构性关系和交互内的结构性关系知识，而这对于交互间的关系reasoning是非常有帮助的。

![（a）之前的Transformer风格的HOI检测范式（b）本文方法示意图](https://img-blog.csdnimg.cn/da6a1f7438ab44f78b3a3ffe4f84ef3d.png#pic_center)


### Inter&Intra-interaction Structure For HOI Detection
**交互间的结构性（Inter-interaction Structure）非常有助于互相提供线索来提高检测效果**，例如上图中“human wear (baseball) glove” 就提供了非常强的线索来提示另一个interaction：“human hold bat”，有趣的是,**内部交互结构（Intra-interaction Structure）可以解释为每个HOI三元组的局部空间结构，例如人和物体的布局结构以一种额外的先验知识来将model的注意力引导到有效的图像区域，从而描述交互行为**。

## STIP : Structure-aware Transformer over Interaction Proposals
讲了背景知识和基本的motivations，终于步入正题了，作者提出的方法叫做STIP（ Structure-aware Transformer over Interaction Proposals），其将一阶段的HOI检测方案分解为了两阶段的级联pipeline：首先产生交互proposals（有可能存在交互的人-物对），接着基于这些proposal 执行HOI集合预测，如上图中所示，STIP将这些proposal视为非参交互queries，从而启发后续的HOI集合预测问题，也可以将其视为静态的、query-based的HOI检测pipeline。下面将分别介绍Interaction Proposal Network 、Interaction-centric Graph和Structure-aware Transformer。

![STIP整体流程示意图](https://img-blog.csdnimg.cn/4fa457c994774ff4aa4f644c9177d9e0.png#pic_center)
### Interaction Proposal Network
STIP利用DETR作为物体（和人）检测的base network，训练过程中，DETR部分的权重是冻住的，不进行学习，基于DETR给出的检测结果，Interaction Proposal Network（IPN）将构建存在潜在交互的的human-object对，对于每个human-object对，IPN将通过MLP给出潜在交互的分数，即 interactiveness score。只有Top-K个最高得分的human-object对将送入下一阶段。
### Human-Object Pairs Construction
STIP为了充分利用knowledge，从不同的信息层次来构建Human-Object对，每个HO对都由外观特征、空间特征、和语言学特征（linguistic features）来构成。具体来说，外观特征是从DETR中得到的human和object实例特征来构建，即分类头前的、256通道维度的区域特征（即ROI区域特征）。我们将human和object的bounding box定义为：$\left(c_x^h, c_y^h\right)$ and $\left(c_x^O, c_y^O\right)$，则空间特征由$
\left[d x, d y, \text { dis, } \arctan \left(\frac{d y}{d x}\right), A_h, A_o, I, U\right]
$来构建，其中$c_x^h-c_x^o, d y=c_u^h-c_u^o, d i s=\sqrt{d x^2+d y^2}$，$A_h, A_o, I, U$则分别代表了人的区域，物体的区域，交叉的区域和联合bounding box的区域信息。语言学特征则是将bounding box的类别名编码为one-hot向量，向量的通道维度大小为300。每个HO对都将被如上方式进行表征，最终concat到一起，送入MLP中。



### Interactiveness Prediction
构建Human-Object Pairs 后，将构建出的Human-Object Pairs 经过hard mining strategy（难样本挖掘策略）来筛选出负样本，正样本则是由置信度大于0.5的human和object的bounding box IOUs组成。STIP需要预测出每个proposal成立的可能度，因此将其视为一个二分类问题，从而利用Focal loss来进行优化。在推理阶段，只有top-K个最高得分的human-object 对将被送入下个阶段作为交互proposal。

### Interaction-Centric Graph
利用IPN来筛选出潜在的proposal后，接着STIP利用Interaction-Centric Graph来充分利用丰富的inter-interaction和intra-interaction structure的先验知识，在实际实现中，将每个interaction proposal作为一个单一的graph node（图节点），因此完整的interaction-centric graph利用每两个nodes之间的连接来作为图的edge。

#### Exploit Inter-interaction in Interaction-Centric Graph
回到本文开头提到的motivation：**交互间的结构性（Inter-interaction Structure）非常有助于互相提供线索来提高检测效果**，举个栗子，当图中有一个interaction为 human hold mouse，那么很有可能图中还有另一个相同human instance的interaction：human look-at screen。这个有趣的先验现象启发了作者构建一个graph来充分利用该prior的知识。作者定义了下图所示的六种交互间的关系来充分利用该先验：

![1666013372251.png](https://img-blog.csdnimg.cn/be9d77a500514e9bbc07e86f822996cc.png#pic_center)

这六种类间语义关系由两个交互proposal之间是否共享相同的human\object来被具体指派。

#### Exploit Intra-interaction in Interaction-Centric Graph
接着我们看本文开头提到的另一条motivation：**内部交互结构（Intra-interaction Structure）可以解释为每个HOI三元组的局部空间结构，例如人和物体的布局结构以一种额外的先验知识来将model的注意力引导到有效的图像区域。** STIP也通过分类、编码来利用interaction内的空间关系，如下图所示：

![1666013726929.png](https://img-blog.csdnimg.cn/58ec65e94b384e4db4ba5d0917587986.png#pic_center)
将背景、union、human、object、intersection分别进行转换编码，从而将spatial layout structures编码进features中，参与特征交互。

#### Structure-aware Self-attention & Structure-aware Cross-attention
Structure-aware Self-attention & Structure-aware Cross-attention和传统的self- attention基本类似，就不细讲了～其中值得注意的是，作者受相对位置编码的启发，将每个key $q_{j}$与其的 inter-interaction semantic dependency $E_{dep}(d_{ij})$结合：
$$e_{i j}^{\text {self }}=\frac{\left(\boldsymbol{W}_q \boldsymbol{q}_i\right)^T\left(\boldsymbol{W}_k \boldsymbol{q}_j+\boldsymbol{\psi}\left(\boldsymbol{q}_j, \boldsymbol{E}_{d e p}\left(d_{i j}\right)\right)\right)}{\sqrt{d_{k e y}}}$$
#### Training Objective
针对action的监督，也是利用folcal loss：
$$
L_{c l s}=\frac{1}{\sum_{i=1}^N \sum_{c=1}^C y_{i c}} \sum_{i=1}^N \sum_{c=1}^C F L\left(\hat{y}_{i c}, y_{i c}\right)
$$

### Experiments

![1666014520989.png](https://img-blog.csdnimg.cn/0fb8921d61af42b49a5c40784ecab5e7.png#pic_center)
可以看出，在VCOCO数据集上，STIP的性能非常强劲，比之前的IDN高了十几个点，HICO-DET上的性能也很强。
#### Ablation Study.

![1666014778935.png](https://img-blog.csdnimg.cn/e0d79b6702014e98b24be972967d2b15.png#pic_center)


从消融实验中可以看出， inter-interaction 和intra-interaction的相关module都非常涨点，

###  Conclusion and Discussion
STIP不同与以往的query-based 范式，将proposal set prediction拆开为两个stage，第一个stage产生高质量的queries，其中融合了丰富、多样的的prior features来充分利用背景知识，从而有了非常惊艳的性能效果。