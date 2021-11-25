# Towards Oracle Knowledge Distillation with NAS

【GiantPandaCV导语】本文介绍的如何更好地集成教师网络，从而更好地提取知识到学生网络，提升学生网络的学习能力和学习效率。从方法上来讲是模型集成+神经网络结构搜索+知识蒸馏的综合问题，在这里使用简单的NAS来降低教师网络与学生网络之间的差距。


## 背景介绍

解决的问题？

- 希望从集成的教师网络中提取知识到学生网络，从而提升学习能力和学习效率。

- model ensemble + NAS + KD

- Motivation: This is motivated by the fact that knowledge distillation is less effective when the capacity gap (e.g., the number of parameters) between teacher and student is large as discussed in (Mirzadeh et al. 2019).

如何解决？

- 提出了OD（Oracle Knowledge Distillation）的方法，我们的方法解决了教师和学生之间固有的模型能力问题，旨在通过缩小教师模型的能力差距，使其在蒸馏过程中受益最大化。

- 使用NAS技术来增强有用的架构和操作，这里搜索的网络适用于蒸馏学生网络。

- 提出了Oracle KD Loss来实施模型搜索，同时使用集成的教师网络进行蒸馏。


具体如何组织集成教师网络？搜索对象是教师网络？如何动态处理模型capacity?

- 灵感：动态的组织整个过程的学习，教师网络容量大，学生网络容量小，可以让容量大的表征迁移到容量小的模型。

- 搜索对象是学生网络，学生网络是可以动态变化的，从而可以适应教师网络的容量。而教师网络在整个过程中是不进行训练的。



## Oracle KD Loss

![](https://img-blog.csdnimg.cn/0154055d20334f94b261dae9038365d5.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)

Independent Ensemble(IE) 将网络集成的方式有：

- Simple model average: 在不同的seed下训练同一个网络，并将最终的logits平均起来。

- Majority voting: 投票法

Oracle KD认为这种将所有的教师网络都利用的方法并不一定合适，比如可能模型的子集可以预测正确模型，如上图所示，就可以选择子集的模型进行预测，所以需要实现一个**模型选择过程** 。

提出Oracle KD Loss来提升集成教师的性能。

![](https://img-blog.csdnimg.cn/7f66de91f8de49bf9a3533be835b11e7.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)

上述公式表达意思是：u是一个指标，判断该模型是否正确分类，如果正确分类，那就将其作为教师网络进行蒸馏，否则使用普通的CrossEntropy来学习。


## Optimal Model Search for KD

为解决教师网络和学生网络之间存在的容量不匹配问题，提出了Knowledge Distillation framework with Architecture Search (KDAS)策略。

这里使用NAS只是在原有backbone上进行微调，并不是从头开始选择模型（搜索对象要比backbone略大一些）。

![](https://img-blog.csdnimg.cn/a2b15156fe344bf19b7e7f78357cc4e9.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)

搜索空间设计：

- identity operation

- 3x3 conv

- 5x5 conv

- 3x3 深度可分离conv

- 5x5 深度可分离conv

- maxpool

- avgpool


**优化方法：** 

使用REINFORCE强化学习算法结合LSTM控制器采样网络，动态控制子网的容量。

![](https://img-blog.csdnimg.cn/47731a24650e4d1ea7ef3026935ca040.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)


## 实验结果

使用ResNet32x5作为教师网络，使用ResNet32作为学生网络，

![](https://img-blog.csdnimg.cn/5a72af22d6e1443887a3895e94e914de.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)


这个图比较的是memory-accuracy的trade off：

![](https://img-blog.csdnimg.cn/446fa038a3c64df8819c8f233b5fca0d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)

