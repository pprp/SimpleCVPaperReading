# 【神经网络搜索】Once for all





## 4. 实验结果

**数据集：** 在ImageNet上完成，将原先的训练集切分成50000张作为验证集，剩下的作为训练集，原先的验证集作为测试集。

**搜索空间：block** 基于ShuffleNetv2设计的搜索空间，具体采用的架构如下，总共有20个CB(choice block)，每个choice block 可以选择四个block，分别是kernel=3、5、7的shufflenet Units和一个Xception的block组成。

![超网的架构](https://img-blog.csdnimg.cn/20210413200846700.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

**初步baseline:**

![](https://img-blog.csdnimg.cn/20210413202041843.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

其baseline是所有的choice block中都选择相同的选择，比如3x3的shufflenet Units，得到的top1准确率都差不太多；从搜索空间中随机采样一些候选网络，得到的结果虽然一样，但是作者认为这是由于随机搜索方法太过简单，以至于不能从大型的搜索空间找到好的候选网络；使用进化算法进行搜索，得到的结果是74.3，比所有的的baeline模型都高。

![block+channel](https://img-blog.csdnimg.cn/20210413203516962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

可以看出，同时搜索block和channel的结果性能更高，超过了其他同类型方法。

![搜索代价对比](https://img-blog.csdnimg.cn/20210413204013785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



**一致性分析：** 基于**超网模型的表现**和**独立训练模型的表现**之间的相关性越强代表NAS算法有效性更强。SPOS对相关性进行了分析来测试SPOS的有效性，使用NAS-Bench-201这个benchmark来分析SPOS的有效性。NASBench201是cell-based搜索空间，搜索空间中包含5个可选操作zero, skip connection,1x1卷积，3x3卷积，3x3 average pooling。基于这个进一步设计了一些缩小的搜索空间，Reduce-1代表删除了1x1卷积、Reduce-2代表删除了3x3 average pooling， Reduce-3代表删除了以上两者。使用的是kendell Tau来计算相关性。

![相关性](https://img-blog.csdnimg.cn/20210413205623591.png)

通过以上实验可以看出，SPOS有一定的局限性：SPOS的超网是部分相关的，无法实现完美的真实排序。搜索空间越小，其相关性越强。



## 结论

Sinlge Path One Shot分析了以往的基于权重共享的NAS方法中存在的权重耦合问题，并提出了单路径训练策略来缓解耦合问题。本文还分析了SPOS的搜索代价和排序一致性问题，还指出了算法的局限在于超网的排序一致性是部分关联的，并不能完美的符合真实排序。搜索空间越小，排序一致性越强。