# 【CV中的Attention机制】DCANet解读

【GiantPandaCV导读】DCANet与其他文章不同之处在于，DCANet用增强其他Attention模块能力的方式来改进的，可以让注意力模块之间的信息流动更加充分，提升注意力学习的能力。目前文章还没有被接收。

本文首发于GiantPandaCV，未经允许，不得转载。

![](https://img-blog.csdnimg.cn/20210218110600518.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 1、摘要

自注意力机制在很多视觉任务中效果明显，但是这些注意力机制往往都是考虑当前的特征，而没有考虑和其他层特征进行融合（其实也有几个工作都在做融合的Attention比如BiSeNet、AFF等）。

本文提出的DCANet（Deep Connected Attention Network）就是用来提升attention模块能力的。主要做法是：将相邻的Attention Block互相连接，让信息在Attention模块之间流动。

## 2、思想

**自注意力机制：** 自注意力机制可以通过探索特征之间的依赖关系来得到更好的特征表示。自注意力机制在NLP和CV领域中的各个任务都得到了广泛应用。















## 总结



笔者维护了一个有关Attention机制和其他即插即用模块的库，欢迎在Github上进行PR或者Issue。

https://github.com/pprp/awesome-attention-mechanism-in-cv



