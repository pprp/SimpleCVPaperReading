# 【CV中的Attention机制】Non-Local改进CCNet

> 前言：Non-Local提出以后，有很多人提出了其改进版本用来解决其计算量大的问题。CCNet(ICCV2019)是解决在语义分割领域中Non-Local计算量过大的问题，并提出了一个解决方案：通过堆叠两个十字注意力模块，来捕捉长距离的、像素级别语义信息。之后在语义分割benchmarks中进行试验，达到了SOTA。

