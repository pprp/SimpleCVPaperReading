# CenterNet测试推理过程



## 1. eval部分数据加载

由于CenterNet是生成了一个heatmap进行的目标检测，而不是传统的基于anchor的方法，所以训练时候的数据加载和测试时的数据加载结果是不同的。并且在测试的过程中使用到了Test Time Augmentation（TTA），使用到了多尺度测试，翻转等