# CenterNet中的骨干网络

CenterNet中主要提供了三个骨干网络ResNet-18(ResNet-101), DLA-34, Hourglass-104，本文从结构和代码对这三个骨干网络。

## 1. Hourglass

Hourglass网络结构最初是在ECCV2016的Stacked hourglass networks for human pose estimation文章中提出的，用于人体姿态估计。Stacked Hourglass就是把多个漏斗形状的网络级联起来，可以获取多尺度的信息。

