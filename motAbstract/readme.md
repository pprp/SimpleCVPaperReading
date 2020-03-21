# MOT Abstract

[TOC]

## 2008

Title:BraMBLe:A BayesianMultiple-BlobTracker

Link:https://link.zhihu.com/?target=http%3A//users.dickinson.edu/~jmac/publications/bramble.pdf

Abstract:

Blob跟踪器近些年变得越来越流行很大程度上是由于使用了统计学的表观模型，可以使用有效的背景相减和鲁棒地跟踪可变化前景目标。这种处理方法成为了一种标准，但是这种方法是将前景和背景当做两个互相独立的过程。背景相减以后就是块检测和跟踪，这样违背了图片中可能性的原则性计算。

本文提出了两个理论上的改进，一个是仔细讨论跟踪系统的局限性，一个是提出了一个适合于多摄像头、实时监视的鲁棒的多行人跟踪。

第一个创新点是一个多块可能性公式，直接将可比较的可能性分配给包含不同数量目标的假设。这个可能性公式基于一个充满活力的基础，他改编自贝叶斯相关性理论，但是使用了一个前提就是

