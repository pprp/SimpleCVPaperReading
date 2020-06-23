# pytorch版CenterNet训练自己的数据集

CenterNet(Objects as points)已经有一段时间了，之前这篇文章-[【目标检测Anchor-Free】CVPR 2019 Object as Points（CenterNet）](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484887&idx=1&sn=7367588eb0ba14a8da75f9e8f27af7fb&chksm=9f80bf41a8f73657ed7d82e654b330d64f2d1ca18ee33a21a297469ff04a2835ed023396ae10&scene=21#wechat_redirect)中讲解了CenterNet的原理，不熟悉的可以回顾一下。

这篇文章是基于非官方的CenterNet实现，https://github.com/zzzxxxttt/pytorch_simple_CenterNet_45，这个版本的实现更加简单，基于官方版本(https://github.com/xingyizhou/CenterNet)进行修改，要比官方代码更适合阅读和理解，dataloader、hourglass、训练流程等原版中比较复杂的部分都进行了重写，最终要比官方的速度更快。

这篇博文主要讲解如何用这个版本的CenterNet训练自己的VOC数据集，环境的配置。

## 1. 环境配置



