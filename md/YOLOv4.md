# YOLOv4论文解读

> 当YOLOv3作者说明退出CV圈的时候，大家都很失落。
>
> 结果昨天群友在arxiv上发现了YOLOv4, 不是别人，作者是AlexeyAB大佬。
>
> 我们很多人见证着AB版Darknet的不断更新，优化，不断跟随目标检测最新的成果(如ASFF、Swish、Mixup等)融合到Darknet框架中，终于YOLOv4来了。

AlexeyAB将其命名为YOLOv4确实得到YOLOv3作者Joseph Redmon的许可了，下面是Darknet原作者的在readme中更新的声明。

![Darknet原作者pjreddie在readme中承认了YOLOv4](https://img-blog.csdnimg.cn/20200424101249538.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 1. 介绍

![FPS vs AP](https://img-blog.csdnimg.cn/2020042410252679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

