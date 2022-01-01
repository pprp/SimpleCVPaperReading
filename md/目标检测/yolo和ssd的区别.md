# YOLOv3和SSD有什么区别

1. 尺度上的差异

YOLOv3仅仅在三个尺度上进行检测，SSD则是在五个尺度上，其中最小的尺度是1x1的。

2. 检测头的差异

YOLOv3是对feature map的Tensor的对应位置施加Loss，包括obj loss, cls loss, bbox loss；

SSD则是对Tensor分出来两个分支，一个分支做bbox loss, 一个分支做cls loss, 不存在obj loss，obj在SSD中相当于背景类，没有像YOLO中作为一个条件概率来进行学习。

3. Anchor设置

SSD的anchor设置和当前的feature map大小有关。

YOLO的anchor则是通过聚类得到的结果，对数据集有一定耦合，有强先验。

数量上：

- 300输入的SSD需要8000+的Anchor。
- 416输入的YOLOv3需要10000+的Anchor。

4. 输入分辨率

SSD貌似输入要求比较严格300x300的分辨率。

YOLOv3则需要输入分辨率满足是32的倍数即可，常见设置为416x416和608x608。

5. Backbone差异

SSD用的是修改过的VGG

YOLOv3用的是Darknet53

6. 数据增强策略





7. 训练策略

YOLOv3有一个多尺度训练



8. FPN

YOLOv3中用到了三层的FPN，对小目标效果提升了很多；

SSD中没有用到FPN，改进版FSSD主要改进就是加入FPN,不过只加了一个FPN，对小目标的效果依然很差。

