# 【从零开始学习YOLOv3】8. YOLOv3中Loss部分计算

YOLOv1是一个anchor-free的，从YOLOv2开始引入了Anchor，在VOC2007数据集上将mAP提升了10个百分点。YOLOv3也继续使用了Anchor，本文主要讲U版YOLOv3的Loss部分的计算。

### 1. Anchor

Faster R-CNN中Anchor的大小和比例是由人手工设计的，可能并不贴合数据集，有可能会给模型性能带来负面影响。YOLOv2和YOLOv3则是通过聚类算法得到最适合的k个框。聚类距离是通过IoU来定义，IoU越大，边框距离越近。
$$
d(box,centroid)=1-IoU(box,centroid)
$$
Anchor越多，平均IoU会越大，效果越好，但是会带来计算量上的负担，下图是YOLOv2论文中的聚类数量和平均IoU的关系图，在YOLOv2中选择了5个anchor作为精度和速度的平衡。

![YOLOv2中聚类Anchor数量和IoU的关系图](https://img-blog.csdnimg.cn/20200326152932491.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 2. 偏移公式

在Faster RCNN中，中心坐标的偏移公式是：

$$
\left\{
\begin{aligned}
x=&(t_x\times w_a)+x_a\\
y=&(t_y\times h_a)+y_a
\end{aligned}
\right.
$$

其中$x_a$、$y_a$ 代表中心坐标，$w_a$和$h_a$代表宽和高，$t_x$和$t_y$是模型预测的Anchor相对于Ground Truth的偏移量，通过计算得到的x,y就是最终预测框的中心坐标。

而在YOLOv2和YOLOv3中，对偏移量进行了限制，如果不限制偏移量，那么边框的中心可以在图像任何位置，可能导致训练的不稳定。
$$
\left\{
\begin{aligned}
b_x&=\sigma(t_x)+c_x\\
b_y&=\sigma(t_y)+c_y\\
b_w&=p_we^{t_w}\\
b_h&=p_he^{t_h}\\
\sigma(t_o)&=Pr(object)\times IOU(b,object)
\end{aligned}
\right.
$$

![公式对应的意义](https://img-blog.csdnimg.cn/20200326165301453.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

对照上图进行理解：

- $c_x$和$c_y$分别代表中心点所处区域的左上角坐标。
- $p_w$和$p_h$分别代表Anchor的宽和高。
- $\sigma(t_x)$和$\sigma(t_y)$分别代表预测框中心点和左上角的距离，$\sigma$代表sigmoid函数，将偏移量限制在当前grid中，有利于模型收敛。
- $t_w$和$t_h$代表预测的宽高偏移量，Anchor的宽和高乘上指数化后的宽高，对Anchor的长宽进行调整。

- $\sigma(t_o)$是置信度预测值，是当前框有目标的概率乘以bounding box和ground truth的IoU的结果

## 3. Loss

YOLOv3中有一个参数是ignore_thresh，在U版的YOLOv3中对应的是train.py文件中的`iou_t`参数（默认为0.225）。

**正负样本是按照以下规则决定的**：

- 如果一个预测框与所有的Ground Truth的最大IoU<ignore_thresh时，那这个预测框就是**负样本**。

- 如果Ground Truth的中心点落在一个区域中，该区域就负责检测该物体。将与该物体有最大IoU的预测框作为**正样本**（注意这里没有用到ignore thresh,即使该最大IoU<ignore thresh也不会影响该预测狂为正样本）

在YOLOv3中，Loss分为三个部分:

- 一个是xywh部分带来的误差，也就是bbox带来的loss
- 一个是置信度带来的误差，也就是obj带来的loss
- 最后一个是类别带来的误差，也就是class带来的loss

在代码中分别对应lbox, lobj, lcls。







