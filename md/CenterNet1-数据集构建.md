# CenterNet 数据加载解析

主要解读CenterNet如何加载数据，并将标注信息转化为CenterNet规定的高斯分布的形式。

## 1. YOLOv3和CenterNet流程对比

CenterNet和Anchor-Based的方法不同，以YOLOv3为例，大致梳理一下模型的框架和数据处理流程。

YOLOv3是一个经典的单阶段的目标检测算法，图片进入网络的流程如下：

- 对图片进行resize，长和宽都要是32的倍数。
- 图片经过网络的特征提取后，空间分辨率变为原来的1/32。
- 得到的Tensor去代表图片不同尺度下的目标框，其中目标框的表示为(x,y,w,h,c)，分别代表左上角坐标，宽和高，含有某物体的置信度。
- 训练完成后，测试的时候需要使用非极大抑制算法得到最终的目标框。

CenterNet是一个经典的Anchor-Free目标检测方法，图片进入网络流程如下：

- 对图片进行resize，长和宽一般相等，并且至少为4的倍数。
- 图片经过网络的特征提取后，得到的特征图的空间分辨率依然比较大，是原来的1/4。这是因为CenterNet采用的是类似人体姿态估计中用到的骨干网络，基于heatmap提取关键点的方法需要最终的空间分辨率比较大。
- 训练的过程中，CenterNet得到的是一个heatmap，所以标签加载的时候，需要转为类似的heatmap热图。
- 测试的过程中，由于只需要从热图中提取目标，这样就不需要使用NMS，降低了计算量。



## 2. CenterNet部分详解

设输入图片为$I\in R^{W\times H\times 3}$, W代表图片的宽，H代表高。CenterNet的输出是一个关键点热图heatmap。
$$
\hat{Y}\in[0,1]^{\frac{W}{R}\times\frac{H}{R}\times C}
$$
其中R代表输出的stride大小，C代表关键点的类型的个数。

举个例子，在COCO数据集目标检测中，R设置为4，C的值为80，代表80个类别。

如果$\hat{Y}_{x,y,c}=1$代表检测到一个物体，表示对类别c来说，(x,y)这个位置检测到了c类的目标。

既然输出是热图，标签构建的ground truth也必须是热图的形式。标注的内容一般包含（x1,y1,x2,y2,c）,目标框左上角坐标、右下角坐标和类别c，按照以下流程转为ground truth：

- 得到原图中对应的中心坐标$p=(\frac{x1+x2}{2}, \frac{y1+y2}{2})$
- 得到下采样后的feature map中对应的中心坐标$\tilde{p}=\lfloor \frac{p}{R}\rfloor$, R代表下采样倍数，CenterNet中R为4
- 如果输入图片为512，那么输出的feature map的空间分辨率为[128x128], 将标注的目标框以高斯核的方式将关键点分布到特征图上：

$$
Y_{xyc}=exp(-\frac{(x-\tilde p_x)^2+(y-\tilde p_y)^2}{2\sigma ^2_p})
$$

其中$\sigma_p$是一个与目标大小相关的标准差。对于特殊情况，相同类别的两个高斯分布发生了重叠，重叠元素间最大的值作为最终元素。下图是

![图源知乎@OLDPAN](https://img-blog.csdnimg.cn/20200721090749730.png)





## 3. heatmap上应用高斯核

heatmap

![](https://img-blog.csdnimg.cn/20200721220135116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)









## 参考

https://zhuanlan.zhihu.com/p/66048276

https://www.cnblogs.com/shine-lee/p/9671253.html

https://zhuanlan.zhihu.com/p/96856635