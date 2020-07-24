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

heatmap上使用高斯核有很多需要注意的细节。CenterNet官方版本实际上是在CornerNet的基础上改动得到的，有很多祖传代码。

在使用高斯核前要考虑这样一个问题，下图来自于CornerNet论文中的图示，红色的是标注框，但绿色的其实也可以作为最终的检测结果保留下来。那么这个问题可以转化为绿框在红框多大范围以内可以被接受。使用IOU来衡量红框和绿框的贴合程度，当两者IOU>0.7的时候，认为绿框也可以被接受，反之则不被接受。

![图源CornerNet](https://img-blog.csdnimg.cn/20200722102906603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

那么现在问题转化为，如何确定半径r, 让红框和绿框的IOU大于0.7。

![](https://img-blog.csdnimg.cn/20200721220135116.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

以上是三种情况，其中蓝框代表标注框，橙色代表可能满足要求的框。这个问题最终变为了一个一元二次方程有解的问题，同时由于半径必须为正数，所以r的取值就可以通过求根公式获得。

```python
def gaussian_radius(det_size, min_overlap=0.7):
    # gt框的长和宽
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    
    return min(r1, r2, r3)
```

可以看到这里的公式和上图计算的结果是一致的，需要说明的是，CornerNet最开始版本中这里出现了错误，分母不是2a，而是直接设置为2。CenterNet也延续了这个bug，CenterNet作者回应说这个bug对结果的影响不大，但是根据issue的讨论来看，有一些人通过修正这个bug以后，可以让AR提升1-3个百分点。以下是有bug的版本，CornerNet最新版中已经修复了这个bug。

```python
def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2

  return min(r1, r2, r3)
```

同时有一些人认为圆并不普适，提出了使用椭圆来进行计算，也有人在issue中给出了推导，感兴趣的可以看以下链接：https://github.com/princeton-vl/CornerNet/issues/110





```python
import numpy as np
y,x = np.ogrid[-4:5,-3:4]
sigma = 1
h=np.exp(-(x*x+y*y)/(2*sigma*sigma))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x,y,h)
plt.show()
```







## 参考

https://zhuanlan.zhihu.com/p/66048276

https://www.cnblogs.com/shine-lee/p/9671253.html

https://zhuanlan.zhihu.com/p/96856635

http://xxx.itp.ac.cn/pdf/1808.01244

https://github.com/princeton-vl/CornerNet/issues/110