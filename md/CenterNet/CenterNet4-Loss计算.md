# CenterNet之loss计算

## 1. 网络输出

论文中提供了三个用于目标检测的网络，都是基于编码解码的结构构建的。

1. ResNet18 + upsample + deformable convolution : COCO AP 28%/142FPS
2. DLA34 + upsample + deformable convolution :  COCO AP 37.4%/52FPS
3. Hourglass104: COCO AP 45.1%/1.4FPS

这三个网络中输出内容都是一样的，80个类别，2个预测的中心坐标，2个中心点的偏差。

```python
# heatmap 输出的tensor的通道个数是80，每个通道代表对应类别的heatmap
(hm): Sequential(
(0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 80, kernel_size=(1, 1), stride=(1, 1))
)
# wh 输出是中心坐标位置，通道数为2
(wh): Sequential(
(0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
)
# reg 输出的tensor通道个数为2，分别是w,h方向上的偏移量
(reg): Sequential(
(0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1))
)
```

## 2. 损失函数

## 2.1 heatmap loss

输入图像$I\in R^{W\times H\times 3}$, W为图像宽度，H为图像高度。网络输出的关键点热图heatmap为$\hat{Y}\in [0,1]^{\frac{W}{R}\times \frac{H}{R}\times C}$其中，R代表得到输出相对于原图的步长stride。C代表类别个数。

下面是CenterNet中核心loss公式：

$$
L_k=\frac{-1}{N}\sum_{xyc}\begin{cases}
(1-\hat{Y}_{xyc})^\alpha log(\hat{Y}_{xyc})& Y_{xyc}=1\\
(1-Y_{xyc})^\beta(\hat{Y}_{xyc})^\alpha log(1-\hat{Y}_{xyc})& otherwise
\end{cases}
$$

这个和Focal loss形式很相似，$\alpha$和$\beta$是超参数，N代表的是图像关键点个数。

- 在$Y_{xyc}=1$的时候，

对于易分样本来说，预测值$\hat{Y}_{xyc}$接近于1，$(1-\hat{Y}_{xyc})^\alpha$就是一个很小的值，这样loss就很小，起到了矫正作用。

对于难分样本来说，预测值$\hat{Y}_{xyc}$接近于0，$ (1-\hat{Y}_{xyc})^\alpha$就比较大，相当于加大了其训练的比重。

- otherwise的情况下：

![otherwise分为两个情况A和B](https://img-blog.csdnimg.cn/20200808103212439.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图是一个简单的示意，纵坐标是${Y}_{xyc}$，分为A区（距离中心点较近，但是值在0-1之间）和B区（距离中心点很远接近于0）。

**对于A区来说**，由于其周围是一个高斯核生成的中心，$Y_{xyc}$的值是从1慢慢变到0。

举个例子(CenterNet中默认$\alpha=2,\beta=4$)：

$Y_{xyc}=0.8$的情况下，

- 如果$\hat{Y}_{xyc}=0.99$，那么loss=$(1-0.8)^4(0.99)^2log(1-0.99)$,这就是一个很大的loss值。
- 如果$\hat{Y}_{xyc}=0.8$, 那么loss=$(1-0.8)^4(0.8)^2log(1-0.8)$, 这个loss就比较小。
- 如果$\hat{Y}_{xyc}=0.5$, 那么loss=$(1-0.8)^4(0.5)^2log(1-0.5)$, 

- 如果$\hat{Y}_{xyc}=0.99$，那么loss=$(1-0.5)^4(0.99)^2log(1-0.99)$,这就是一个很大的loss值。
- 如果$\hat{Y}_{xyc}=0.8$, 那么loss=$(1-0.5)^4(0.8)^2log(1-0.8)$, 这个loss就比较小。
- 如果$\hat{Y}_{xyc}=0.5$, 那么loss=$(1-0.5)^4(0.5)^2log(1-0.5)$, 

总结一下：为了防止预测值$\hat{Y}_{xyc}$过高接近于1，所以用$(\hat{Y}_{xyc})^\alpha$来惩罚Loss。而$(1-Y_{xyc})^\beta$这个参数距离中心越近，其值越小，这个权重是用来减轻惩罚力度。

**对于B区来说**，$\hat{Y}_{xyc}$的预测值理应是0，如果该值比较大比如为1，那么$(\hat{Y}_{xyc})^\alpha$作为权重会变大，惩罚力度也加大了。如果预测值接近于0，那么$(\hat{Y}_{xyc})^\alpha$会很小，让其损失比重减小。对于$(1-Y_{xyc})^\beta$来说，B区的值比较大，弱化了中心点周围其他负样本的损失比重。

## 2.2 offset loss















## 参考

https://zhuanlan.zhihu.com/p/66048276

http://xxx.itp.ac.cn/pdf/1904.07850

