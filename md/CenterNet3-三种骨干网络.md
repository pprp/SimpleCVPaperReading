# CenterNet骨干网络之hourglass

CenterNet中主要提供了三个骨干网络ResNet-18(ResNet-101), DLA-34, Hourglass-104，本文从结构和代码先对hourglass进行讲解。

本文对应代码位置在：https://github.com/pprp/SimpleCVReproduction/tree/master/Simple_CenterNet

## 1. Ground Truth Heatmap

在开始讲解骨干网络之前，先提一下上一篇文章中有朋友问我的问题：**CenterNet为什么要沿用CornerNet的半径计算方式？**

查询了CenterNet论文还有官方实现的issue，其实没有明确指出为何要用CornerNet的半径，issue中回复也说是这是沿用了CornerNet的祖传代码。经过和@tangsipeng的讨论，讨论结果如下：

以下代码是涉及到半径计算的部分：

```python
# 根据一元二次方程计算出最小的半径
radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
# 得到高斯分布
draw_umich_gaussian(hmap[label], obj_c_int, radius)
```

在centerNet中，半径的存在主要是用于计算高斯分布的sigma值，而这个值也是一个经验性判定结果。

```python
def draw_umich_gaussian(heatmap, center, radius, k=1):
    # 得到直径
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # 对边界进行约束，防止越界
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    # 选择对应区域
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # 将高斯分布结果约束在边界内
    masked_gaussian = gaussian[radius - top:radius + bottom, 
                               radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，相当于不断的在heatmap基础上添加关键点的高斯，
        # 即同一种类型的框会在一个heatmap某一个类别通道上面上面不断添加。
        # 最终通过函数总体的for循环，相当于不断将目标画到heatmap
    return heatmap
```

合理推测一下（不喜勿喷），之前很多人在知乎上issue里讨论这个半径计算的时候，有提到这样的问题，就是如果将CenterNet对应的2a改正确了，反而效果会差。

我觉得这个问题可能和这里的`sigma=diameter / 6`有一定的关系，作者当时用祖传代码（2a那部分有错）进行调参，然后确定了sigma。这时这个sigma就和祖传代码是对应的，如果修改了祖传代码，同样也需要改一下sigma或者调一下参数。

tangsipeng同学分享的文章《Training-Time-Friendly Network for Real-Time Object Detection》对应计算高斯核sigma部分就没有用cornernet的祖传代码，对应代码可以发现，这里的sigma是一个和h,w相关的超参数，也是手工挑选的。 

![tangsipeng同学提供的截图](https://img-blog.csdnimg.cn/20200730115008204.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

综上，目前暂时认为CenterNet直接沿用CornerNet的祖传代码没有官方的解释，我们也暂时没有想到解释。如果对这个问题有研究的同学欢迎联系笔者。

## 2. Hourglass

Hourglass网络结构最初是在ECCV2016的Stacked hourglass networks for human pose estimation文章中提出的，用于人体姿态估计。Stacked Hourglass就是把多个漏斗形状的网络级联起来，可以获取多尺度的信息。

Hourglass的设计比较有层次，通过各个模块的有规律组合成完整网络。

### 2.1 Residual模块

```python
class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim,
                               out_dim, (3, 3),
                               padding=(1, 1),
                               stride=(stride, stride),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim,
                               out_dim, (3, 3),
                               padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                                  nn.BatchNorm2d(out_dim)) \
            if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)
```

就是简单的残差链接网络中的最基础的残差模块。

### 2.2 Hourglass子模块

```python
class kp_module(nn.Module):
    '''
    kp module指的是hourglass基本模块
    '''
    def __init__(self, n, dims, modules):
        super(kp_module, self).__init__()

        self.n = n

        curr_modules = modules[0]
        next_modules = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # curr_mod x residual，curr_dim -> curr_dim -> ... -> curr_dim
        self.top = make_layer(3, # 空间分辨率不变
                              curr_dim,
                              curr_dim,
                              curr_modules,
                              layer=residual)
        self.down = nn.Sequential() # 暂时没用
        # curr_mod x residual，curr_dim -> next_dim -> ... -> next_dim
        self.low1 = make_layer(3,
                               curr_dim,
                               next_dim,
                               curr_modules,
                               layer=residual,
                               stride=2)# 降维
        # next_mod x residual，next_dim -> next_dim -> ... -> next_dim
        if self.n > 1:
            # 通过递归完成构建
            self.low2 = kp_module(n - 1, dims[1:], modules[1:])
        else:
            # 递归出口
            self.low2 = make_layer(3,
                                   next_dim,
                                   next_dim,
                                   next_modules,
                                   layer=residual)
        # curr_mod x residual，next_dim -> next_dim -> ... -> next_dim -> curr_dim
        self.low3 = make_layer_revr(3, # 升维
                                    next_dim,
                                    curr_dim,
                                    curr_modules,
                                    layer=residual)
        self.up = nn.Upsample(scale_factor=2) # 上采样进行升维

    def forward(self, x):
        up1 = self.top(x)
        down = self.down(x)
        low1 = self.low1(down)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        return up1 + up2
```

其中有两个主要的函数`make_layer`和`make_layer_revr`，`make_layer`将空间分辨率降维，`make_layer_revr`函数进行升维，所以将这个结构命名为hourglass(沙漏)。

核心构建是一个递归函数，递归层数是通过n来控制，称之为n阶hourglass模块。

![论文中的n阶hourglass模块示意图](https://img-blog.csdnimg.cn/20200730150621168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

### 2.3 Hourglass

```python
class exkp(nn.Module):
    '''
     整体模型调用
     large hourglass stack为2
     small hourglass stack为1
     n这里控制的是hourglass的阶数，以上两个都用的是5阶的hourglass
     exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4]),
    '''
    def __init__(self, n, nstack, dims, modules, cnv_dim=256, num_classes=80):
        super(exkp, self).__init__()

        self.nstack = nstack # 堆叠多次hourglass
        self.num_classes = num_classes

        curr_dim = dims[0]

        # 快速降维为原来的1/4
        self.pre = nn.Sequential(convolution(7, 3, 128, stride=2),
                                 residual(3, 128, curr_dim, stride=2))

        # 堆叠nstack个hourglass
        self.kps = nn.ModuleList(
            [kp_module(n, dims, modules) for _ in range(nstack)])

        self.cnvs = nn.ModuleList(
            [convolution(3, curr_dim, cnv_dim) for _ in range(nstack)])

        self.inters = nn.ModuleList(
            [residual(3, curr_dim, curr_dim) for _ in range(nstack - 1)])

        self.inters_ = nn.ModuleList([
            nn.Sequential(nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                          nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)
        ])
        self.cnvs_ = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                          nn.BatchNorm2d(curr_dim)) for _ in range(nstack - 1)
        ])
        # heatmap layers
        self.hmap = nn.ModuleList([
            make_kp_layer(cnv_dim, curr_dim, num_classes) # heatmap输出通道为num_classes
            for _ in range(nstack)
        ])
        for hmap in self.hmap:
            # -2.19是focal loss中的默认参数，论文的4.1节有详细说明，-ln((1-pi)/pi),这里的pi取0.1
            hmap[-1].bias.data.fill_(-2.19)

        # regression layers
        self.regs = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]) # 回归的输出通道为2
        self.w_h_ = nn.ModuleList(
            [make_kp_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)]) # wh

        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        inter = self.pre(image)

        outs = []
        for ind in range(self.nstack): # 堆叠两次hourglass
            kp = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if self.training or ind == self.nstack - 1:
                outs.append([
                    self.hmap[ind](cnv), self.regs[ind](cnv),
                    self.w_h_[ind](cnv)
                ])

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs
```

这里需要注意的是inters变量，这个变量保存的是中间监督过程，可以在这个位置添加loss，具体如下图蓝色部分所示，在这个部分可以添加loss，然后再用1x1卷积重新映射到对应的通道个数并相加。

![论文中提供的对inter层的图示](https://img-blog.csdnimg.cn/20200730151024925.png)

然后再来谈三个输出，假设当前是COCO数据集，类别个数为80，那么hmap相当于输出了通道个数为80的heatmap，每个通道负责预测一个类别；wh代表对应中心点的宽和高；regs是偏置量。

CenterNet论文详解可以点击[【目标检测Anchor-Free】CVPR 2019 Object as Points（CenterNet）](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247484887&idx=1&sn=7367588eb0ba14a8da75f9e8f27af7fb&chksm=9f80bf41a8f73657ed7d82e654b330d64f2d1ca18ee33a21a297469ff04a2835ed023396ae10&scene=21#wechat_redirect)

整个网络就梳理完成了，笔者简单画了一下nstack为2时的hourglass网络，如下图所示：

![nstack为2时的hourglass网络](https://img-blog.csdnimg.cn/20200730153137608.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 3. Reference

https://blog.csdn.net/shenxiaolu1984/article/details/51428392

http://xxx.itp.ac.cn/pdf/1603.06937.pdf

http://xxx.itp.ac.cn/pdf/1904.07850v1