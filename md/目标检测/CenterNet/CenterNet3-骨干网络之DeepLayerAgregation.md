# CenterNet的骨干网络之DLASeg

DLA全称是Deep Layer Aggregation, 于2018年发表于CVPR。被CenterNet, FairMOT等框架所采用，其效果很不错，准确率和模型复杂度平衡的也比较好。

CenterNet中使用的DLASeg是在DLA-34的基础上添加了Deformable Convolution后的分割网络。

## 1. 简介

Aggretation聚合是目前设计网络结构的常用的一种技术。如何将不同深度，将不同stage、block之间的信息进行融合是本文探索的目标。

目前常见的聚合方式有skip connection, 如ResNet，这种融合方式仅限于块内部，并且融合方式仅限于简单的叠加。

本文提出了DLA的结构，能够迭代式地将网络结构的特征信息融合起来，让模型有更高的精度和更少的参数。

![DLA的设计思路](https://img-blog.csdnimg.cn/20200804202908321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

上图展示了DLA的设计思路，Dense Connections来自DenseNet，可以聚合语义信息。Feature Pyramids空间特征金字塔可以聚合空间信息。DLA则是将两者更好地结合起来从而可以更好的获取what和where的信息。仔细看一下DLA的其中一个模块，如下图所示：

![DLA其中一个Tree结构](https://img-blog.csdnimg.cn/20200804203952451.png)

研读过代码以后，可以看出这个花里胡哨的结构其实是按照树的结构进行组织的，红框框住的就是两个树，树之间又采用了类似ResNet的残差链接结构。

## 2. 核心

先来重新梳理一下上边提到的语义信息和空间信息，文章给出了详细解释：

- 语义融合：在通道方向进行的聚合，能够提高模型推断“是什么”的能力（what）
- 空间融合：在分辨率和尺度方向的融合，能够提高模型推断“在哪里”的能力（where）

![DLA34完整结构图](https://img-blog.csdnimg.cn/20200804205203420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

Deep Layer Aggregation核心模块有两个IDA(Iterative Deep Aggregation)和HDA(Hierarchical Deep Aggregation)，如上图所示。

- 红色框代表的是用树结构链接的层次结构，能够更好地传播特征和梯度。

- 黄色链接代表的是IDA，负责链接相邻两个stage的特征让深层和浅层的表达能更好地融合。
- 蓝色连线代表进行了下采样，网络一开始也和ResNet一样进行了快速下采样。

论文中也给了公式推导，感兴趣的可以去理解一下。本文还是将重点放在代码实现上。

## 3. 实现

这部分代码复制自CenterNet官方实现，https://github.com/pprp/SimpleCVReproduction/blob/master/CenterNet/nets/dla34.py

### 3.1 基础模块

首先是三个模块，BasicBlock和Bottleneck和ResNet中的一致，BottleneckX实际上是ResNeXt中的基础模块，也可以作为DLA中的基础模块。DLA34中调用的依然是BasicBlock。

```python
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out
```

### 3.2 Root类

然后就是Root类，对应下图中的绿色模块

![Root类对应图示](https://img-blog.csdnimg.cn/20200804211100782.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

所有的Aggregation Node都是通过调用这个模块完成的，这个绿色结点也是其连接两个树的根，所以形象地称之为Root。下面是代码实现，forward函数中接受的是多个对象，用来聚合多个层的信息。

```python
class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        # 输入是多个层输出结果
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x
```

### 3.3 Tree类

Tree类对应图中的HDA模块，是最核心最复杂的地方，建议手动画一下。其核心就是递归调用的Tree类的构建，以下是代码。

```python
class Tree(nn.Module):
    '''
    self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                    level_root=True, root_residual=residual_root)
    '''
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children

        bottom = self.downsample(x) if self.downsample else x
        # project就是映射，如果输入输出通道数不同则将输入通道数映射到输出通道数
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            # root是出口
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x
```

经过笔者研究，这里涉及了两个比较重要的参数level和level root。

这个类有两个重要的成员变量tree1和tree2，是通过递归的方式迭代生成的，迭代层数通过level进行控制的，举两个例子，第一个是level为1，并且level root=True的情况，对照代码和下图可以理解得到：

![](https://img-blog.csdnimg.cn/20200804213702844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

也就是对应的是：

![](https://img-blog.csdnimg.cn/20200804214219525.png)

代码中的children参数是一个list，保存的是所有传给Root的成员，这些成员将作为其中的叶子结点。

第二个例子是level=2， level root=True的情况，如下图所示：

![](https://img-blog.csdnimg.cn/20200804213727550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这部分代码对应的是：

![](https://img-blog.csdnimg.cn/20200804214530683.png)

粉色箭头是children对象，都交给Root进行聚合操作。

### 3.4 DLA

Tree是DLA最重要的模块，Tree搞定之后，DLA就按顺序拼装即可。

```python
class DLA(nn.Module):
    '''
    DLA([1, 1, 1, 2, 2, 1],
        [16, 32, 64, 128, 256, 512],
        block=BasicBlock, **kwargs)
    '''
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        # 在最初前两层仅仅使用卷积层
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        '''
        if level_root:
            root_dim += in_channels
        '''
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            # 将几个level串联起来
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            return x
```

## 4. DLASeg

DLASeg是在DLA的基础上使用Deformable Convolution和Upsample层组合进行信息提取，提升了空间分辨率。

```python
class DLASeg(nn.Module):
    '''
    DLASeg('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
    '''
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        # globals() 函数会以字典类型返回当前位置的全部全局变量。
        # 所以这个base就相当于原来的DLA34
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # first_level = 2 if down_ratio=4
        # channels = [16, 32, 64, 128, 256, 512] to [64, 128, 256, 512]
        # scales = [1, 2, 4, 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        # 进行上采样
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(channels[self.first_level], head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
              if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(channels[self.first_level], classes, 
                  kernel_size=final_kernel, stride=1, 
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]
```

以上就是DLASeg的主要代码，其中负责上采样部分的是：

```python
 self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
[2 ** i for i in range(self.last_level - self.first_level)])
```

这部分负责解码，将空间分辨率提高。

```python
class IDAUp(nn.Module):
    '''
    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j])
    ida(layers, len(layers) -i - 2, len(layers))
    '''
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))

            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])
```

其核心是DLAUP和IDAUP, 这两个类中都使用了两个Deformable Convolution可变形卷积，然后使用ConvTranspose2d进行上采样，具体网络结构如下图所示。

![DLASeg结构图](https://img-blog.csdnimg.cn/20200805203836322.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 5. Reference

https://arxiv.org/abs/1707.06484

https://github.com/pprp/SimpleCVReproduction/blob/master/CenterNet/nets/dla34.py