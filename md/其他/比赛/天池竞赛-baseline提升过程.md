# 天池竞赛-布匹缺陷检测baseline提升过程-给yolov5模型添加注意力机制

这次比赛选择了官方提供的baseline yolov5进行训练，一开始使用的是yolov5s.yml配置文件进行训练的，并且数据也只是train2一小部分，由于笔者这里服务器只有一个1080Ti的可以使用，所以实验跑起来速度还是有点慢的，做的尝试也不是很多，下面是流水账。

![第一次成功提交](https://img-blog.csdnimg.cn/20210225174946141.png)

第一次提交就是使用了train2部分数据集，设置了50个epoch，使用迁移学习，分辨率设置为500x500,花费大概2个小时训练完成。这个成绩的acc还不错，是因为conf thresh设置的值比较低，所以acc可以达到比较高的结果。但是mAP就很差，一方面是数据量不足导致的，另一方面是模型容量比较小。

![第二个比较不错的结果](https://img-blog.csdnimg.cn/20210225175352107.png)

之后开始将train1部分的数据加进来，增大epoch个数到100，模型使用更大的yolov5x.yml,分辨率也提高到1000x1000，虽然有所提高，但是提高并不多。值得一提的是数据直接通过wget在linux中下载，并解压会出现错误，使用了论坛提供的tar方法也没有很好的解决。window上测试解压效果就很好，图片都没有损坏，不知道具体原因。但是如果从我本地window上传到服务器上，速度慢的简直不可忍受，所以就放弃了上传。采用了那些没有损坏的图片进行训练，尽管失去了一部分数据集，数据量还是很大的，训练yolov5x一般需要12-24个小时，时间比较久。

后边怀疑可能是yolov5自带的mosic数据增强方法有问题，因为它会将四张图片组成一个进行训练，比较长的目标会有所损耗，所以关闭了这个数据增强方法。经过很长时间的训练，发现mosic还是有效果的，去掉了应该会掉点。

![去掉了mosic数据增强方法](https://img-blog.csdnimg.cn/20210225180028450.png)

后边时间就到现在了，期间研究了一下yolov5的模型组织方式。因为之前笔者曾经用过yolov3, 那时候的数据组织方式是cfg文件，比较容易理解，但是也比较难改。在yolov5中使用了yaml文件进行组织，重复的模块可以通过number设置即可，降低了构建的难度。yolov5中也提供了多种多样的新模块，比如：CSP模块、SPP模块、GhostBottleneck模块、MixConv2d模块、CrossConv模块等等，这都是比较新的文章中提到的，方便进行实验。

因为笔者之前研究过attention机制，也成功在yolov3中添加过attention模块，带来了一定的收益。所以之后的改进思路是添加SELayer，这个注意力模块的鼻祖。一般来说注意力模块作用是：增加模型的远距离依赖、增加模型复杂度、提高准确率（不绝对）等作用。这次也想在yolov5中研究添加SE的方法，这里做一个笔记总结。实验还在跑，后边会补充结果。

先讲一下配置文件：以yolov5x.yaml为例：

```yaml
# parameters
nc: 15  # number of classes
depth_multiple: 1.33  # model depth multiple
width_multiple: 1.25  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2                 #1
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4              #2
   [-1, 3, C3, [128]],                                #3
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8              #4
   [-1, 9, C3, [256]],                                #5
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16             #6
   [-1, 9, C3, [512]],                                #7
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32            #8
   [-1, 1, SPP, [1024, [5, 9, 13]]],                  #9
   [-1, 3, C3, [1024, False]],  # 9                   #10
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],                        #11
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],        #12
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4      #13
   [-1, 3, C3, [512, False]],  # 13                   #14

   [-1, 1, Conv, [256, 1, 1]],                        #15
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],        #16 
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3      #17
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)      #18

   [-1, 1, Conv, [256, 3, 2]],                        #19
   [[-1, 14], 1, Concat, [1]],  # cat head P4         #20
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)    #21

   [-1, 1, Conv, [512, 3, 2]],                        #22
   [[-1, 11], 1, Concat, [1]],  # cat head P5         #23
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)    #24

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

- nc ： 代表class数目，目标检测一共有几个类
- depth_multiple: 控制模型重复个数
- width_multiple: 控制模型

- anchors: 每一行对应一个detect，注意和head最后一行第一个参数相匹配。
- backbone: 这个计数是从0开始的，不要被我后边写的数字误导。
  - 每行有四个参数：[from, number, module, args]
  - from: 代表连接哪一层，-1代表上一层。
  - number：这个模块的重复次数
  - module：模块的名称，一般在common.py和experimental.py中有这些模块的名称，也可以在里边添加新的模块。
  - args：模块需要的参数。
- head：对yolov3熟悉的应该知道，这部分构建的是FPN, 其中最后一行就是检测头。
  - detect：最后一行是检测头，第一个参数代表是从哪一层添加检测头。

如何添加注意力机制？这里提供其中一个改动方法，可能不是很科学，欢迎留言指出问题。

```yaml
## author: pprp
## parameters
nc: 15 # number of classes
depth_multiple: 1 # model depth multiple
width_multiple: 1 # layer channel multiple

# anchors
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Focus, [64, 3]], # 0-P1/2                 #1
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4              #2
    [-1, 3, C3, [128]], #3
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8              #4
    [-1, 9, C3, [256]], #5
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16             #6
    [-1, 9, C3, [512]], #7
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32            #8
    [-1, 1, SPP, [1024, [5, 9, 13]]], #9
    [-1, 3, C3, [1024, False]], # 9                   #10
    [-1, 1, SELayer, [1024, 4]], #10
  ]

# YOLOv5 head
head: [
    [-1, 1, Conv, [512, 1, 1]], #11 /32
    [-1, 1, nn.Upsample, [None, 2, "nearest"]], #12 /16
    [[-1, 6], 1, Concat, [1]], # cat backbone P4 /16       #13
    [-1, 3, C3, [512, False]], # 13 / 16                  #14

    [-1, 1, Conv, [256, 1, 1]], #15 /16 
    [-1, 1, nn.Upsample, [None, 2, "nearest"]], #16 /8
    [[-1, 4], 1, Concat, [1]], # cat backbone P3  /8    #17
    [-1, 3, C3, [256, False]], # 17 (P3/8-small) /8    #18

    [-1, 1, Conv, [256, 3, 2]], #19 /16
    [[-1, 6], 1, Concat, [1]], # cat head P4         #20
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)    #21

    [-1, 1, Conv, [512, 3, 2]], #22 /32
    [[-1, 8], 1, Concat, [1]], # cat head P5         #23
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)    #24

    [[18, 21, 24], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
```

可以看出笔者在backbone最后一层添加了SELayer，这个类我已经在common.py中添加进来，如下所示：

```python
class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1//r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1//r, c1, bias=False)
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
```

还需要在yolo.py中添加这个改动，这里参考了yolo守望者的代码。

```python
for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
    m = eval(m) if isinstance(m, str) else m  # eval strings
    for j, a in enumerate(args):
        try:
            args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        except:
            pass
    n = max(round(n * gd), 1) if n > 1 else n  # depth gain
    if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP,
                DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, 
                C3]:
        c1, c2 = ch[f], args[0]
        if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, 8)

        args = [c1, c2, *args[1:]]
        if m in [BottleneckCSP, C3]:
            args.insert(2, n)  # number of repeats
            n = 1
    elif m is nn.BatchNorm2d:
        args = [ch[f]]
    elif m is Concat:
        c2 = sum([ch[x] for x in f])
    elif m is Detect:
        args.append([ch[x] for x in f])
        if isinstance(args[1], int):  # number of anchors
            args[1] = [list(range(args[1] * 2))] * len(f)
    elif m is Contract:
        c2 = ch[f] * args[0] ** 2
    elif m is Expand:
        c2 = ch[f] // args[0] ** 2
    elif m is SELayer: # 这里是修改的部分
        channel, re = args[0], args[1]
        channel = make_divisible(channel * gw, 8) if channel != no else channel 
        args = [channel, re]
    else:
        c2 = ch[f]
```

这个方案目前还在运行中，等此次热身赛结束以后，会公开源码, 希望这个修改可以提升精度。

