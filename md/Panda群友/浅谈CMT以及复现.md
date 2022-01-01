# 浅谈CMT以及0-1复现

【GiantPandaCV导语】本篇博客讲解CMT模型并给出从0-1复现的过程以及实验结果，由于论文的细节并没有给出来，所以最后的复现和paper的精度有一点差异，等作者release代码后，我会详细的校对我自己的code，找找原因。

**论文链接**: https://arxiv.org/abs/2107.06263

**论文代码(个人实现版本): https://github.com/FlyEgle/CMT-pytorch**

**知乎专栏:https://www.zhihu.com/people/flyegle**

### 1. 出发点
- Transformers与现有的卷积神经网络（CNN）在性能和计算成本方面仍有差距。
- 希望提出的模型不仅可以超越典型的Transformers，而且可以超越高性能卷积模型。
### 2. 怎么做
1. 提出混合模型(串行)，通过利用Transformers来捕捉长距离的依赖关系，并利用CNN来获取局部特征。
2. 引入depth-wise卷积，获取局部特征的同时，减少计算量
3. 使用类似R50模型结构一样的stageblock，使得模型具有下采样增强感受野和迁移dense的能力。
4. 使用conv-stem来使得图像的分辨率缩放从VIT的1/16变为1/4，保留更多的patch信息。
### 3. 模型结构
![模型结构](https://tva1.sinaimg.cn/large/008i3skNgy1gtajxyhhurj31g90u0n55.jpg)
- (a)表示的是标准的R50模型，具有4个stage，每个都会进行一次下采样。最后得到特征表达后，经过AvgPool进行分类
- (b)表示的是标准的VIT模型，先进行patch的划分，然后embeeding后进入Transformer的block，这里，由于Transformer是long range的，所以进入什么，输出就是什么，引入了一个非image的class token来做分类。
- (c)表示的是本文所提出的模型框架CMT，由CMT-stem, downsampling, cmt block所组成，整体结构则是类似于R50，所以可以很好的迁移到dense任务上去。

#### 3.1. CMT Stem
使用convolution来作为transformer结构的stem，这个观点FB也有提出一篇paper，[Early Convolutions Help Transformers See Better](https://arxiv.org/abs/2106.14881)。

**CMT&Conv stem共性**
- 使用4层conv3x3+stride2 + conv1x1 stride 1 等价于VIT的patch embeeding，conv16x16 stride 16.
- 使用conv stem，可以使模型得到更好的收敛，同时，可以使用SGD优化器来训练模型，对于超参数的依赖没有原始的那么敏感。好处那是大大的多啊，仅仅是改了一个conv stem。

**CMT&Conv stem异性**
- 本文仅仅做了一次conv3x3 stride2，实际上只有一次下采样，相比conv stem，可以保留更多的patch的信息到下层。

从时间上来说，一个20210628(conv stem)， 一个是20210713(CMT stem)，存在借鉴的可能性还是比较小的，也说明了conv stem的确是work。

#### 3.2. CMT Block
每一个stage都是由CMT block所堆叠而成的，CMT block由于是transformer结构，所以没有在stage里面去设计下采样。每个CMT block都是由```Local Perception Unit, Ligntweight MHSA, Inverted Residual FFN```这三个模块所组成的，下面分别介绍：

- **Local Perception Unit(LPU)**

![LPU](https://tva1.sinaimg.cn/large/008i3skNgy1gtaltwnqddj308s062aa4.jpg)
本文的一个核心点是希望模型具有long-range的能力，同时还要具有local特征的能力，所以提出了LPU这个模块，很简单，一个3X3的DWconv，来做局部特征，同时减少点计算量，为了让Transformers的模块获取的longrange的信息不缺失，这里做了一个shortcut，公式描述为:

$$LPU(X) = DWConv(X) + X$$

- **Lightweight MHSA(LMHSA)**

![LMHSA](https://tva1.sinaimg.cn/large/008i3skNgy1gtam39b969j30f20g60tn.jpg)
MHSA这个不用多说了，多头注意力，Lightweight这个作用，PVT(链接：https://arxiv.org/abs/2102.12122)曾经有提出过，目的是为了降低复杂度，减少计算量。那本文是怎么做的呢，很简单，假设我们的输入为$H_{i} \times W_{i} \times C_{i}$, 对其分别做一个scale，使用卷积核为$k \times k$，stride为$k$的Depth Wise卷积来做了一次下采样，得到的shape为$\frac{H_{i}}{k} \times \frac{W_{i}}{k} \times C_{i}$，那么对应的Q,K,V的shape分别为:
$$
\begin{aligned}
    Q_{shape} = (H_{i}\times W_{i}) \times C_{i} = N_{i} \times C_{i} \\
    K_{shape} = (\frac{H_{i}}{k_{i}}\times \frac{W_{i}}{k_{i}}) \times C_{i} = N_{I}^{'} \times C_{i} \\
    V_{shape} = (\frac{H_{i}}{k_{i}}\times \frac{W_{i}}{k_{i}}) \times C_{i} = N_{I}^{'} \times C_{i} \\
    \end{aligned}
$$

我们知道，在计算MHSA的时候要遵守两个计算原则:
1. Q, K的序列dim要一致。
2. K, V的token数量要一致。

所以，本文中的MHSA计算公式如下:

$$
LeightweightMHSA(Q,K,V) = Softmax(\frac{Q{K^{'}}^{T}}{\sqrt{d_{k}}} + B)V^{'}
$$

- **Inverted Resdiual FFN(IRFFN)**

![ffn](https://tva1.sinaimg.cn/large/008i3skNgy1gtamlfvdsmj30eu0cu3yz.jpg)

FFN的这个模块，其实和mbv2的block基本上就是一样的了，不一样的地方在于，使用的是GELU，采用的也是DW+PW来减少标准卷积的计算量。很简单，就不多说了，公式如下:

$$
\begin{aligned}
IRFFN(X) = Conv(F(Conv(X))) \\ 
F(X) = DWConv(X) + X
\end{aligned}
$$

那么我们一个block里面的整体计算公式如下:

$$
\begin{aligned}
X_{i}^{'} = LPU(X_{i-1})\\
X_{i}^{''} = LMHSA(LN(X_{i}^{'})) + X_{i}^{'} \\
X_{i} = IRFFN(LN(X_{i}^{''})) + X_{i}^{''}
\end{aligned}
$$

#### 3.3 patch aggregation 
每个stage都是由上述的多个CMTblock所堆叠而成, 上面也提到了，这里由于是transformer的操作，不会设计到scale尺度的问题，但是模型需要构造下采样，来实现层次结构，所以downsampling的操作单独拎了出来，每个stage之前会做一次卷积核为2x2的，stride为2的卷积操作，以达到下采样的效果。


所以，整体的模型结构就一目了然了，假设输入为224x224x3,经过CMT-STEM和第一次下采样后，得到了一个56x56的featuremap，然后进入stage1，输出不变，经过下采样后，输入为28x28，进入stage2，输出后经过下采样，输入为14x14，进入stage3，输出后经过最后的下采样，输入为7x7，进入stage4，最后输出7x7的特征图，后面接avgpool和分类，达到分类的效果。

我们接下来看一下怎么复现这篇paper。


### 4. 论文复现

**ps**: 这里的**复现**指的是没有源码的情况下，实现网络，训练等，如果是结果复现，会标明为**复现精度**。

这里存在几个问题
- 文章的问题：我看到paper的时候，是第一个版本的arxiv，大概过了一周左右V2版本放出来了，这两个版本有个很大的diff。
Version1
![V1](https://tva1.sinaimg.cn/large/008i3skNgy1gtao4xnv46j316u0lstg7.jpg)
Version2
![V2](https://tva1.sinaimg.cn/large/008i3skNgy1gtao8h75tdj312y0k0q9a.jpg)
网络结构可以说完全不同的情况下，FLOPs竟然一样的，当然可能是写错了，这里就不吐槽了。不过我一开始代码复现就是按下面来的，所以对于我也没影响多少，只是体验有点差罢了。
- 细节的问题：paper和很多的transformer一样，都是采用了Deit的训练策略，但是差别在于别的paper或多或少会给出来额外的tirck，比如最后FC的dp的ratio等，或者会改变一些，再不济会把代码直接release了，所以只好闷头尝试Trick。

#### 4.1 复现难点
paper里面采用的Position Embeeding和Swin是类似的，都是Relation Position Bias，但是和Swin不相同的是，我们的Q,K,V尺度是不一样的。这里我考虑了两种实现方法，一种是直接bicubic插值，另一种则是切片，切片更加直观且embeeding我设置的可BP，所以，实现里面采用的是这种方法，代码如下：
```python
def generate_relative_distance(number_size):
    """return relative distance, (number_size**2, number_size**2, 2)
    """
    indices = torch.tensor(np.array([[x, y] for x in range(number_size) for y in range(number_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances = distances + number_size - 1   # shift the zeros postion
    return distances
...
elf.position_embeeding = nn.Parameter(torch.randn(2 * self.features_size - 1, 2 * self.features_size - 1))

...
q_n, k_n = q.shape[1], k.shape[2]
attn = attn + self.position_embeeding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]][:, :k_n]
```

#### 4.2 复现trick历程(血与泪TT)
一方面想要看一下model是否是work的，一方面想要顺便验证一下DeiT的策略是否真的有效，所以从头开始做了很多的实验，简单整理如下：
- 数据:
    1. 训练数据: 20%的imagenet训练数据(快速实验)。
    2. 验证数据: 全量的imagenet验证数据。

- 环境:
    1. 8xV100 32G
    2. CUDA 10.2 + pytorch 1.7.1 

- sgd优化器实验记录

|model|augments|resolution|batchsize|epoch|optimizer|LR|strategy|weightdecay|top-1@acc|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CMT-TINY | crop+flip                                                       | 184->160   | 512X8 | 120   | SGD                        | 1.6 | cosine | 1.00E-04 | 0.55076 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD                        | 1.6 | cosine | 1.00E-04 | 0.59714 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup                             | 184->160   | 512X8 | 120   | SGD                        | 1.6 | cosine | 1.00E-04 | 0.57034 |
| CMT-TINY | crop+flip+colorjitter+randaug+cutmix                            | 184->160   | 512X8 | 120   | SGD                        | 1.6 | cosine | 1.00E-04 | 0.57264 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD                        | 1.6 | cosine | 5.00E-05 | 0.59452 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup                             | 184->160   | 512X8 | 200   | SGD                        | 1.6 | cosine | 1.00E-04 | 0.60532 |
| CMT-TINY | crop+flip+colorjitter+randaug+cutmix                            | 184->160   | 512X8 | 300   | SGD                        | 1.6 | cosine | 1.00E-04 | 0.61192 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 200   | SGD                        | 1.6 | cosine | 5.00E-05 | 0.60172 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+ape(wrong->resolution) | 1.6 | cosine | 1.00E-04 | 0.60276 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+rpe                    | 1.6 | cosine | 1.00E-04 | 0.6016  |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+ape(real->resolution)  | 1.6 | cosine | 1.00E-04 | 0.60368 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+pe_nd                  | 1.6 | cosine | 1.00E-04 | 0.59494 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+qkv_bias               | 1.6 | cosine | 1.00E-04 | 0.59902 |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+qkv_bias+rpe           | 1.6 | cosine | 1.00E-04 | 0.6023  |
| CMT-TINY | crop+flip+colorjitter+randaug                                   | 184->160   | 512X8 | 120   | SGD+qkv_bias+ape           | 1.6 | cosine | 1.00E-04 | 0.5986  |
| CMT-TINY | crop+flip+colorjitter+randaug+no mixup+no_cutmix+labelsmoothing | 184->160   | 512X8 | 300   | SGD+qkv_bias+rpe           | 1.6 | cosine | 1.00E-04 | 0.62108 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing       | 184->160   | 512X8 | 300   | SGD+qkv_bias+rpe           | 1.6 | cosine | 1.00E-04 | 0.6612  |

**结论**: 可以看到在SGD优化器的情况下，使用1.6的LR，训练300个epoch，warmup5个epoch，是用cosine衰减学习率的策略，用randaug+colorjitter+mixup+cutmix+labelsmooth，设置weightdecay为0.1的配置下，使用QKV的bias以及相对位置偏差，可以达到比baseline高11%个点的结果，所有的实验都是用FP16跑的。

- adamw优化器实验记录

|model|augments|resolution|batchsize|epoch|optimizer|LR|strategy|weightdecay|top-1@acc|
|----------|------------------------------------------------------------------------|------------|-------|-------|--------------------|----------|--------|----------|---------|
| CMT-TINY | crop+flip                                                              | 184->160   | 512X8 | 120   | AdamW              | 4.00E-03 | cosine | 5.00E-02 | 0.50994 |
| CMT-TINY | crop+flip+colorjitter+randaug                                          | 184->160   | 512X8 | 300   | AdamW              | 4.00E-03 | cosine | 5.00E-02 | 0.57646 |
| CMT-TINY | crop+flip+colorjitter+randaug                                          | 184->160   | 512X8 | 120   | AdamW              | 4.00E-03 | cosine | 1.00E-04 | 0.56504 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 4.00E-03 | cosine | 1.00E-04 | 0.63606 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing + repsampler | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 4.00E-03 | cosine | 1.00E-04 | 0.61826 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 4.00E-03 | cosine | 5.00E-02 | 0.64228 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 1.00E-04 | cosine | 5.00E-02 | 0.4049  |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing + repsampler | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 4.00E-03 | cosine | 5.00E-02 | 0.63816 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 8.00E-03 | cosine | 5.00E-02 | 不收敛     |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 5.00E-03 | cosine | 5.00E-02 | 0.65118 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 6.00E-03 | cosine | 5.00E-02 | 0.65194 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 6.00E-03 | cosine | 5.00E-03 | 0.63726 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing              | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 6.00E-03 | cosine | 1.00E-01 | 0.65502 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing+warmup20     | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 6.00E-03 | cosine | 1.00E-01 | 0.65082 |
| CMT-TINY | crop+flip+colorjitter+randaug+mixup+cutmix+labelsmoothing+droppath     | 184->160   | 512X8 | 300   | adamw+qkv_bias+rpe | 6.00E-03 | cosine | 1.00E-01 | 0.66908 |

**结论**：使用AdamW的情况下，对学习率的缩放则是以512的bs为基础，所以对于4k的bs情况下，使用的是4e-3的LR，但是实验发现增大到6e-3的时候，还会带来一些提升，同时放大一点weightsdecay，也略微有所提升，最终使用AdamW的配置为，6e-3的LR，1e-1的weightdecay，和sgd一样的增强方法，然后加上了随机深度失活设置，最后比baseline高了16%个点，比SGD最好的结果要高0.8%个点。


#### 4.3. imagenet上的结果
![result](https://tva1.sinaimg.cn/large/008i3skNgy1gtar15ktqtj31hm05sgmr.jpg)

最后用全量跑，使用SGD会报nan的问题，我定位了一下发现，running_mean和running_std有nan出现，本以为是数据增强导致的0或者nan值出现，结果空跑几次数据发现没问题，只好把优化器改成了AdamW，结果上述所示，CMT-Tiny在160x160的情况下达到了75.124%的精度，相比MbV2,MbV3的确是一个不错的精度了，但是相比paper本身的精度还是差了将近4个点，很是离谱。

速度上，CMT虽然FLOPs低，但是实际的推理速度并不快，128的bs条件下，速度慢了R50将近10倍。

### 5. 实验结果
总体来说，CMT达到了更小的FLOPs同时有着不错的精度, imagenet上的结果如下：
![imagenet](https://tva1.sinaimg.cn/large/008i3skNgy1gtarc73l00j30u00vawlg.jpg)

coco2017上也有这不错的精度
![coco](https://tva1.sinaimg.cn/large/008i3skNgy1gtarcq9nf5j318o0fadk4.jpg)

### 6. 结论
本文提出了一种名为CMT的新型混合架构，用于视觉识别和其他下游视觉任务，以解决在计算机视觉领域以粗暴的方式利用Transformers的限制。所提出的CMT同时利用CNN和Transformers的优势来捕捉局部和全局信息，促进网络的表示能力。在ImageNet和其他下游视觉任务上进行的大量实验证明了所提出的CMT架构的有效性和优越性。

**代码复现repo: https://github.com/FlyEgle/CMT-pytorch**, 实现不易，求个star！





