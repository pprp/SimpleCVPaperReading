# 【Pytorch基础】BatchNorm常识梳理与使用

BatchNorm, 批规范化，主要用于解决协方差偏移问题，主要分三部分：

- 计算batch均值和方差
- 规范化
- 仿射affine

算法内容如下：

![图源https://blog.csdn.net/LoseInVain/article/details/86476010](https://img-blog.csdnimg.cn/20210529092041293.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



需要说明几点：

- 均值和方差是batch的统计特性，pytorch中用running_mean和running_var表示
- $\gamma $和$\beta$是可学习的参数，分别是affine中的weight和bias

![](https://img-blog.csdnimg.cn/20210529100607538.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

以BatchNorm2d为例，分析其中变量和参数的意义：

- affine: 仿射的开关，决定是否使用仿射这个过程。

  - affine=False则$\gamma=1,\beta=0$ ，并且不能学习和更新。
  - affine=True则以上两者都可以更新

- training：模型为训练状态和测试状态时的运行逻辑是不同的。

- track_running_stats: 决定是否跟踪整个训练过程中的batch的统计特性，而不仅仅是当前batch的特性。

- num_batches_tracked：如果设置track_running_stats为真，这个就会起作用，代表跟踪的batch个数，即统计了多少个batch的特性。

- momentum: 滑动平均计算running_mean和running_var

  $\hat{x}_{\text {new }}=(1-$ momentum $) \times \hat{x}+$ momentum $\times x_{t}$

  

```python
class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
```

training和tracking_running_stats有四种组合：

- training=True,tracking_running_stats=True: 这是正常的训练过程，BN跟踪的对象是整个训练过程的batch特性。
- training=True,tracking_running_stats=False: BN不会跟踪整个训练过程的batch特性，而只是计算当前batch的统计特性。
- training=False,tracking_running_stats=True: 正常的测试过程，BN会用之前训练好的running_mean和running_var，并且不会对其进行更新。（ps: 这就是有时候为何有一些NAS算法会使用BN校正技术，即在训练集上运行几个batch，更新running_mean和running_var）
- training=False,tracking_running_stats=False: 一般不采用这种，只计算当前测试batch统计特性，容易造成统计特性偏移，对结果造成不好的结果。

更新过程：

- running_mean和running_var是在forward过程中更新的，记录在buffer中（即不可通过反向传播算法影响的部分）
- $\alpha, \gamma$是在反向传播中更新的。
- 在蒸馏过程中，需要注意教师模型需要设置model.eval()来固定running_mean和running_var，否则会不发生变化，对结果造成不确定的影响。





**参考文献：**

https://blog.csdn.net/LoseInVain/article/details/86476010

https://blog.csdn.net/yangwangnndd/article/details/94901175