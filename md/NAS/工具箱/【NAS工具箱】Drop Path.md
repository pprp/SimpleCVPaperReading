# 【NAS工具箱】Drop Path介绍 + Dropout回顾

【前言】Drop Path是NAS中常用到的一种正则化方法，由于网络训练的过程中常常是动态的，Drop Path就成了一个不错的正则化工具，在FractalNet、NASNet等都有广泛使用。

## Dropout

Dropout是最早的用于解决过拟合的方法，是所有drop类方法的大前辈。Dropout在12年被Hinton提出，并且在ImageNet Classification with Deep Convolutional Neural Network工作AlexNet中使用到了Dropout。

**原理** ：在前向传播的时候，让某个神经元激活以概率1-keep_prob（0<p<1）停止工作。

**功能** ： 这样可以让模型泛化能力更强，因为其不会过于以来某些局部的节点。训练阶段以keep_prob的概率保留，以1-keep_prob的概率关闭；测试阶段所有的神经元都不关闭，但是对训练阶段应用了dropout的神经元，输出值需要乘以keep_prob。

具体是这样的：

> 假设一个神经元的输出激活值为`a`，在不使用dropout的情况下，其输出期望值为`a`，如果使用了dropout，神经元就可能有保留和关闭两种状态，把它看作一个离散型随机变量，它就符合概率论中的**0-1分布**，其输出激活值的期望变为 `p*a+(1-p)*0=pa`，此时若要保持期望和不使用dropout时一致，就要除以 `p`。
> 作者：种子_fe
> 链接：https://www.imooc.com/article/30129

**实现** ： pytorch中的实现如下。

```python
class _DropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)
    
class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)
```

funtional.py中的dropout实现：

```python
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.
    See :class:`~torch.nn.Dropout` for details.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(dropout, (input,), input, p=p, training=training, inplace=inplace)
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
```

最终在Dropout.cpp中找到具体实现：

```c++
template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  noise.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }  

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}
```

流程：

- 判断p的范围 以及训练状态
- 使用1-p的概率得到伯努利分布（0-1分布）
- (input / 1-p) * 伯努利分布



## Drop Path

**原理** ：字如其名，Drop Path就是随机将深度学习网络中的多分支结构随机删除。

**功能** ：一般可以作为正则化手段加入网络，但是会增加网络训练的难度。尤其是在NAS问题中，如果设置的drop prob过高，模型甚至有可能不收敛。

**实现** ：

```python
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

有了Dropout的理论铺垫，这里的实现就比较明了了，具体使用的时候一般是这样的：

```python
x = x + self.drop_path(self.conv(x))
```

Drop Path不能直接这样使用：

```python
x = self.drop_path(x)
```



## Reference

https://www.cnblogs.com/dan-baishucaizi/p/14703263.html

https://www.imooc.com/article/30129

https://www.github.com/pytorch/pytorch











