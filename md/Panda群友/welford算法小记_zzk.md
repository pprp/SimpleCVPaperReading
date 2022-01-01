【GiantPandaCV导语】
前段时间debug LayerNorm的时候，看见Pytorch LayerNorm计算方差的方式与我们并不一样。它使用了一种在线更新算法，速度更快，数值稳定性更好，这篇笔记就当一篇总结。

## 回顾常见的方差计算方法

### Two-pass方法

这种方法就是方差的定义式了：
$$
\sigma^2 = \frac{\Sigma_{i=1}^{n}(x_i - mean)^2}{n}
$$
简单来说就是样本减去均值，取平方，然后再累加起来除以样本数量（这里就不再具体分总体方差和样本方差了）。

那为什么他叫Two-pass方法呢？因为他需要循环两遍原始数据：
- 第一遍统计，计算均值
- 第二遍再将样本值和均值计算，得到方差
当数据比较大的时候，两遍循环耗时也比较多

### Naive方法

我们还知道方差和均值的一个关系式子
$$
D(X) = E(X^2) - E(X)^2
$$
相比Two-pass方法，这种方法仅仅只需要遍历一遍数据。
我们只需要在外面统计两个变量,`sum` 和 `sum_square`。

最后再分别计算两者的均值，通过上述关系式子得到结果

根据维基百科的介绍，前面这两种方法的一个共同缺点是，其结果依赖于数据的排序，存在累加的舍入误差，对于大数据集效果较差

### Welford算法

此前大部分深度学习框架都采用的是Naive的计算方法，后续Pytorch转用了这套算法。

首先给出结果，我们再来进行一步步的推导：
$$
\overline{x_{n+1}} = \overline{x_{n}} + \frac{x_{n+1} - \overline{x_{n}}}{n+1}
$$
其中$\overline{x_n}$表示前n个元素的均值

$$
\sigma_{n+1}^2 = \sigma_{n}^2 + \frac{(x_{n+1} - \overline{x_{n}})(x_{n+1} - \overline{x_{n+1}}) - \sigma_{n}^2}{n+1}
$$

### 推导

首先我们推导均值的计算:
$$
\overline{x_n} = \frac{1}{N}\Sigma_{i=1}^n{x_i}
$$
当为n+1的情况下：
$$
\overline{x_{n+1}} = \frac{1}{N+1}\Sigma_{i=1}^{n+1}{x_i}
$$
$$
(N+1)\overline{x_{n+1}} = \Sigma_{i=1}^{n+1}{x_i}
$$
$$
(N+1)\overline{x_{n+1}} = \Sigma_{i=1}^{n+1}{x_i}
$$
$$
(N+1)\overline{x_{n+1}} = \Sigma_{i=1}^{n}{x_i} + x_{n+1}
$$
$$
(N+1)\overline{x_{n+1}} = N\overline{x_n} + x_{n+1}
$$
$$
\overline{x_{n+1}} = \frac{N\overline{x_n} + x_{n+1}}{N+1}
$$
$$
\overline{x_{n+1}} = \overline{x_n} + \frac{1}{N+1}(x_{n+1} - \overline{x_n})
$$

方差的推导稍微有点复杂，做好心理准备！

首先我们回到Naive公式
$$
\sigma_n^2 = E(X^2) - \overline{x_n}^2
$$
$$
\sigma_n^2 = \frac{1}{N}\Sigma_{i=1}^{n}x_i^2 - \overline{x_n}^2
$$
$$
N\sigma_n^2 = \Sigma_{i=1}^{n}x_i^2 - N\overline{x_n}^2
$$
我们看下n+1时候的情况
$$
\sigma_{n+1}^2 = \frac{1}{N+1}\Sigma_{i=1}^{n+1}x_i^2 - \overline{x_{n+1}}^2
$$
我们把n+1乘到左边，并把n+1的平方项单独拆出来
$$
(N+1)\sigma_{n+1}^2 = \Sigma_{i=1}^{n}x_i^2 + x_{n+1}^2 - (N+1)\overline{x_{n+1}}^2
$$
而根据前面计算我们可以把$\Sigma_{i=1}^{n}x_i^2$替换掉
$$
(N+1)\sigma_{n+1}^2 = N\sigma_n^2 + N\overline{x_n}^2 + x_{n+1}^2 - (N+1)\overline{x_{n+1}}^2
$$
而$\overline{x_{n+1}}$我们前面推导均值的时候推导过，此时替换进来
$$
(N+1)\sigma_{n+1}^2 = N\sigma_n^2 + N\overline{x_n}^2 + x_{n+1}^2 - \frac{(N\overline{x_n} + x_{n+1})^2}{N+1}
$$
左右两遍，同时乘上N+1，并进行化简，可以得到：
$$
(N+1)^2\sigma_{n+1}^2 = N(N+1)\sigma_{n}^2 + N(\overline{x_n} - x_{n+1})^2
$$
把$(N+1)^2$挪到右边就可以得到
$$
\sigma_{n+1}^2 = \sigma_{n}^2 + \frac{N(\overline{x_n}-x_{n+1})^2 - (N+1)\sigma_n^2}{(N+1)^2}
$$

而根据平方公式的特性有
$$
(\overline{x_n}-x_{n+1})^2 = (x_{n+1}-\overline{x_n})^2
$$
我们将其中一项用前面推导得到的均值来进行转换
$$
(x_{n+1}-\overline{x_n}) = (N+1)(\overline{x_{n+1}} - \overline{x_n})
$$
然后替换到前面的公式进行化简就可以得到最终结果
$$
\sigma_{n+1}^2 = \sigma_{n}^2 + \frac{(x_{n+1} - \overline{x_{n}})(x_{n+1} - \overline{x_{n+1}}) - \sigma_{n}^2}{n+1}
$$
### 额外拓展：
这样子更新方差，**每一次都可能会加一个较小的数字，也会导致舍入误差**，因此又做了个变换：

每次统计：
$$
M_{2, n} = M_{2, n-1} + (x_n - \overline{x_{n-1}})(x_n -  \overline{x_{n}})
$$
最后再得到方差：
$$
\sigma_n^2 = \frac{M_{2, n}}{N}
$$
这个转换是一个**等价转换**，感兴趣的读者可以从头一项一项的推导。

## 实现代码
简单用python写了个脚本
```python
import numpy as np


def welford_update(count, mean, M2, currValue):
    count += 1
    delta = currValue - mean
    mean += delta / count
    delta2 = currValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


def naive_update(sum, sum_square, currValue):
    sum = sum + currValue
    sum_square = sum_square + currValue * currValue
    return (sum, sum_square)


x_arr = np.random.randn(100000).astype(np.float32)

welford_mean = 0
welford_m2 = 0
welford_count = 0
for i in range(len(x_arr)):
    new_val = x_arr[i]
    welford_count, welford_mean, welford_m2 = welford_update(welford_count, welford_mean, welford_m2, new_val)
print("Welford mean: ", welford_mean)
print("Welford var: ", welford_m2 / welford_count)

naive_sum = 0
naive_sum_square = 0
for i in range(len(x_arr)):
    new_val = x_arr[i]
    naive_sum, naive_sum_square = naive_update(naive_sum, naive_sum_square, new_val)
naive_mean = naive_sum / len(x_arr)
naive_var = naive_sum_square/ len(x_arr) - naive_mean*naive_mean
print("Naive mean: ", naive_mean)
print("Naive var: ", naive_var)
```
更多的代码可以参考pytorch和apex实现：

pytorch moments实现：
https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SharedReduceOps.h#L95-L113

apex实现：https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu#L11-L24

## 参考资料：

- wiki：https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
- https://changyaochen.github.io/welford/ 

笔者主要是根据上面这两个材料进行学习，第二个博客写的十分详细，还有配套的jupyter notebook代码跑，十分推荐。