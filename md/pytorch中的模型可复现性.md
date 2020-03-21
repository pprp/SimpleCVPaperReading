# PyTorch中模型的可复现性

> 在深度学习模型的训练过程中，难免引入随机因素，这就会对模型的可复现性产生不好的影响。但是对于研究人员来讲，模型的可复现性是很重要的。这篇文章收集并总结了可能导致模型难以复现的原因，虽然**不可能完全避免随机因素**，但是可以通过一些设置尽可能降低模型的随机性。

### 1. 常规操作

PyTorch官方提供了一些关于可复现性的解释和说明。

在PyTorch发行版中，**不同的版本或不同的平台上，不能保证完全可重复的结果**。此外，即使在使用相同种子的情况下，结果也不能保证在CPU和GPU上再现。

但是，为了使计算能够在一个特定平台和PyTorch版本上确定特定问题，需要采取几个步骤。

PyTorch中涉及两个伪随机数生成器，需要手动对其进行播种以使运行可重复。此外，还应确保代码所依赖的所有其他库以及使用随机数的库也使用固定种子。

常用的固定seed的方法有：

```python
import torch
import numpy as np
import random

seed=0

random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Remove randomness (may be slower on Tesla GPUs) 
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

API中也揭示了原因，PyTorch使用的CUDA实现中，有一部分是**原子操作**，尤其是`atomicAdd`，使用这个操作就代表数据不能够并行处理，需要串行处理，使用到`atomicAdd`之后就会按照不确定的并行加法顺序执行，从而引入了不确定因素。PyTorch中使用到的`atomicAdd`的方法：

**前向传播时：**

- torch.Tensor.index_add_()
- torch.Tensor.scatter_add_()
- torch.bincount()

**反向传播时：**

- torch.nn.functional.embedding_bag()
- torch.nn.functional.ctc_loss()
- 其他pooling，padding, sampling操作

可以说由于需要并行计算，从而引入atomicAdd之后，必然会引入不确定性，目前没有一种简单的方法可以完全避免不确定性。

### 2. upsample层

upsample导致模型可复现性变差，这一点在PyTorch的官方库issue`#12207`中有提到。也有很多热心的人提供了这个的解决方案：

```python
import torch.nn as nn
class UpsampleDeterministic(nn.Module):
    def __init__(self,upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        '''
        return x[:, :, :, None, :, None]\
        .expand(-1, -1, -1, self.upscale, -1, self.upscale)\
        .reshape(x.size(0), x.size(1), x.size(2)\
                 *self.upscale, x.size(3)*self.upscale)
        
# or
def upsample_deterministic(x,upscale):
    return x[:, :, :, None, :, None]\
    .expand(-1, -1, -1, upscale, -1, upscale)\
    .reshape(x.size(0), x.size(1), x.size(2)\
             *upscale, x.size(3)*upscale)
```

可以将以上模块替换掉官方的nn.Upsample函数来避免不确定性。

### 3. Batch Size

Batch Size这个超参数很容易被人忽视，很多时候都是看目前剩余的显存，然后再进行设置合适的Batch Size参数。**模型复现时Batch Size大小是必须相同的。**

Batch Size对模型的影响很大，Batch Size决定了要经过多少对数据的学习以后，进行一次反向传播。

Batch Size过大：

- 占用显存过大，在很多情况下很难满足要求。对内存的容量也有更高的要求。
- 容易陷入局部最小值或者鞍点，模型会在发生过拟合，在训练集上表现非常好，但是测试集上表现差。

Batch Size过小：

- 假设bs=1,这就属于在线学习，每次的修正方向以各自样本的梯度方向修正，很可能将难以收敛。
- 训练时间过长，难以提高资源利用率

另外，由于CUDA的原因，Batch Size设置为2的幂次的时候速度更快一些。所以尝试修改Batch Size的时候就按照4,8,16,32,...这样进行设置。

## 4. 数据在线增强

在这里参考的库是ultralytics的yolov3实现，数据增强分为**在线增强**和**离线增强**：

- **在线增强**：在获得 batch 数据之后，然后对这个 batch 的数据进行增强，如旋转、平移、翻折等相应的变化，由于有些数据集不能接受线性级别的增长，这种方法常常用于大的数据集。
- **离线增强**：直接对数据集进行处理，数据的数目会变成增强因子 x 原数据集的数目 ，这种方法常常用于数据集很小的时候。

在yolov3中使用的就是在线增强，比如其中一部分增强方法：

```python
if self.augment:
    # 随机左右翻转
    lr_flip = True
    if lr_flip and random.random() < 0.5:
        img = np.fliplr(img)
    if nL:
        labels[:, 1] = 1 - labels[:, 1]

    # 随机上下翻转
    ud_flip = False
    if ud_flip and random.random() < 0.5:
        img = np.flipud(img)
        if nL:
            labels[:, 2] = 1 - labels[:, 2]
```

可以看到，如果设置了在线增强，那么模型会**以一定的概率**进行增强，这样会导致每次运行得到的训练样本可能是不一致的，这也就造成了模型的不可复现。为了复现，这里暂时将在线增强的功能关掉。

### 5. 多线程操作

FP32(或者FP16 apex)中的随机性是由**多线程**引入的，在PyTorch中设置DataLoader中的num_worker参数为0，或者直接不使用GPU，通过`--device cpu`指定使用CPU都可以避免程序使用多线程。但是这明显不是一个很好的解决方案，因为两种操作都会显著地影响训练速度。

任何多线程操作都可能会引入问题，甚至是对单个向量求和，因为线程求和将导致FP16 / 32的精度损失，从而执行的顺序和线程数将对结果产生轻微影响。

### 6. 其他

- 所有模型涉及到的文件中使用到random或者np.random的部分都需要设置seed

- dropout可能也会带来随机性。

- 多GPU并行训练会带来一定程度的随机性。
- 可能还有一些其他问题，感兴趣的话可以看一下知乎上问题: **PyTorch 有哪些坑/bug？**

### 7. 总结

上面大概梳理了一下可能导致PyTorch的模型可复现性出现问题的原因。可以看出来，有**很多问题是难以避免**的，比如使用到官方提及的几个方法、涉及到atomicAdd的操作、多线程操作等等。

笔者也在yolov3基础上修改了以上提到的内容，**固定了seed,batch size,关闭了数据增强**。在模型运行了10个epoch左右的时候，前后两次训练的结果是一模一样的，但是**随着epoch越来越多，也会产生一定的波动**。

总之，应该尽量满足可复现性的要求，我们可以通过设置固定seed等操作，尽可能保证前后两次相同实验得到的结果波动不能太大，不然就很难判断模型的提升是**由于随机性导致的还是对模型的改进导致**的。

目前笔者进行了多次试验来研究模型的可复现性，**偶尔会出现两次一模一样的训练结果**，但是更多实验中，两次的训练结果都是略有不同的，不过通过以上设置，可以让训练结果差距在1%以内。

在目前的实验中还无法达到每次前后两次完全一样，如果有读者有类似的经验，欢迎来交流。

### 8. 参考链接

<https://pytorch.org/docs/stable/notes/randomness.html>

<https://github.com/pytorch/pytorch/issues/12207>

<https://www.zhihu.com/question/67209417>

