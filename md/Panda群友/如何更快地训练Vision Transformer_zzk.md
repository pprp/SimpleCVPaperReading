如何更快地训练Vision Transformer

近期MetaAI发布了一篇博客，关于如何显著提升Vision Transformer的训练效率。

原文：[Significantly faster Vision Transformer training]
链接：https://ai.facebook.com/blog/significantly-faster-vision-transformer-training

## What the research is

Vision Transformer模型几乎火遍计算机视觉各个领域，其性能随着参数增加和更久的训练过程而得到提升。随着模型越来越大，超过了万亿次浮点运算的规模，该领域达到了瓶颈：训练一个模型往往要一个月，需要几百上千个GPU，导致大部分人无法接触到大规模ViT模型，并进而增加了对加速器的需求。

为了降低门槛，让更多人能够应用ViT，我们开发一系列方法来加速整个训练。我们基于MetaAI的图像分类模型库**PyCls**实现了一系列优化，这些优化极大的提升了模型训练过程的吞吐量：

![](https://files.mdnice.com/user/4601/6a5d908a-35bf-4c45-b527-095418947443.png)

## How it works ？

我们首先对代码库进行分析，以定位训练效率低下的原因，最后关注点落在计算类型上：大部分模型都是用FP32进行训练，如果使用FP16训练的话，可以降低显存占用，并提高模型训练速度，但这一做法经常会导致准确率下降

所以我们选了一个折中的方法：**自动混合精度**。在该方法下，我们用half类型进行计算，以加快训练，减少显存使用。并以fp32类型存储参数，以保证模型准确率。其中我们没有手动将网络各部分转换成half类型，而是应用AMP各种模式（如O1, O2, O3)，以寻找一个既能提升速度又不影响精度的平衡点。

## FSDP

为了让训练更加高效，我们应用了FSDP训练策略，他能够将参数，梯度，优化器状态分片到各GPU上。在FSDP的帮助下，我们可以用更少的GPU资源构建更大的模型。

> FSDP策略可以参考 [数据并行Deep-dive: 从DP 到 Fully Sharded Data Parallel （FSDP）完全分片数据并行] 链接：https://zhuanlan.zhihu.com/p/485208899

## MTA Optimizer

前向计算完毕后，优化器需要对各个参数进行修改。而当参数比较多的情况下，对应启动的Optimizer Kernel就会变得很多，通常这些Kernel都比较小，计算负担不大，启动Kernel的开销反而占了大头。

在**ContiguousParams**中，它将模型参数放置到一块连续的显存中进行计算，这样就能减少优化器这部分的时间。下图是Resnet50+SGD是否应用ContiguousParams的比较，可以看到OptimizerStep这部分时间显著减少了。

![](https://files.mdnice.com/user/4601/ef19eaef-71d1-483d-93ec-e15fb2bd5d8b.png)

而NVIDIA的Apex库的做法则是在底层重新实现了一系列MultiTensorOptimizer，如Adam, Adagrad等等。

Apex这种方法比较硬核，普通用户如果想要自己自定义优化器并应用Multi Tensor的优化，就必须改动底层CUDA代码。而最近PyTorch也在计划提供了一系列`foreach`接口[Replace optimizers in torch.optim with the ones from torch.optim._multi_tensor] 链接：https://github.com/pytorch/pytorch/pull/49039，让用户只需要在Python层即可享受到优化，对应的MultiTensor版Momentum优化器代码如下所示：

```python
torch._foreach_mul_(bufs, momentum)
torch._foreach_add_(bufs, grads, alpha=1 - dampening)
```

## Pooled Classifier

原版的ViT是额外加了一个分类token，来输出最后的分类结果。而这里采用平均池化 如：https://github.com/facebookresearch/pycls/blob/main/pycls/core/config.py#L205 处理最后的分类

## Batch Second Input Tensor Layout

这里的数据格式与以往不同，将batch维度放在第二维，并在调用`nn.MultiheadAttention`的时候，设置`batch_first=False`，以减少不必要的转置

```python
if self.batch_first and is_batched:
    return attn_output.transpose(1, 0), attn_output_weights
else:
    return attn_output, attn_output_weights
```

> 总感觉这个实现怪怪的

## 其他优化

我们在采取560大小的batchsize下，达到了1.51倍的加速比，进一步的我们将batchsize设置为384，并将图片大小增大到256，达到了1.86倍加速比。在全FP16运算下，能够达到2.18倍加速比，尽管这偶尔会降低准确率（在实验中，准确率降低不到10%）。

![](https://files.mdnice.com/user/4601/a3cfb98d-8fec-4d64-8446-f2a3bc69bce9.png)

使用上述优化，我们将Imagenet1K数据集每epoch训练时间从0.65小时降低到0.43小时

![](https://files.mdnice.com/user/4601/926818b8-a831-4756-8540-896af69bc141.png)

我们还研究了不同GPU配置对训练速度的影响，在不同配置下我们都实现了比DDP baseline更高的吞吐量。随着GPU增加，吞吐量会因为设备之间的通信开销略微下降。然而即使在64块GPU下，我们仍然比DDP基线快1.83倍

![](https://files.mdnice.com/user/4601/11d5d7f3-e2a7-4b92-98c3-eac465288e9b.png)

## 文中链接

PyCls ： https://github.com/facebookresearch/pycls

ContiguousParams：https://github.com/PhilJd/contiguous_pytorch_params

Adam：https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_adam.cu
