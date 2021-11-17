---
title: python切片用法总结
date: 2019-07-02 14:54:46
tags: 
- python
categories:
- 深度学习
---

> 在深度学习中进行张量操作的时候经常会遇到这样的切片形式，一开始总会感到头疼，所以现在总结一下用法，欢迎大家进行补充。
>
> data[x:y:z]
>
> data[::-1]
>
> data[:-1]
>
> data[-3:]
>
> data[1:4, 2:4]
>
> data[..., 1:2]
>
> data[:, None]

这是最近在看的代码，进行loss的计算，用到了很多切片的知识，所以进行总结，希望看完这篇博客以后大家也能看懂类似的代码

```python
def compute_loss(p, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lxy, lwh, lcls, lconf = ft([0]), ft([0]), ft([0]), ft([0])
    # build_targets对targets向量进行处理
    txy, twh, tcls, indices = build_targets(model, targets)

    # Define criteria
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()  # (weight=model.class_weights)
    BCE = nn.BCEWithLogitsLoss()

    # Compute losses
    h = model.hyp  # hyperparameters
    bs = p[0].shape[0]  # batch size
    k = bs  # loss gain
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tconf = torch.zeros_like(pi0[..., 0])  # conf

        # Compute losses
        if len(b):  # number of targets
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            tconf[b, a, gj, gi] = 1  # conf
            # pi[..., 2:4] = torch.sigmoid(pi[..., 2:4])  # wh power loss (uncomment)

            lxy += (k * h['xy']) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
            lwh += (k * h['wh']) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss
            lcls += (k * h['cls']) * CE(pi[..., 5:], tcls[i])  # class_conf loss

        # pos_weight = ft([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * h['conf']) * BCE(pi0[..., 4], tconf)  # obj_conf loss
    loss = lxy + lwh + lconf + lcls

    return loss, torch.cat((lxy, lwh, lconf, lcls, loss)).detach()
```

[TOC]

## 1. 单列表切片

对于一个普遍形式：data[x:y:z] 讲一下他的意义：

> The syntax `[x:y:z]` means "take every `z`th element of a list from index `x` to index `y`". When `z` is negative, it indicates going backwards. When `x` isn't specified, it defaults to the first element of the list in the direction you are traversing the list. When `y` isn't specified, it defaults to the last element of the list. So if we want to take every 2th element of a list, we use `[::2]`.

-- from github:python-is-cool

翻译过来就是：x是起始坐标，y是结束坐标（不包含），z是代表每隔z个数取一个数,如果z是负数代表方向是从后往前。

看一个例子：

```python
>>> data = list(range(10))
>>> data
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> data[1:5] # 代表[1,5)
[1, 2, 3, 4]
>>> data[-3:] # 代表从倒数第3个到最后， 方向默认还是从前往后
[7, 8, 9]
>>> data[:-3] # 代表从第1个到倒数第4个，方向还是默认从前向后
[0, 1, 2, 3, 4, 5, 6]
>>> data[1:8:2] # 从[1,2,3,4,5,6,7]中每两个选一个，得到[1,3,5,7]
[1, 3, 5, 7]
>>> data[1:8:-2] # 注意如果z是负数，那么前边也需要是负数，方向需要正确
[]
>>> data[::-2] # 两个冒号代表取全体
[9, 7, 5, 3, 1]
>>> data[-1:-8:-2] 
# 注意data[-1:-8:-1]=[9,8,7,6,5,4,3]意思是倒数第1个到倒数第7个数
# 这里的x,y依然意义是起始和结尾
[9, 7, 5, 3]
>>>
```

总结：

1. 注意起始和结尾的[x,y] 实际的数学意义是[x,y), 前取到，后取不到

2. 注意z是负数的情况下，整个数列方向是从后往前数的，对应的[x,y]也应该是负数，从后往前，这里的x,y意义依然是起始和结尾。

3. 双冒号`::`代表取全部

4. 注意整个数列方向问题，默认都是从前往后，只有z的正负性能控制方向。

5. 只有一个冒号的情况是两个冒号的特殊情况，省略了最后一个冒号，如：

    ```python
    >>> data[3:]
    [3, 4, 5, 6, 7, 8, 9]
    >>> data[3::1] # 默认方向从前往后，且z的值为1
    [3, 4, 5, 6, 7, 8, 9]
    >>>
    ```

    

## 2. 多维列表

> ```python
> data[..., 1:2] 
> data[1:2, 2:4, 5:9]
> data[:,None]
> data[1:2, 2:4] == data [1:2][2:4] ? 
> data[:]
> ```


干货：

1. 对于三个点的情况, 代表前边所有维度
2. 对于逗号分隔的情况，每个逗号将以上一个单独的分割开来
3. None代表增加一个维度
4. data[1:2, 2:4] 与 data [1:2] [2:4] 并不相等 
5. data[:]代表取全体数据

举例子：

- data[..., 1:2] 举例

```python
>>> import numpy as np
>>> data1 = np.arange(27).reshape(3,3,3)
>>> data1
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>> data1[...,1:3]
array([[[ 1,  2],
        [ 4,  5],
        [ 7,  8]],

       [[10, 11],
        [13, 14],
        [16, 17]],

       [[19, 20],
        [22, 23],
        [25, 26]]])
>>> data1[:,:,1:3]
array([[[ 1,  2],
        [ 4,  5],
        [ 7,  8]],

       [[10, 11],
        [13, 14],
        [16, 17]],

       [[19, 20],
        [22, 23],
        [25, 26]]])
# 可以看出data1[:,:,1:3]与data1[...,1:3]是等价的，三个点的存在简化了前边维度，
# 不管多少维度存在都可以使用，取代前边所有维度。
```

- data[1:2, 2:4, 5:9]

```python
>>> import numpy as np
>>> data1 = np.arange(27).reshape(3,3,3)
>>> data1[0:2]
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> data1[0:2,0:2]
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 9, 10, 11],
        [12, 13, 14]]])
>>> data1[0:2,0:2,0:2]
array([[[ 0,  1],
        [ 3,  4]],

       [[ 9, 10],
        [12, 13]]])
# 一步一步的切片可以看出，每一步都是单独的一个单列表切片，满足单列表切片原则
```

- data[:,None]

```python
>>> data1.shape
(3, 3, 3)
>>> data1[:,None].shape
(3, 1, 3, 3)
>>> data1[...,None].shape
(3, 3, 3, 1)
>>> data1[None,...].shape
(1, 3, 3, 3)
# None代表增加一个维度,通常使用在数据预处理，给图片添加batch列
```

- data[1:2, 2:4] == data[1：2] [2:4] 

```python
>>> import numpy as np
>>> data1 = np.arange(27)
>>> data1
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
>>> data1[1:7][3:6] # 两个可以理解为切片的切片
array([4, 5, 6])
>>> data1[1:7,3:6] # 只有一维，所以无法运行
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: too many indices for array
>>> data1 = np.arange(27).reshape(3,3,3)
>>> data1
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>> data1[0:2]
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> data1[0:2][1:3]
# 切片的切片， 且过一次以后就只有0,1两个维度了，所以只能取第一维
array([[[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
>>> data1[0:2,1:3] # 所有满足第一维为0,1 第二维为1,2的
array([[[ 3,  4,  5],
        [ 6,  7,  8]],

       [[12, 13, 14],
        [15, 16, 17]]])
>>>
```

- data[:] 表示取全体

```python
>>> data1[:]
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>> data1[:,:,:]
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
>>>
```

---

> 后记：这篇博客总结了目前常见到的问题，在深度学习中经常用到，tensor的切片常常让人有点微微头疼，所以特意总结一下，大家遇见新的切片方式可以在评论中补充，有问题的也可以提出。之后还有一些张量操作会单独出一篇博客。