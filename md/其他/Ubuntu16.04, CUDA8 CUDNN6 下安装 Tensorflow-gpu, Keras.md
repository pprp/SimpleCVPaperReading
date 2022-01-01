---
title: Ubuntu16.04, CUDA8 CUDNN6 下安装 Tensorflow-gpu, Keras, Pytorch, fastai
date: 2019-07-16 14:54:46
tags: 
- ubuntu16.04,
- cuda 
- cudnn
- kears
- tensorflow
categories:
- 环境配置
---

## 如何访问tensorflow官方网站

tensorflow官方网站变为：<https://tensorflow.google.cn/>

## 安装深度学习框架

### 0. ubuntu查看CUDA和cuDNN版本

CUDA:  

```
cat /usr/local/cuda/version.txt
```

cuDNN:

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

### 1. keras


```
tensorflow 1.5 和 keras 2.1.4搭配
tensorflow 1.4 和 keras2.1.3搭配
tensorflow 1.3 和keras 2.1.2 搭配
tensorflow 1.2 和keras 2.1.1搭配
```



| 版本                        | Python 版本  | 编译器  | 编译工具     | cuDNN             | CUDA                              |
| --------------------------- | ------------ | ------- | ------------ | ----------------- | --------------------------------- |
| tensorflow_gpu-2.0.0-alpha0 | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.19.2 | 7.4.1以及更高版本 | CUDA 10.0 (需要 410.x 或更高版本) |
| tensorflow_gpu-1.13.0       | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.19.2 | 7.4               | 10                                |
| tensorflow_gpu-1.12.0       | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.15.0 | 7                 | 9                                 |
| tensorflow_gpu-1.11.0       | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.15.0 | 7                 | 9                                 |
| tensorflow_gpu-1.10.0       | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.15.0 | 7                 | 9                                 |
| tensorflow_gpu-1.9.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.11.0 | 7                 | 9                                 |
| tensorflow_gpu-1.8.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.10.0 | 7                 | 9                                 |
| tensorflow_gpu-1.7.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.9.0  | 7                 | 9                                 |
| tensorflow_gpu-1.6.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.9.0  | 7                 | 9                                 |
| tensorflow_gpu-1.5.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.8.0  | 7                 | 9                                 |
| tensorflow_gpu-1.4.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.5.4  | 6                 | 8                                 |
| tensorflow_gpu-1.3.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.4.5  | 6                 | 8                                 |
| tensorflow_gpu-1.2.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.4.5  | 5.1               | 8                                 |
| tensorflow_gpu-1.1.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.4.2  | 5.1               | 8                                 |
| tensorflow_gpu-1.0.0        | 2.7、3.3-3.6 | GCC 4.8 | Bazel 0.4.2  | 5.1               | 8                                 |

本地环境安装的是CUDA8和CUDNN5，所以采用以下命令：

```
pip install tensorflow-gpu==1.2.0 -i https://mirrors.aliyun.com/pypi/simple
pip install keras==2.1.1
```

### 2. fastai

安装比较人性化：

```
pip install fastai
```

测试：

```python
import fastai
import torch
```

如果没有报错就说明正常，一般配合jupyter notebook进行使用，比较方便。

```
conda install jupter 
```

## 报错解决

1. TypeError: softmax

```
TypeError: softmax() got an unexpected keyword argument 'axis'
```

当前keras版本是2.2 退回到2.1 

```
pip install keras==2.1
```

2. TypeError: validation_split

```
TypeError: __init__() got an unexpected keyword argument 'validation_split'
```

将validation_split删除

3. TypeError: fit_generator() missing 1 required positional argument: 'steps_per_epoch'

```
TypeError: fit_generator() missing 1 required positional argument: 'steps_per_epoch'
```

添加上steps_pers_epoch参数，这是一个必要参数，但是不同版本keras要求不太一样。

4. RemoveError: 'setuptools' is a dependency of conda and cannot be removed from

```
conda update conda
```

5. jupyter notebook no module named xxx

```
which jupyter # 查看使用的是哪个jupyter，通常情况下这种情况出现一般用的是系统的jupyter而不是anaconda中的jupyter
```

通过以上分析可以得到解决方案是使用以下命令：

```
conda install jupyter
```

再次查看使用的是哪个jupyter

```
which jupter
```

如果发现使用的是anaconda中的路径那就说明成功了，问题解决。

6. ValueError: `validation_steps=None` is only valid for a generator based on the `keras.utils.Sequence` class. Please specify `validation_steps` or use the `keras.utils.Sequence` class.












