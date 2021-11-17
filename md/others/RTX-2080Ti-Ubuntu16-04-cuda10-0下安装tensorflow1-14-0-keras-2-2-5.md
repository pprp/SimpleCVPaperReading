---
title: 'RTX 2080Ti, Ubuntu16.04, cuda10.0下安装tensorflow1.14.0, keras 2.2.5'
date: 2019-11-15 20:20:09
tags:
- 深度学习环境
- cuda
- tensorflow
- keras
---

> 前言：朋友让我帮他装一下tensorflow,keras的gpu环境，之前明明一句话就能搞定的，所以我就直接pip install了，但很明显这样不行。默认安装的tensorflow是2版本的，之前的代码都是基于1的所以还得再找找教程。现在去网上找一下，还有很多教程都是基于cuda8的，pip源里边已经找不到1.2，1.6这种版本的tensorflow了。所以找了半天，终于找到靠谱的，在这里进行分享。

本教程是安装**RTX 2080Ti, tensorflow1.14.0, keras 2.2.5，Ubuntu16.04, cuda10.0**
**1. 安装成功的关键在哪？**
		关键在于版本匹配问题，必须找到匹配的版本才能正常运行，否则会出各种问题，不过前提是你要安装号cuda和cudnn，具体安装可以看我之前写的一篇：<https://blog.csdn.net/DD_PP_JJ/article/details/103055629>, 写的很详细。
**2. 去哪里找版本的匹配？**
	<https://docs.floydhub.com/guides/environments/>
**3. 在cuda10.0下进行安装**：
经过查询版本匹配问题，选择tensorflow1.14进行安装，安装命令为：
	
```
pip install tensorflow-gpu==1.14
```
然后查询keras版本，发现2.2.5是比较合适的keras版本，继续安装：

```
pip install keras==2.2.5
```
**4. 测试是否安装成功GPU版本?**
输入python命令，然后输入以下代码进行测试：
```
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```
如果可以得到一堆输出，就说明已经配置好GPU版本的tensorflow和keras了。

```
>>> from keras import backend as K
Using TensorFlow backend.
>>> K.tensorflow_backend._get_available_gpus()
WARNING:tensorflow:From /home/user-hlw/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
WARNING:tensorflow:From /home/user-hlw/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:197: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
WARNING:tensorflow:From /home/user-hlw/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:203: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
2019-11-15 15:50:41.358531: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-11-15 15:50:41.374301: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2399995000 Hz
2019-11-15 15:50:41.379576: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564dbea39c60 executing computations on platform Host. Devices:
2019-11-15 15:50:41.379607: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): <undefined>, <undefined>
2019-11-15 15:50:41.383516: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcuda.so.1
2019-11-15 15:50:42.396020: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564dbeaf38a0 executing computations on platform CUDA. Devices:
2019-11-15 15:50:42.396092: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2019-11-15 15:50:42.396108: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (1): GeForce RTX 2080 Ti, Compute Capability 7.5
2019-11-15 15:50:42.396120: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (2): GeForce RTX 2080 Ti, Compute Capability 7.5
2019-11-15 15:50:42.396130: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (3): GeForce RTX 2080 Ti, Compute Capability 7.5
2019-11-15 15:50:42.398726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 0 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:02:00.0
2019-11-15 15:50:42.400394: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 1 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:03:00.0
2019-11-15 15:50:42.401969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 2 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:82:00.0
2019-11-15 15:50:42.403479: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1640] Found device 3 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:83:00.0
2019-11-15 15:50:42.403989: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2019-11-15 15:50:42.406544: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcublas.so.10.0
2019-11-15 15:50:42.408770: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcufft.so.10.0
2019-11-15 15:50:42.409258: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcurand.so.10.0
2019-11-15 15:50:42.411924: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusolver.so.10.0
2019-11-15 15:50:42.413938: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcusparse.so.10.0
2019-11-15 15:50:42.419643: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudnn.so.7
2019-11-15 15:50:42.430706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1763] Adding visible gpu devices: 0, 1, 2, 3
2019-11-15 15:50:42.430769: I tensorflow/stream_executor/platform/default/dso_loader.cc:42] Successfully opened dynamic library libcudart.so.10.0
2019-11-15 15:50:42.437215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1181] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-11-15 15:50:42.437241: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1187]      0 1 2 3
2019-11-15 15:50:42.437250: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 0:   N N N N
2019-11-15 15:50:42.437256: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 1:   N N N N
2019-11-15 15:50:42.437262: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 2:   N N N N
2019-11-15 15:50:42.437270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1200] 3:   N N N N
2019-11-15 15:50:42.443910: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10165 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:02:00.0, compute capability: 7.5)
2019-11-15 15:50:42.450433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 10165 MB memory) -> physical GPU (device: 1, name: GeForce RTX 2080 Ti, pci bus id: 0000:03:00.0, compute capability: 7.5)
2019-11-15 15:50:42.453357: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 10165 MB memory) -> physical GPU (device: 2, name: GeForce RTX 2080 Ti, pci bus id: 0000:82:00.0, compute capability: 7.5)
2019-11-15 15:50:42.455533: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:3 with 10165 MB memory) -> physical GPU (device: 3, name: GeForce RTX 2080 Ti, pci bus id: 0000:83:00.0, compute capability: 7.5)
WARNING:tensorflow:From /home/user-hlw/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

['/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1', '/job:localhost/replica:0/task:0/device:GPU:2', '/job:localhost/replica:0/task:0/device:GPU:3']
>>>

```
**5. 如果想要安装tensorflow2呢？** 
参考tensorflow官方的文档进行安装：<https://tensorflow.google.cn/install/pip>

> 后记：安装这些深度学习库的时候要注意版本对应关系，否则会浪费很长时间，浪费很多流量。