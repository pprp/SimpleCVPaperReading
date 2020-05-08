---
title: 使用python多线程加载模型并测试
date: 2019-11-14 15:16:17
tags:
- python
- threading
- 测试
---

> 前言：之前只用过单线程处理，加载模型进行测试，运行时间上也可以接受。但是现在需要处理比较大量的数据，如果进行线性处理，可能测试一次就花10个小时，而且还不知道结果怎么样，所以多线程就必须使用上了。有关线程部分主要参考：<https://junyiseo.com/python/211.html>

## 1 多线程

多线程类似于同时执行多个不同程序，线程在执行过程中与进程还是有区别的。每个独立的进程有一个程序运行的入口、顺序执行序列和程序的出口。但是线程不能够独立执行，必须依存在应用程序中，由应用程序提供多个线程执行控制。

特点：

- 线程可以被抢占（中断）。
- 在其他线程正在运行时，线程可以暂时搁置（也称为睡眠） -- 这就是线程的退让。

应用场景：

- 使用线程可以把占据长时间的程序中的任务放到后台去处理。
- 用户界面可以更加吸引人，这样比如用户点击了一个按钮去触发某些事件的处理，可以弹出一个进度条来显示处理的进度
- 程序的运行速度可能加快
- 在一些等待的任务实现上如用户输入、文件读写和网络收发数据等，线程就比较有用了。在这种情况下我们可以释放一些珍贵的资源如内存占用等等。

> 以上内容来自：<https://www.runoob.com/python/python-multithreading.html>
>
> 更多多线程python知识可以访问以上网站

## 2 使用多线程进行多模型加载和测试

- 先说一下如何分配多线程执行的思路：

    - 由于单线程进行测试的时候是一张图像进一次网络，然后得到结果。其占用的显存很小，只有900MB左右，而这里使用的是11G显存,所以在这种条件下支持每一个线程分配一个模型，然后进行预测。

    - 然后就是数据分配问题，多线程常常会遇到访问数据冲突的问题，但是这里我们可以避开这个问题，是用一个List存储所有图片，然后根据长度分配每个线程所要处理的图片多少。
    - 剩下就可以看模板了。

- 这里提供一个模板，可以替换其中关键的测试图片的函数，然后就可以运行了。

```python
# -*- coding: UTF-8 -*-

import threading
from time import sleep, ctime

import cv2
import os
import json

totalThread = 16  # 需要创建的线程数，可以控制线程的数量

config_file = '模型配置文件'
checkpoint_file = '模型权重文件'
test_data_dir = '测试集所在文件夹（里边是待测试图片）'

listImg = [file for file in os.listdir(test_data_dir)]  #创建需要读取的列表，可以自行创建自己的列表
lenList = len(listImg)  #列表的总长度
gap = int(lenList / totalThread)  #列表分配到每个线程的执行数

# 按照分配的区间，读取列表内容，需要其他功能在这个方法里设置
def processSection(name, s, e):
    for i in range(s, e):
        processImg(name, listImg[i])


def processImg(name, file):
    # 这个部分内容包括：
    # 1. 加载模型
    # 2. 根据file读取图片
    # 3. 将结果进行处理并进行保存
    print("Thread %s: have processed %s" % (name, filename))
    print(os.path.join('\t resultData', filename + '.json'), end="")
    print(" Length of json: %d" % len(final_list))


class myThread(threading.Thread):

    def __init__(self, threadID, name, s, e):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.s = s
        self.e = e

    def run(self):
        print("Starting " + self.name + ctime(), end="")
        print(" From %d to %d" % (self.s, self.e))
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定
        # 否则超时后将返回False
        # 这里由于数据不存在冲突情况，所以可以注释掉锁的代码
        # threadLock.acquire()
        #线程需要执行的方法
        processSection(self.name, self.s, self.e)
        # 释放锁
        # threadLock.release()



threadLock = threading.Lock()  #锁
threads = []  #创建线程列表

# 创建新线程和添加线程到列表
for i in range(totalThread):
    thread = 'thread%s' % i
    if i == 0:
        thread = myThread(0, "Thread-%s" % i, 0, gap)
    elif totalThread == i + 1:
        thread = myThread(i, "Thread-%s" % i, i * gap, lenList)
    else:
        thread = myThread(i, "Thread-%s" % i, i * gap, (i + 1) * gap)
    threads.append(thread)  # 添加线程到列表

# 循环开启线程
for i in range(totalThread):
    threads[i].start()

# 等待所有线程完成
for t in threads:
    t.join()
print("Exiting Main Thread")
```

结果：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.31       Driver Version: 440.31       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:01:00.0 Off |                  N/A |
| 46%   61C    P2   206W / 250W |  10238MiB / 11016MiB |     68%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      4616      C   python                                      9935MiB |
|    0     11667      G   /usr/lib/xorg/Xorg                           195MiB |
|    0     30334      G   compiz                                        94MiB |
+-----------------------------------------------------------------------------+
```

加载了16个模型，同时进行处理：

```

Starting Thread-0Thu Nov 14 15:09:53 2019 From 0 to 3
Starting Thread-1Thu Nov 14 15:09:53 2019 From 3 to 6
Starting Thread-2Thu Nov 14 15:09:53 2019 From 6 to 9
Starting Thread-3Thu Nov 14 15:09:53 2019 From 9 to 12
Starting Thread-5Thu Nov 14 15:09:53 2019 From 15 to 18
Starting Thread-4Thu Nov 14 15:09:53 2019 From 12 to 15
Starting Thread-6Thu Nov 14 15:09:53 2019 From 18 to 21
Starting Thread-7Thu Nov 14 15:09:53 2019 From 21 to 24
Starting Thread-9Thu Nov 14 15:09:53 2019 From 27 to 30
Starting Thread-10Thu Nov 14 15:09:53 2019 From 30 to 33
Starting Thread-12Thu Nov 14 15:09:53 2019 From 36 to 39
Starting Thread-13Thu Nov 14 15:09:53 2019 From 39 to 42
Starting Thread-11Thu Nov 14 15:09:53 2019 From 33 to 36
Starting Thread-14Thu Nov 14 15:09:53 2019 From 42 to 45
Starting Thread-8Thu Nov 14 15:09:53 2019 From 24 to 27
Starting Thread-15Thu Nov 14 15:09:53 2019 From 45 to 50
```

---

> 后记：主要提供了一个模板进行多模型加载，但是如果一个模型就很大的情况下，这种就明显不合适了。可以想到的是一次从多个batch进行测试，然后记录结果。其他方法大佬可以分享留言。