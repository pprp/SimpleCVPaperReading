---
title:  "opencv python使用方法示例代码以及总结"
date:   2019-05-13 
tags: opencv
categories: ImageProcess
---

> opencv python的使用
>
> 简单的图像处理
>
> 持续更新

## 1. 需要安装的库

- opencv-python
- numpy 
- matplotlib

```
pip install opencv-python numpy matplotlib
```

使用：

```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
```

IDE选择：

​	建议选择pycharm, 可以进行调试

## 2. opencv3图像处理

**读取图片：**

```
cv2.imread(filename,flags)
```

支持图像格式，常用的有：jpg,bmp,png,tiff等

flags: color type

- flags < 0 : 保存原始三通道以及alpha通道（8位用于存储透明度信息）
- flags = 0 : 加载灰度图
- flags > 0 : 加载三通道彩色图，忽略alpha通道

**示例：**

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('test.jpg',1)
cv2.imshow('title', img)

plt_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()
```

> ps: 即使图片路径是错的，那么图像返回一个None，所以最好进行一下判断

**图片保存：**

```python
cv2.imwrite('outpath/filename.jpg',img)
```

**图片属性,分割：**

```
img_h, img_w, img_d = img.shape
cuttedImg = img[10:100,20:500,:] #使用的是numpy的slice
```

**图像空间转化：**

```
gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.COLOR_BGR2GRAY
```

<https://mp.weixin.qq.com/s/AFmGAZ9Ju6fFNd6vkd_E0w>