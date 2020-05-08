---
title: python包以及模块引用梳理
date: 2019-06-20 14:54:46
tags: 
- python
categories:
- 深度学习
---

> 前言：在查看一些工程性代码的时候，总是会发现类似：
>
> ```python
> from .util import utils
> from . import datasets
> ```
>
> 这样的引用语句, 这让我比较困惑，所以趁这次机会，花点时间，好好整理一下相关的知识



## 1. python包机制

```
- 包
- 模块
- 框架： 如何组织包和模块
```

python提供了包的概念，是一个有层次的文件目录结构，用于管理多个模块源文件。

- 包就是文件夹，这个文件夹下有\_\_init\_\_.py文件，并且该文件夹可以包括其他模块
- 多个相关联的模块组成一个包，以便于维护和使用，同时能有限的避免命名空间的冲突。

- 在导入一个包的时候，会先调用这个包的\_\_init\_\_.py文件

层次问题：从小到大

- 语句
- 函数 def
- 类 class
- 模块 module， 物理上是一个python文件
- 包 package， 物理上是一个文件夹， 包中可以含有模块和包

包和模块的作用：

> - 编写好轮子，工具类，供其他模块进行使用
>
> - 有效地对程序进行分解，方便代码的管理和维护
> - 防止同一模块内命名重复的问题(module1.name, module2.name)

-- from bilibili <https://www.bilibili.com/video/av65157512?p=2>

包和模块基本信息

- 包和模块名称
    - `__name__` 
    -  `__package__`
- 存放位置
    - `__file__`
- 查看包和模块的内容
    - dir(os)
    - `__dict__`

导入包/模块的方式：

- 常规导入：
    - import M
        - 同级目录直接导入import M
        - 不同级目录使用点语法import pkg1.pkg2.M
    - import M1, M2
    - import M as m
    - from A import B as b, C as c
        - A 的范围要大于B,C
            - 范围：包>模块>资源
        - B，C这部分要尽可能简单
            - 正确：from A.A1 import aa 
            - 错误：from A import A1.aa
    - from 模块 import *
        - 那么会去该模块对应的python文件中找`__all__`变量对应的内容
    - from 包 import *
        - 那么就会去该包对应的`__init__.py`文件中找`__all__`变量对应内容

- 注意：

    - 使用时，导入的对象应该是模块，而不是包

    - 如果导入的是一个包，默认不会导入任何模块

    - 解决方案：

        - 在该包中的`__init__.py`中导入所有需要的模块

        - 以from 包/文件夹 import 模块/子包 的形式导入

            - from p1 import Tool1 as t1, Tool2 as t2
            - from p1.sub_p import sub_xx
         - from模块 import 资源名 的形式导入


            - from other import num

- 遇到no module named xxx

    - ```python
        import sys
        sys.path.append('rc:\Desktop\xxx_dir')
        import xxx
        # 比较强硬的解决方法
        ```

- 模块查找顺序：

    - 内建> 自定义> sys.path

## 2. python模块相对引用

很多时候会遇见以下错误：

```
ValueError: attempted relative import beyond top-level package
```

这通常是由于相对引用的使用而出现的问题。

需要明确：

1. 相对引用不是相对于文件结构！！
2. 相对引用是相对于`__name__`

举个例子：

```python
- rootdir
  - subdir1
    - __init__.py
  	- subfile1.py
  - subdir2 
    - __init__.py
    - subfile2.py
- test.py
```

test.py中调用subfile1.py的内容：

```python
def print_mod1():
    print('__name__: {}'.format(__name__))
    print('__package__: {}'.format(__package__))
    print('Import Successfully!')
```

输出为：

```
__name__: subdir1.subfile1
__package__: subfile1
Import Successfully!
```

所以这个相对位置就是相对于`__name__`变量，比如：

一个点：`.` 就代表当前是subdir1

两个点：`..`就不存在，就会报错`beyond top-level package`, 这里的top-level package 也很容易理解，那就是当前的subdir1。

更多内容可以查看：<https://www.cnblogs.com/jay54520/p/8438228.html>

## 3. 举例

目录结构如下：

![1572056109473](1572056109473.png)

1. model文件夹下：

`__init__.py` ·文件内容如下：

```python
print('-'*5,"init for model folder",'-'*5)
```



`models.py`文件内容如下：

```python
def mm():
    print("this is rootpkg/model/models/model")
    print('__name__: {}'.format(__name__))
    print('__package__: {}'.format(__package__))
```



2. src文件夹下：

`__init__.py` ·文件内容如下：

```python
print('-'*5,"init for src folder",'-'*5)
```



`source.py`文件内容如下：

```python
def ss():
    print("this is rootpkg/src/source/src")
    print('__name__: {}'.format(__name__))
    print('__package__: {}'.format(__package__))
```



3. RootPkg文件夹下：

`__init__.py` ·文件内容如下：

```python
print('-'*5,"init for rootpkg folder",'-'*5)
```



`main.py`文件内容如下：

```python
from model import models
from src import source
from 模块/文件夹 import 具体某个python文件名

source.ss()
models.mm()
```



4. 运行结果如下：

    ```python
    ----- init for model folder -----  
    ----- init for src folder ----- 
    # 这两个是在import模块的时候执行的__init__.py文件
    this is rootpkg/src/source/src
    __name__: src.source
    __package__: src
    this is rootpkg/model/models/model
    __name__: model.models
    __package__: model
    ```

    运行完以后会出现`__pycache__`文件夹

    ![1572056618738](1572056618738.png)

运行成功，但是Tommy-Yu的最佳实践那部分没有直行通过，大家可以查看一下第一个reference的博客，如果有谁能跑通，欢迎联系我。（ps: 个人感觉这个博客没有讲的很清楚，实际运行确实会出错）







---

## reference

- <https://www.cnblogs.com/Tommy-Yu/p/5794829.html?spm=a2c4e.10696291.0.0.289619a41CBdwB>

- <https://www.cnblogs.com/jay54520/p/8438228.html>

- <https://www.bilibili.com/video/av65157512?from=search&seid=6677976813026578695>