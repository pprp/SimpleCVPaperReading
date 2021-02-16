# Sphinx 快速构建工程文档

[TOC]

## 一、 ReStructuredText 语法

介绍：reStructuredText 是一种易于阅读、所见即所得的纯文本标记语言，常被用于编写行内文档，快速创建简单网页，或者作为独立文档存在。

——David Goodger

rst可以转为html,html5,latex,xetex,xml等格式。

**标题：**

规定一级目录 ##########

规定二级目录 ===========

规定三级目录 --------------------

规定四级目录 ```````````

要求长度大于等于标题长度

**引用：**

用4个空格或者制表符来表示引用。

​    test 
​    test
​        test 

**列表：**

有序列表和markdown一样

1. read a book
2. write a summary
3. close the book

无序列表也是一样的

- read a book
- write a summary
- close the book

**代码：**

```rst
.. code:: python

    import sys
    print(sys.version)
    
```

..代表开始一个rst特定的命令了，具体命令要看后边的名称，比如这里的code代表这是代码块，后边的两个冒号也是固定语法，后边接具体参数 python。

> 特别注意：
>
> - 命令以后要和内容分成两部分，中间要添加空行；
> - 具体内容结束以后必须添加一个空行；

**分割线：**

与markdown类似：

```
------------
```

**链接语法：（特）**

参考式：

```rst
欢迎访问 pprp的github_ 官方主页

.. _pprp的github: https://github.com/pprp
```

> 特别注意：
>
> - 文中下划线以前的部分必须前后空格隔开，代表其是一个整体
> - 链接对象要在最后注明，关键点是需要以下划线开头，内容和上述一致，冒号后边必须有一个空格。
> - 出现多个词组或者中文，需要用`括住，比如： 
> 
```rst
欢迎访问 `pprp github`_ 官方主页

.. _`pprp github`: https://github.com/pprp
```

**自动标题连接跳转：**

比如我有以下几个标题：

```
1. how to be rich
2. how to marry a rich man
3. you can not be rich
```

我想引用第二个部分，我应该这么写：

```
`2. how to marry a rich man`_
```

**图片：**

```
.. image:: /images/nikola.png
   :align: center
   :width: 200px
   :height:150px
```

通过命令可以灵活控制图片的设置，image换成figure也可以。

:target: 可以实现在点击图片的时候，跳转到另外一个链接，或者点击缩略图查看原图的效果。

**脚注**

```rst
就像这样创建一个脚注 [#]_ 。

.. [#] 这里是 **脚注** 的 *文本* 。
```

**目录：**

Tables of Contents

```
.. contents:: 文档目录
```

还有很多参数，比如:depth:设置目录展示的最大深度

**公式：**

```rst
.. math::

   \alpha _t(i) = P(O_1, O_2, \ldots  O_t, q_t = S_i \lambda )
```



## 二、Sphinx使用

**安装：**

```
pip install sphinx 
pip install sphinx_rtd_theme
```

**创建：**

在一个文件夹下，使用sphinx-quickstart命令，根据提问回答问题，大部分都是回答y,得到四个文件：

![](https://img-blog.csdnimg.cn/2021021617190723.png)


- build目录：运行make命令后，生成的文件都在这个目录里面,包括html文件
- source目录：放置文档的源文件，我们编写的内容在这里
- make.bat：批处理命令，不用管
- makefile 

**修改：**

source/conf.py文件，修改html theme主题：

```python
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
```

source/index.rst文件是用来建立文档结构树，将其他文件文件名放在这个文件，其他文件放置的位置在source文件夹下，如果用到相对位置，要加上路径。

**生成：**

在刚刚构建的文件夹下，输入命令，`make html`, 开始生成，在build/html文件中找到index.html打开，就是文件结果。



## 三、工具

markdown 转 ReStructuredText网址： https://cloudconvert.com/md-to-rst

在线渲染：http://rst.ninjs.org/

文档：https://docutils.sourceforge.io/docs/user/rst/quickref.html

根据python文件中注释生成帮助文档：docstring

参考：https://zhuanlan.zhihu.com/p/264647009