---
title: python编码规范
date: 2019-11-08 14:54:46
tags: 
- python
- 编码规范
categories:
- python
---



# python 代码规范

> 前言：之前写python代码感觉注释不规范，冗余性比较大，而且格式不规范，之后再重新理解的时候会很难理解，所以搜索了一些代码规范进行总结，今后写的代码尽量按照规范进行编码。



## 1 通用规范

- 编码

    - 头部需要加入`#-*-coding:utf-8-*-#`

    -  尽可能使用‘is’‘is not’取代‘==’，比如if x is not None 要优于if x

    - 使用startswith() and endswith()代替切片进行序列前缀或后缀的检查。

        `if foo.startswith('bar') `优于`if foo[:3] == ‘bar'`

    - 判断序列空或不空，有如下规则

        ```python
        Yes: if not seq:
        if seq:
        优于
        No: if len(seq)
        if not len(seq)
        ```

        

- 缩进

    - 统一使用4个空格，不能使用tab

- 引号： 自然语言使用双引号，机器标识使用单引号，代码里多数使用单引号

    - 自然语言使用双引号`“...”`例如raise得到的错误信息，使用unicode, u“代码有误”

    - 机器标识使用单引号：`‘key’`

        ```python
        dict={}
        dict['key'] = 'value'# key 使用单引号
        ```

    - 正则表达式使用原生双引号`r"..."`

    - 文档字符串 docstring, 使用三个双引号 `“""..."”"`

## 2 空行

- 空行

    - 模块级函数和类定义之间空两行

        ```python
        def test1(txt):
            print(txt)
        
        
        def test2(txt):
            print(txt)
            
            
        class A:
        	
        	def __init__(self):
        	    pass
        ```

    - 类成员函数之间空一行：

        ```python
        class A:
            
            def __init__(self):
                pass
            
            def hello(self):
                pass
            
        
        def main():
            pass
        ```

    - 可以使用多个空行分割多组相关函数

    - 函数中使用空行分割逻辑相关的代码

## 3 import 规范

- import 语句

    - 尽量分行书写

        ```python
        import os
        import shutil
        import sys
        ```

    - import 语句尽量使用绝对引用，而不要使用相对引用

        ```python
        from pkg import func1# 推荐
        from ..pkg import func2# 不推荐
        ```

    - import 语句应该在文件头部， 至于模块说明之后，在全局变量之前

    - import语句按照顺序排列，按照功能逻辑使用空行分隔。

    - 如果发生命令冲突，可以使用命名空间

        ```python
        import bar
        import foo.bar
        
        bar.Bar()
        foo.bar.Bar()
        ```

## 4 空行空格

- 空格

    - 两元运算符两边各一个空格

        ```python
        i = i + 1
        sumbited += 1
        x = x * 2 + 1
        c = (a - b) * (a + b)
        ```

    - 参数列表中的逗号后要有空格

        ```python
        def complex(real, img):
            pass
        ```

    - 参数列表默认值等号两边不要加空格

        ```python
        def complex(real, img=0.1)
            pass
        ```

- 换行

    - python支持括号内的换行

        - 第二行缩进到括号的起始处

            ```python
            foo = func(var1, var2,
                       var3, var4)
            ```

        - 第二行缩进四个空格，括号其实就换行

            ```python
            def func(
                var1, var11,
                var2, var12, 
                var3, var13,
                var4):
                pass
            ```

## 5 docstring与注释

- docstring

    - 所有公共模块，函数，方法，类，都要写docstring, 私有方法不需要，但是应该在def后提供一个块注释来说明

    - docstring 的结束`“”“`应该独占一行, 除非只有一行

        ```python
        """
        multi-line
        docstring
        """
        
        """one line docstring"""
        ```

- 注释

    - 块注释，空行同样也需要#

    - 行注释,  使用两个空格与语句分开，并且不要使用无意义注释

    - 在比较复杂的代码部分尽量写注释

    - 比较重要的注释端，使用多个等号隔开，更加醒目

        ```python
        # =====================================
        # annotations!
        # =====================================
        ```

    - 可以使用vscode中使用autodocstring插件，然后快捷键`ctrl+shift+2`进行扩充

        ```python
        def testb(es):
            """[summary]
            
            Arguments:
                es {[type]} -- [description]
            """
            print(es)
            
        """[summary]
        """
        class A(Object):
            """[summary]
            """
            def __init__(self, b):
                """[summary]
                
                Arguments:
                    Object {[type]} -- [description]
                    b {[type]} -- [description]
                """
                pass
        ```

    - TODO注释：

        TODO注释应该在所有开头处包含"TODO"字符串, 紧跟着是用括号括起来的你的名字, email地址或其它标识符. 然后是一个可选的冒号. 接着必须有一行注释, 解释要做什么. 主要目的是为了有一个统一的TODO格式, 这样添加注释的人就可以搜索到(并可以按需提供更多细节). 写了TODO注释并不保证写的人会亲自解决问题. 当你写了一个TODO, 请注上你的名字.

        ```python
        # TODO(kl@gmail.com): Use a "*" here for string repetition.
        # TODO(Zeke) Change this to use relations.
        ```

        

## 6 命令规范

- 模块

    - 模块尽量使用小写开头，不要用下划线`import core`

- 类名

    - 尽量使用驼峰命令风格，首字母大写，私有类使用下划线开头

        ```python
        class DogCat():
            pass
        
        class _Action():
            pass
        ```

    - 将相关的类和顶级函数放在同一个模块中，不必要限制一个类一个模块

    - 模块内部类采用`_CapWard`方法

    - 类的方法第一个参数必须是self，而静态方法第一个参数必须是cls

- 函数

    - 函数名一律小写，多个单词用下划线隔开。

        ```python
        def run_with_configs(config):
            print(config)
        ```

    - 私有函数名称前加下划线

- 变量名

    - 全局变量尽量只在模块内有效，具体是前缀使用一个下划线。
    - 尽量小写，多个单词用下划线隔开
    - 常量采用全大写，多个字母用下划线隔开

| Type                       | Public             | Internal                                                     |
| :------------------------- | :----------------- | :----------------------------------------------------------- |
| Modules                    | lower_with_under   | _lower_with_under                                            |
| Packages                   | lower_with_under   |                                                              |
| Classes                    | CapWords           | _CapWords                                                    |
| Exceptions                 | CapWords           |                                                              |
| Functions                  | lower_with_under() | _lower_with_under()                                          |
| Global/Class Constants     | CAPS_WITH_UNDER    | _CAPS_WITH_UNDER                                             |
| Global/Class Variables     | lower_with_under   | _lower_with_under                                            |
| Instance Variables         | lower_with_under   | _lower_with_under (protected) or __lower_with_under (private) |
| Method Names               | lower_with_under() | _lower_with_under() (protected) or __lower_with_under() (private) |
| Function/Method Parameters | lower_with_under   |                                                              |
| Local Variables            | lower_with_under   |                                                              |

## 7 reference

- <https://www.zhihu.com/search?q=vscode%20python%20%E8%A7%84%E8%8C%83&utm_content=search_history&type=content>

- <https://www.runoob.com/w3cnote/google-python-styleguide.html>
- <https://www.cnblogs.com/liangmingshen/p/9273413.html>

---

> 后记： 虽然比较多，需要花费时间去记忆，但是一旦形成习惯，养成良好的严谨的编码习惯，之后带来的益处也是无穷的。