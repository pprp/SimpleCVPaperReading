# 【新手向】PyTorch实践之环境配置

这是新手向的第一篇，讲的是PyTorch的环境配置，主要是CPU环境配置。

## 一、Anaconda

conda 是开源包（packages）和虚拟环境（environment）的管理系统。

- **packages 管理：** 可以使用 conda 来安装、更新 、卸载工具包 ，并且它更关注于数据科学相关的工具包。在安装 anaconda 时就预先集成了像 Numpy、Scipy、 pandas、Scikit-learn 这些在数据分析中常用的包。另外值得一提的是，conda 并不仅仅管理Python的工具包，它也能安装非python的包。比如在新版的 Anaconda 中就可以安装R语言的集成开发环境 Rstudio。
- **虚拟环境管理：** 在conda中可以建立多个虚拟环境，用于隔离不同项目所需的不同版本的工具包，以防止版本上的冲突。对纠结于 Python 版本的同学们，我们也可以建立 Python2 和 Python3 两个环境，来分别运行不同版本的 Python 代码。

### 1. 创建自己的虚拟环境

```
conda create -n yourname python=3.6/2.7(版本自己选择)
```

### 2. 切换环境

进入你的环境`source activate yourname`

退出你的环境`source deactivate`

### 3. 查看当前所有的环境

```
conda env list
```

or

```
conda info -e
```

### 4. 安装第三方包

```
conda install nameofpackage
```

or

```
pip install nameofpackage
```

or

```
easy_install nameofpackage
```

其中可以带上安装的版本 eg：`conda install numpy=1.10`

### 5. 卸载第三方包

```
conda remove nameofpackage
```

or

```
pip uninstall nameofpackage
```

### 6. 查看当前环境下所有的包

```
conda list
```

### 7. 克隆一个本地的环境

```
conda create -n 新环境名称 --clone 旧环境名称
```

### 8. 环境的导入与导出

导入：`conda env create -f environment.yml`

导出：`conda env export > environment.yml`

### 9. 升级环境

对所有安装包进行升级：`conda upgrade --all`

升级某个安装包：`conda update nameofpackage`

### 10. 查询包的具体信息

```
conda search nameofpackage
```

不仅可以搜索到对应的包，还可以查看相关的依赖

### 11.删除一个环境

```
conda env remove -n env_name
```



## 二、CPU环境配置

一般来说CPU环境还是很容易配置，因为不需要GPU、不需要找对应的CUDA、cuDNN。

在自己笔记本上搭建环境可以先跑通较小的数据，或者方便进行debug，在这里推荐一下笔者用的一个组合。

- anaconda(环境搭建)

- vscode(编辑器)
- MobaTerm(终端连接服务器)
- winscp(传输数据集或者大文件)
- SFTP(vscode插件)

访问pytorch官网，https://www.pytorch.org, 按照以下配置找到对应命令。

![官网上提供的命令](https://img-blog.csdnimg.cn/20200430111806406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

安装完anaconda后，会出现一个Anaconda Prompt终端，在这个终端中输入以上命令：

```python
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

安装完成后测试一下：

```python
(base) C:\Users\pprp>python
Python 3.7.6 (default, Jan  8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'1.5.0+cpu'
>>>
```

这样笔记本本地环境就可以了，建议再在vscode中anaconda extension pack插件，可以在vscode中方便地切换不同环境。