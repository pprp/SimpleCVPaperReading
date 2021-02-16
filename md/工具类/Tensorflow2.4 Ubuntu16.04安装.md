# Ubuntu16.04 Cuda11.1 Cudnn8.1 Tensorflow2.4环境配置

[TOC]

## 1、环境

显卡：Gtx 1080Ti

系统：Ubuntu16.04

并行：cuda11.1和对应的cudnn8.1

软件：Tensorflow2.4 和 PyTorch1.7

驱动：460.39 

> cuda链接：https://pan.baidu.com/s/1_01EZN_UjQHFyr72ZeFhGA 
> 驱动链接：https://pan.baidu.com/s/1fcPakuEOeRPNaSzL1XTEKw 
> cudnn链接：https://pan.baidu.com/s/1JZcH7KDtRCuPZMx9Xk4K3g 
> 提取码都是：pand 

## 2、驱动安装

建议先安装驱动，然后再安装cuda，虽然安装cuda的时候会带有一个驱动程序，但是总是会遇到错误。具体方法如下：

```
chmod +x NVIDIA-Linux-x86_64-460.39.run
sudo ./NVIDIA-Linux-x86_64-460.39.run
```

然后按照下图指示选项进行选择即可。

![](https://img-blog.csdnimg.cn/20210215232504880.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20210215232539668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/2021021523265158.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20210215232759524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20210215232837474.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20210215232904696.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

安装完成，使用nvidia-smi命令验证结果：

![](https://img-blog.csdnimg.cn/20210215232943423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)



## 3、cuda安装

访问https://developer.nvidia.com/zh-cn/cuda-downloads，按照下图进行选择：

![](https://img-blog.csdnimg.cn/20210216090407460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

```
chmod +x cuda_11.1.0_455.23.05_linux.run
sudo sh cuda_11.1.0_455.23.05_linux.run
```

![](https://img-blog.csdnimg.cn/20210215231914498.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)



![](https://img-blog.csdnimg.cn/20210215231952559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)

开始安装，如果你没有安装驱动，直接安装cuda可能会聚到下面的问题：

![](https://img-blog.csdnimg.cn/2021021523220358.png)

查看对应的log文件发现安装驱动失败，解决方法就是按照第2步先安装驱动，然后再安装cuda。

![](https://img-blog.csdnimg.cn/20210215232252459.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

如果之前安装过cuda，就会遇到以下界面，选择Upgrade all。

![](https://img-blog.csdnimg.cn/20210215233434672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

得到结果，如下图所示，几乎成功了。

![](https://img-blog.csdnimg.cn/20210215233616193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

安装完成以后，还必须进行以下设置：

编辑~/.bashrc，添加：

```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

编辑/etc/profile，添加：

```
export PATH=/usr/local/cuda/bin:$PATH
```

创建链接文件，打开`sudo gedit /etc/ld.so.conf.d/cuda.conf`，在文件中添加：

```
/usr/local/cuda/lib64
```

最后执行 `sudo ldconfig`，使上述设置立即生效。

## 4、cudnn安装

在NIVIDA开发者官网上，找到cudnn的下载页面: [https://developer.nvidia.com/rdp/cudnn-download](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/rdp/cudnn-download) ，选择合适的cudnn，然后在安装完成cuda以后，执行以下命令，就可以完成cudnn的安装了。

![](https://img-blog.csdnimg.cn/20210216090911220.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

PS: cudnn下载必须要登录，比较麻烦，如果和笔者配置差不多的机器，可以用笔者传到百度云的链接下载。

```
cp cudnn-11.2-linux-x64-v8.1.0.77.solitairetheme8 cudnn-11.2-linux-x64-v8.1.0.77.tgz
sudo tar -xzf cudnn-11.2-linux-x64-v8.1.0.77.tgz -C /usr/local
```

## 5、Tensorflow2.4安装

安装最新版的非常简单，直接通过pip安装即可。

```
pip install tensorflow-gpu -U
```


如果没有配置好驱动、cuda、cudnn的情况，运行tensorflow会遇到以下问题。

![](https://img-blog.csdnimg.cn/20210216000026334.png)

安装成功的情况下就如下图所示。

![](https://img-blog.csdnimg.cn/20210216074516711.png)

## 6、PyTorch 1.7 安装

按照官网提示的命令进行安装，直接通过pip安装即可，注意选择好对应的cuda版本。

```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

![](https://img-blog.csdnimg.cn/20210216091658516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)


