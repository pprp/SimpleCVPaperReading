---
title: 'Ubuntu16.04,2080Ti 下配置CUDA10,cuDNN7.5'
date: 2019-11-13 16:53:56
tags:
- Ubuntu
- 深度学习环境
- 2080Ti
- CUDA10, cuDNN7.5
---

> 前言：最近又配置了一个深度学习服务器，这让我回想起来第一次配置服务器时候的迷茫，我当时为了配置好深度学习服务器，从安装ubuntu到配置成功tensorflow花了快要一个月时间。感觉那段时间就是在虚度光阴，遇到问题还没有解决就开始怕前边步骤出错，然后重装系统。从源码进行开始配置的tensorflow, 即便查看的是官方的文档，也没有成功配置好的caffe, 巨难配的detectron2 这些环境都给我留下了深刻的印象，也浪费了我大量的时间。还好现在的环境问题变得很好解决了，conda, pip等工具很好地封装了，极大地降低了配置深度学习环境的难度。当时我配置的还是cuda8, cudnn6, 现在随着发展，出现了cuda10.0, 10.1, cudnn 7.5, cudnn7.6等，越来越多的新框架是基于这些新的环境进行配置的。第二次配置服务器只用了2个小时左右，从安装ubuntu到配置好pytorch， 一气呵成，感觉很爽，这也跟之前经历过的挫折息息相关。所以这里总结一下配置方法，希望新入门的小伙伴能少走弯路，直达终点。

## 1. Ubuntu 的安装

- 先制作一个启动盘

- 在你的window下安装UltraISO软件

- 下载Ubuntu16.04对应的镜像，如果有条件可以进行md5检查，因为有可能在数据传输过程中系统有问题。之前就遇到过这种情况，很让人无从下手。所以稳妥起见可以进行md5检查。

- 使用UltralSO软件制作启动盘，具体教程随便百度就能找到，很简单。就是有可能遇到无法格式化U盘的问题，这个时候：

    - 右键我的电脑
    - 选择管理
    - 选择存储中的磁盘管理
    - 右键点击你想格式化的盘符， 选择格式化
    - 然后选择FAT32进行格式化
    - 如果不可行，再自行百度

- 然后将U盘插入服务器，进行启动，进入bios, 一般是在开机的时候狂按F2, F12之类的，这个是根据你主板决定的，大部分服务器也会在开机的时候显示几秒时间。

- 进入bios后选择你的U盘进行启动，比如我的U盘是kingston的，选择的时候就比较容易看到Kingston选项。

- 进入以后选择install ubuntu, 然后在选择语言的时候建议选择英语，不建议选中文，因为可能会出现编码问题。

- 然后比较关键的就是盘区的划分，之前遇到过很多问题，有一次系统都已经安装好了，但是由于安装的软件太多， / 也就是根目录下被占满，或者/home被占满，这都很影响使用，所以在划分的时候需要有前瞻性，为根目录和/home目录提供足够的空间。

    - 如果是在已有window的服务器上安装，且其他盘符都没有什么冲突，就可以选择第一个选项，不用自己操心一下就完成了分配。

    - 如果对安装的盘符有需求，那就需要安装以下方法进行安装：

        - 由于深度学习数据比较大，所以占用的空间会很大，要注意两个地方的空间问题。

            1. /home 这里必须足够大，因为用户数据都放在这个地方
            2. / 根目录也必须足够大，因为系统配置所需要安装的东西也很大

            安装过程参考：<https://blog.csdn.net/zhangxiangweide/article/details/74779652?tdsourcetag=s_pctim_aiomsg>

            以下是我的配置：500G空间：

            - 创建主分区：

            100G = 102400MB 主分区 空间起始位置 Ext4日志文件系统 /

            - 创建swap分区：

            2048MB 逻辑分区 空间起始位置 交换空间

            - 创建boot分区：

            200MB 逻辑分区 空间起始位置 Ext4日志文件系统 /boot

            - 创建home分区:

            剩余的空间 逻辑分区 空间起始位置 Ext4日志文件系统 /home

    - 然后安装完成以后就可以进行重启了，进系统的时候选择Ubuntu,这样系统部分就成功构建了。

    

## 2. Ubuntu个性化设置

首先需要联网，如果是学生，可以用锐捷，然后使用命令：
    
```shell
sudo chmod +x ./rjsupplicant.sh
sudo ./rjsupplicant.sh -u 学号 -p 密码 -d 1
```

就可以联网了，之后进行个性化设置，其中：ssh, git, vim等必须安装。
    
```shell
sudo apt-get update

# ssh 安装
sudo apt-get install openssh-server
sudo service ssh start

# git 安装
sudo apt-get install git

# vim 安装
sudo apt-get install vim
```

剩下的软件比如：chrome, vscode, ssr, wps, atom, typora, jupyter等软件的安装可以查看以下博客：
<https://www.cnblogs.com/pprp/p/9607245.html>
还有一些推荐的博客可以查看以下博客，如果赶时间可以直接安装以上的就足够用了。

- <https://www.jianshu.com/p/48bdc763fde7>
  
- <https://www.jianshu.com/p/223146d671a4>
  
- <https://www.jianshu.com/p/8cae8c37c130>

## 3. 显卡驱动

不出意外，很有可能安装好的分辨率有一点问题，所以这个时候可以先不进行个性化设置，优先将显卡驱动安装，等分辨率等显示正常在慢慢配置。

在此之前先根据显卡的版本下载驱动，可以从官网上找到，比如2080Ti显卡对应的 `NVIDIA-Linux-x86_64-440.31.run`，下载下来放到/home/user等明显好找的地方，不要放在U盘，不太好找。下面教程来自我之前的一篇教程：<https://www.cnblogs.com/pprp/p/9430836.html>， 成功测试过很多次，很可靠的教程。

### 3.1 删除原有驱动

```shell
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo ./NIVIDIA-Linux-X86_64-384.59.run --uninstall # 这里的run就是旧版本的驱动
```

### 3.2 安装依赖

```shell
sudo apt-get install build-essential gcc-multilib dkms
```

如果显示没有找到，那就先执行：`sudo apt-get update`

### 3.3 禁用nouveau驱动

编辑 /etc/modprobe.d/blacklist-nouveau.conf 文件:

```shell
sudo vim  /etc/modprobe.d/blacklist-nouveau.conf 
```

添加以下内容：

```
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

关闭nouveau：

```shell
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
```

### 3.4 重启电脑

```shell
sudo update-initramfs -u
sudo reboot
```

重启后，执行：lsmod | grep nouveau。如果没有屏幕输出，说明禁用nouveau成功。

### 3.5 获取kernel source(重要)

```shell
apt-get install linux-source
apt-get install linux-headers-$(uname -r)
```

`$(uname -r)`就是Ubuntu的发行版号

### 3.6 关掉 X graphic 服务

```shell
sudo systemctl stop lightdm  (or sudo service lightdm stop # 推荐使用这个)
sudo systemctl stop gdm
sudo systemctl stop kdm
```

这个时候会黑屏，不要慌，`ctrl+alt+F1`进入终端，先登录，然后继续进行下一步操作。

### 3.7 安装NVIDIA驱动

```shell
sudo chmod NVIDIA*.run
sudo ./NVIDIA-Linux-****.run –no-x-check -no-nouveau-check -no-opengl-files
```

1. –no-opengl-files：表示只安装驱动文件，不安装OpenGL文件。这个参数不可省略，否则会导致登陆界面死循环，英语一般称为”login loop”或者”stuck in login”。
2. –no-x-check：表示安装驱动时不检查X服务，非必需。
3. –no-nouveau-check：表示安装驱动时不检查nouveau，非必需。
4. -Z, --disable-nouveau：禁用nouveau。此参数非必需，因为之前已经手动禁用了nouveau。
5. -A：查看更多高级选项。

安装过程中一些选项

```
The distribution-provided pre-install script failed! Are you sure you want to continue?
```

选择 `yes` 继续。

```
Would you like to register the kernel module souces with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later?
```

选择 `No` 继续。

问题大概是：`Nvidia's 32-bit compatibility libraries?`

选择 `No` 继续。

```
Would you like to run the nvidia-xconfigutility to automatically update your x configuration so that the NVIDIA x driver will be used when you restart x? Any pre-existing x confile will be backed up.
```

选择`Yes` 继续

### 3.8 挂载NVIDIA驱动并检查

```shell
modprobe nvidia
nvidia-smi
```

出现以下显示才正常：

```
Wed Nov 13 18:35:52 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.31       Driver Version: 440.31       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:01:00.0 Off |                  N/A |
| 53%   68C    P2   238W / 250W |    104MiB / 11016MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     14330      G   /usr/lib/xorg/Xorg                           104MiB |
+-----------------------------------------------------------------------------+
```

### 3.9 返回图形界面

- sudo init 5
- ctrl+alt+F7
- sudo service  lightdm restart

以上三条命令均可。

## 4. 安装CUDA, CUDNN

只要成功安装好驱动，那之后就很简单了，cuda和cudnn的安装很简单，但是需要选对正确的版本进行安装，否则后期还需要删除，很麻烦，尽量一次安装好。比如，这次我是为了mmdetection而配置的环境，所以我选去的是cuda10.0和cudnn7.5, 从官网上下载，其中cudnn可能需要你有账号才可以，这个可能需要你有科学上网的方法。如果没有的话，可以看看国内源。

> ps: cuda下载的时候尽量选择run的形式，不要选deb形式，这样可以稍微提高一下成功率。

如果是RTX 2080Ti的显卡，可以试一下目前的搭配，亲测是可行的：

- cuda_10.0.130_410.48_linux.run
- cudnn-10.0-linux-x64-v7.5.0.56.tgz

下载好，然后准备好安装。

### 4.1 安装CUDA

```shell
sudo chmod +x cuda_10.0.130_410.48_linux.run
sudo sh ./cuda_10.0.130_410.48_linux.run
```

然后安装的过程会出现几个问题，除了问你是否安装driver的那个选`NO`, 其他一路`YES`， 然后就安装好了，很流畅。

然后我们进行一些配置, 让我们的CUDA可以被找到。

1. 当前用户下配置：

```
vim ~/.bashrc
```

ctrl+F 翻页到最后，敲o变成编辑模式，加入以下内容：

```shell
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
```

2. 系统层面的设置：

```shell
sudo vim /etc/profile
```

在文件末尾加入：

```shell
export PATH=/usr/local/cuda-10.0/bin:$PATH
```

运行命令：

```shell
sudo vim /etc/ld.so.conf.d/cuda.conf
```

在文件中加入：

```shell
/usr/local/cuda/lib64
```

3. 让设置生效

```shell
# 用户层面：
bash
# 系统层面：
sudo ldconfig
```

4. 检查是否成功

可以发现用户主目录下生成了一个文件夹  `NVIDIA_CUDA-10.0_Samples`, 进入文件夹，然后make, 如果顺利通过，那就成功了！

### 4.2 安装CUDNN

cudnn安装相对于cuda更简单不过了，只不过有几个点需要理解和注意。

1. cudnn是一个cuda的扩展包，针对深度学习进行计算的加速，文件夹内容解压出来也跟cuda安装完成的文件夹有一定相似，而且不会重复，所以放心复制过去。
2. 需要注意的是，需要有sudo权限，才能将cudnn放过去

cuda安装的位置在：`/usr/local/cuda-10.0`, 同时该文件夹中也有一个cuda文件，作为`cuda-10.0`的软连接。

所以命令就很简单了：

```shell
sudo cp cuda/* /usr/local/cuda-10.0
```

其中cuda/* 是cudnn解压得到的文件夹，还有一种比较简单的，直接解压并放到cuda安装位置：

```shell
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
```



---

以上操作完成后，就剩下pytorch, tensorflow 等环境需要安装了。

## 5. 环境管理软件Anaconda的安装

百度搜索anaconda，下载linux版本64位的anaconda进行安装。

```shell
chmod +x Anaconda3-4.2.0-Linux-x86_64.sh
sudo sh ./Anaconda3-4.2.0-Linux-x86_64.sh
```

安装完以后，运行命令：`bash` , 就可以激活conda环境。

现在默认国外的源，换源执行命令参考自：https://blog.csdn.net/watermelon1123/article/details/88122020

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

另外为了保险起见，建议同时添加第三方conda源：

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

anaconda的主要功能是管理python环境，便于切换，比如新建tensorflow的环境，pytorch的环境，分别隔开进行操作比较安全。具体anaconda中conda命令可以查看我之前写的博客：<https://www.cnblogs.com/pprp/p/9463124.html>

## 6. 安装tensorflow

安装tensorflow很简单，先去查一下cuda10, cudnn7.5对应的tensorflow版本，然后就可以安装。

```shell
pip install tensorflow -i https://mirrors.aliyun.com/pypi/simple
```

采用以上方法是默认的，会安装最新的v2版本的tensorflow， 后边`-i`命令是用来指定源，使用阿里云的源可以更快下载。

## 7. 安装pytorch

进入：<https://pytorch.org/get-started/locally/>

选择合适的版本，如果没有合适的，点击[install previous versions of PyTorch](https://pytorch.org/get-started/previous-versions)选择合适的命令。



---

> 后记： 从一开始安装一个月，到现在安装2小时结束，感觉现在深度学习是真的很方便，工具集成的很好。可能中途还会遇到新的问题，但是大部分人都走过了，稍微查一下，就可以解决问题。希望这篇博客能作为大家的垫脚石，更快地入门吧。之后会介绍mmdetection配置以及注意点。