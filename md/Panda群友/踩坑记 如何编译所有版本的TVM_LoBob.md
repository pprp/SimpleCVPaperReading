踩坑记 如何编译所有版本的TVM

【GiantPandaCV导语】笔者把tvm v0.9、v0.8、v0.6、v0.5、v0.4、v0.3、v0.2、v0.1都本地安装编译了，也就是除了v0.7没有本地编译以外所有版本都测试了，docker也测试了。遇到了好多小问题，故记录一下。然后测试dlsys的课的作业，原link为[dlsys-course/assignment2-2018: (Spring 2018) Assignment 2: Graph Executor with TVM (github.com)](https://github.com/dlsys-course/assignment2-2018)

[TOC]

一般使用两种方式安装：

1. **docker方式**
2. **本地源码编译**

## 一、docker版本

这个tvm版本是v0.6

```
nvidia-docker run --rm -v /home/zhangxiaoyu/OneFlowWork/tvm/:/home/tvm_learn -it tvmai/demo-gpu bash
```

```l
root@6813267b08b0:/# python3
Python 3.6.8 (default, Oct  9 2019, 14:04:01) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tvm
>>> tvm.__version__
'0.6.dev'
>>> 
```

或者

```
docker pull tvmai/demo-gpu
nvidia-docker run --rm -it tvmai/demo-gpu bash
```



## 二、安装下载v0.4~v0.9版本环境

0. 安装llvm

```
sudo apt install llvm
```

本地环境：ubuntu 18.04

总结：

1. 安装v0.9到v0.4版本，都是一样改cmake的配置，设置相应的后端为ON，我这边测试的dlsys是USE_LLVM = ON；

2. 安装v0.3到v0.1是改make的配置，我测试的是dlsys，直接是LLVM_CONFIG = llvm-config，而且make的位置不一样；

3. tvm v0.7之后，不能直接import topi，要改成import tvm.topi, 0.1到0.6可以直接import topi，但会有新问题，topi这个lib损坏了，需要自己重新安装，方法如下：cd tvm/topi/python;python setup.py install，就可以修复。



下载v0.4~v0.9版本

v0.4可以直接import topi

```
# 创建虚拟环境
conda create -n tvm python=3.7
conda activate tvm

# 下载源码
git clone --recursive -b v0.4 https://github.com/apache/tvm tvm   #-b 这个-b就是修改你要clone哪个版本
cd tvm
git submodule init
git submodule update

# 更新lib
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

看pip是python2还是python3，这步可选

```
which pip 			 # /home/ml/.local/bin/pip
which python		 # /home/ml/anaconda3/envs/tvm_v9/bin/python
```

修改pip的python

```
vim /home/ml/.local/bin/pip
```

```python
#!/home/ml/anaconda3/envs/tvm_v9/bin/python 

# -*- coding: utf-8 -*-
import re
import sys

from pip._internal.cli.main import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

编译tvm

```
mkdir build;cp cmake/config.cmake build
# 修改 tvm/build/config.cmake, 讲USE_LLVM设置成ON即set(USE_LLVM ON)
cd build;cmake ..;make -j4
```

环境变量

```
export TVM_HOME=/path/to/tvm # 注意这个/path/to/tvm是用户本地的路径
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

因为这个版本的topi lib坏了，需要自己重新安装

```
cd tvm/topi/python 
python setup.py install
```

安装一些库

```
pip install numpy decorator scipy nose
```

测试tvm是否配置成功

```
python
import tvm
tvm.__version__ 
import topi 
# or
import tvm.topi 
```

![](https://img-blog.csdnimg.cn/97f97458302e42b8812c7a51426f9407.png)

测试代码

**测试这个dlsys的代码时候，v0.1~v0.4的版本代码不用修改，v0.5~v0.9的需要更改好多api，这部分我也修改完了，放在这个git里面。**

```
git clone -b tvm_v4 https://github.com/RobertLuobo/tvm_dlsys_test.git #tvm版本v0.1~v0.4的版本用这个
# or 
git clone -b tvm_v9 https://github.com/RobertLuobo/tvm_dlsys_test.git #tvm版本v0.5~v0.9的版本用这个

nosetests -v tests/test_tvm_op.py
```

![](https://img-blog.csdnimg.cn/81a68be5f0b14b5191ea7e57f2d3818f.png)

```python
python tests/mnist_dlsys.py -l -m logreg
```

![](https://img-blog.csdnimg.cn/2dd5b0fcacbc4ba6baeb9fbd273a3d97.png)

```python
python tests/mnist_dlsys.py -l -m mlp
```

![](https://img-blog.csdnimg.cn/a90efdfecb95445584a82c4b78cc5e00.png)

安装下载v0.1~v0.3版本环境

下载v0.2版本

这边我clonev0.3好像直接也是v0.2，v0.1我也本地编译安装了一遍

```
# 创建虚拟环境
conda create -n tvm python=3.7
conda activate tvm

# 下载源码
git clone --recursive -b v0.2 https://github.com/apache/tvm tvm  
cd tvm
git submodule init
git submodule update

# 更新lib
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

看pip是python2还是python3，这步可选

```
which pip 				# /home/ml/.local/bin/pip
which python		 # /home/ml/anaconda3/envs/tvm_v9/bin/python
```

修改pip的python

```
vim /home/ml/.local/bin/pip
```

```python
#!/home/ml/anaconda3/envs/tvm_v9/bin/python 

# -*- coding: utf-8 -*-
import re
import sys

from pip._internal.cli.main import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

编译tvm，这里不是cmake，是make，这是跟前面不一样的地方

```
# 在 tvm 主目录下即可/path/to/tvm
cp make/config.mk .
# 修改 /path/to/tvm/config.mk, 改LLVM_CONFIG = llvm-config，即可要执行这段
# 直接make，不需要cmake ..
make -j4
```

环境变量

```
export TVM_HOME=/path/to/tvm # 注意这个/path/to/tvm是用户本地的路径
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

因为有些版本的topi lib坏了，需要自己重新安装

```
cd tvm/topi/python 
python setup.py install
```

安装一些库

```
pip install numpy decorator scipy nose
```


测试tvm是否配置成功

```
python
import tvm
tvm.__version__ 
import topi 
```

测试代码

```
git clone -b tvm_v4 https://github.com/RobertLuobo/tvm_dlsys_test.git

nosetests -v tests/test_tvm_op.py
python tests/mnist_dlsys.py -l -m logreg
python tests/mnist_dlsys.py -l -m mlp
```



## 三、粗略地看看dlsys里面的测试代码

1. 看看有什么文件

```markdown
├── dl_stack.png
├── python
│   └── dlsys
│       ├── autodiff.py
│       ├── __init__.py
│       └── tvm_op.py
├── README.md
└── tests
    ├── dlsys
    │   ├── autodiff.py
    │   ├── __init__.py
    │   └── tvm_op.py
    ├── mnist_dlsys.py
    ├── mnist.pkl.gz
    └── test_tvm_op.py
```

2. 测试了什么

```python
matrix_elementwise_add
matrix_elementwise_add_by_const
matrix_elementwise_mul
matrix_multiply
conv_2d
relu
relu_gradient
softmax
softmax_cross_entropy
reduce_sum_axis_zero
broadcast_to
```

3. 小小的跑了3层mlp的训练，对是训练，python tests/mnist_dlsys.py -l -m mlp

