# docker入门级使用教程



## 1. Docker是什么

简单理解为主要应用在Linux上的虚拟机，后台中常用到。

镜像：系统的镜像文件，是一个系统

容器：正在运行中的虚拟机

tar文件：将镜像直接保存为tar文件，是一个可加载的中间文件。

Dockfile: 配置文件，根据其中内容进行build

远程仓库：仓库是远端保存好的镜像文件

![](https://img-blog.csdnimg.cn/20210220102126703.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 2. 流程

本地先拉取一个基础镜像，然后自己需要在本地进行加工成自己的需要的环境，这时候需要把镜像转化成容器操作，配置完自己想要的环境以及编写完必须脚本之后，重新把容器转化成镜像提交到云端，云端根据提交的镜像运行得出结果。

- 拉一个镜像：可以从下面地址找连接：https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282029.0.0.2dd1f33aTGlXoR&postId=67720

  Dockfile中FROM字段就可以填写镜像的地址连接，

  比如说： registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3

  或者也可以docker pull +连接 下载镜像

- 运行镜像，得到容器：docker run命令

  docker run -it -w 工作目录 -v 本地目录：映射到容器的目录 --name 容器名字 imageid /bin/bash

  - -it 可以生成终端入口进入
  - -w 这里指定工作目录。建议直接用/,指定为根目录
  - -v 映射本地数据目录,这个很重要！！
  - /bin/bash 这可以是的容器一直都是保持状态
  - -d 后台运行
  - -p 制定内外端口映射 eg 80:80

- 建立必要脚本：

  - run.sh 其中编写推理的代码，比如python infer.py
  - infer.py 推理代码，生成最终预测文件

- 容器转镜像并提交：

  - docker commit 容器id 容器名字 
  - 通过阿里云上命令提示，将镜像上传到仓库
  - 仓库地址+版本号：即可提交

- 提交前的验证：

  - docker run -v /data:/tcdata your_image sh run.sh 



## 3. 常用

docker ps查看正在运行的容器

docker images查看镜像

删除容器： docker rm -f 容器id

删除镜像： docker rmi 镜像id

保存docker镜像: docker save/export 镜像id > 1.tar

加载docker镜像: docker load< 1.tar or cat docker/ubuntu.tar | docker import - test/ubuntu:v1

退出容器：exit

退出容器但不关闭容器Crtl+P+Q

查看log: docker logs + 容器id

停止/启动容器：docker stop/start + 容器id

载入、获取镜像： docker pull ubuntu

当使用了docker run -d 的时候，代表容器在后台运行，想要进入容器需要使用命令：docker exec + 容器id

清理所有终止状态的容器：docker container prune

查看容器内部进程：docker top 容器名称

清理所有None的镜像：docker image prune or docker rmi $(docker images -f "dangling=true" -q)

查找镜像：docker search 字段

查看挂载情况：docker volume inspect volume_idxx

将容器转化为镜像：需要退出但不关闭容器，ctrl+p+q以后，使用命令docker commit+容器id 仓库地址+版本号

## 4. Dockfile

Dockerfile 是一个用来构建镜像的文本文件，文本内容包含了一条条构建镜像所需的指令和说明。

- FROM + 连接 ： 构建镜像 这里也可以添加本地镜像id，这样就可以不频繁从网络上下载了。

  `FROM 756 As build`

- RUN + 命令： 执行的命令 eg: RUN ['sh', 'run.sh']

- ADD . / 将本地目录映射到容器中的/下边

- CMD ['sh', 'run.sh'] 运行命令

## 5. Nvidia-docker

docker 安装

- 安装：sudo apt install docker.io 
- 验证：sudo docker info

nvidia-docker安装方式：

```
sudo usermod -aG docker pdluser

curl https://get.docker.com | sh
 
sudo systemctl start docker && sudo systemctl enable docker
 
# 设置stable存储库和GPG密钥：
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
 
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
 
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
 
# 要访问experimental诸如WSL上的CUDA或A100上的新MIG功能之类的功能，您可能需要将experimental分支添加到存储库列表中.
# 可加可不加
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
 
# nvidia-docker2更新软件包清单后，安装软件包（和依赖项）：
sudo apt-get update
 
sudo apt-get install -y nvidia-docker2
 
# 设置默认运行时后，重新启动Docker守护程序以完成安装：
sudo systemctl restart docker
```
