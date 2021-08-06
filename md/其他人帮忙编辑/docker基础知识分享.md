# Docker必备基础知识

## 一、docker是什么？

docker可以实现虚拟机隔离应用环境的功能，并且开销比虚拟机小。

## 二、身为算法攻城狮，我们要掌握docker的哪些知识？

1）docker的基础组成部分

docker由：客户端、守护进程、镜像、容器和仓库构成。

**客户端(client)和守护进程(daemon)**

docker的客户端就是你的操作端，你在这里输入docker的一些指令，守护进程就是docker的服务器端，这一端会执行你的指令并返回结果

**镜像(image)**

是一个层叠的只读文件系统，docker通过读取其中的文件来启动一个指定的容器

**容器(containter)**

使用者通过启动某个指定镜像而构件的一个虚拟的操作系统(guest os)就叫做容器，在容器内就像在一个真正的系统内操作一样。

**仓库**

仓库放置了别人做好的多个镜像。可根据需要下载

2）docker基本操作：安装、仓库设置、启动、关闭、退出、进入

docker安装

[https://yeasy.gitbook.io/docker_practice/install/ubuntu](https://yeasy.gitbook.io/docker_practice/install/ubuntu)

docker的镜像加速器，用以从官方镜像仓库中拉取自己需要的镜像

[https://yeasy.gitbook.io/docker_practice/install/mirror](https://yeasy.gitbook.io/docker_practice/install/mirror)

启动docker

```bash
sudo systemctl daemon-reload  #重新加载某个服务的配置文件
sudo systemctl restart docker #重新确定docker
```

基础操作

```bash
# 获取镜像
docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]
# 启动镜像
docker run -it ubuntu:18.04 bash # dokcer run的参数可通过--help来查看
# 关闭容器
docker stop -t=ContainerID 或 docker kill -t=ContainerID
# 暂时退出镜像
exit
# 重新进入镜像
sudo docker exec -it 067 /bin/bash
# 启动一个已停止的容器
docker start 067
# 查看当前docker上容器的运行状态
docker ps
```

3）docker进阶操作：

```bash
请通过 docker command --help查看
```

## 三、使用别人的镜像具体案例

```bash
# 拉取镜像
docker pull ubuntu:18.04 #从仓库中拉取镜像
# 查看本地的image
docker images #查看本地仓库内的镜像
# 用image启动一个容器
docker run -it ubuntu:18.04 bash
# 暂时退出并重新进入容器
exit
docker ps -a
sudo docker exec -it 067 /bin/bash  #此处067是上一指令查到的需要进入的容器ID
# 在容器内进行操作
和ubuntu系统内的操作一致
cat /etc/os-release
```

## 四、做自己的镜像具体案例

```bash
# 拉取基础镜像
docker pull ubuntu:18.04 #从仓库中拉取镜像
# 制作自己的Dockerfile
mkdir ppp&&cd ppp
touch Dockerfile 或 vi Dockerfile
# 制作镜像
docker build -t aaa:bbb .
docker images #在本地镜像列表里就可以看到一个tag为aaa:bbb的image
# 上传镜像
docker commit -a "aaa.com" -m "my apache" a404c6c174a2  ccc:ddd
```

Dockerfile说明

是用来逐层构件一个image的脚本文件，是一个文本文件，其内包含了一条条的指令(Instruction)，每一条指令构建一层，因此每一条指令的内容，就是描述该层应当如何构建。

```bash
FROM ubuntu:18.04  # 指定基础镜像 如果为scratch代表从下一行开始是镜像的第一层
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html # RUN指令用来执行命令，每一行代表新建docker的一个layer
#能在一个layer内执行的指令就通过&& 进行联接，并可应用shell中的换行符\
#在dockerfile每层都要检查，下载，展开的多余文件，以及缓存等能删除的尽量都去掉

COPY #COPY 指令将从构建上下文目录中 <源路径> 的文件/目录复制到新的一层的镜像内的 <目标路径> 位置。
COPY package.json /usr/src/app/ # 将当前上下文路径的json文件复制到image的指定路径下

AND #丰富了COPY的功能，但是会降低构件image速度，如果不需要自动解压缩，则不推荐使用该指令

CMD # ？？？？？？？？？ 还没理解

ENTRYPOINT # 当存在 ENTRYPOINT 后，CMD 的内容将会作为参数传给ENTRYPOINT，从而达到了我们预期的效果。

ENV #用来设置环境变量  ENV <key> <value> 或 ENV <key1>=<value1> <key2>=<value2>...
ENV VERSION=1.0 DEBUG=on \
    NAME="Happy ONE"

ENV LD_LIBRARY_PATH=\
$LD_LIBRARY_PATH:\
$NAME/alpha

ARG # ARG <参数名>[=<默认值>] Dockerfile 中的 ARG 指令是定义参数名称，以及定义其默认值。该默认值可以在构建命令 docker build 中用 --build-arg <参数名>=<值> 来覆盖

ARG DOCKER_USERNAME=library # 注意：在FROM之前定义的ARG参数，会消失，在FROM后需要重新定义
# ARG 所设置的构建环境的环境变量，在将来容器运行时是不会存在这些环境变量的。但是不要因此就使用 ARG 保存密码之类的信息，因为 docker history 还是可以看到所有值的。

VOLUME # 用于指定image启动时挂载到容器中的默认卷，而不是写入容器存储层
VOLUME /data # VOLUME ["<路径1>", "<路径2>"...] 或 VOLUME <路径>
在image启动时可替换
docker run -d -v mydata:/data xxxx #其中的 -v mydata:/data 就是挂载宿主机的卷到容器内

EXPOSE # EXPOSE <端口1> [<端口2>...] EXPOSE 指令是声明容器运行时提供服务的端口，这只是一个声明，在容器运行时并不会因为这个声明应用就会开启这个端口的服务
# 在 Dockerfile 中写入这样的声明有两个好处，一个是帮助镜像使用者理解这个镜像服务的守护端口，以方便配置映射；另一个用处则是在运行时使用随机端口映射时，也就是 docker run -P 时，会自动随机映射 EXPOSE 的端口

WORKDIR # WORKDIR <工作目录路径> 使用 WORKDIR 指令可以来指定工作目录（或者称为当前目录），以后各层的当前目录就被改为指定的目录，如该目录不存在，WORKDIR 会帮你建立目录。

USER  # USER <用户名>[:<用户组>] 指定当前用户
HEALTHCHECK
ONBUILD
LEBEL
SHELL #SHELL 指令可以指定 RUN ENTRYPOINT CMD 指令的 shell，Linux 中默认为 ["/bin/sh", "-c"]   
Dockerfile 多阶段构建
```

## 五、一些docker使用时的小习惯

docker的文件管理系统是逐层实现的，所以构件一个docker的image时，不要添加过多的layers以避免image过大且过于复杂。