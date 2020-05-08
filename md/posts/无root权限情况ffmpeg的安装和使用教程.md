---
title: 无root权限情况ffmpeg的安装和使用教程
date: 2019-12-14 20:19:10
tags:
- ubuntu
- 无管理员权限
- ffmpeg
- 使用教程
---

## 1. 无管理员权限下的安装方法

首先下载安装包并解压：

```
cd ~/installed
wget http://ffmpeg.org/releases/ffmpeg-1.2.6.tar.bz2
tar -jxvf ffmpeg-1.2.6.tar.bz2
```

得到文件夹`ffmpeg-1.2.6`

```
cd ffmpeg-1.2.6
./configure --enable-libmp3lame --enable-libx264 --enable-gpl --prefix=./software/make-3.8.2
# 比较关键的在于--prefix，由于我们没有管理员权限，所以设置的安装位置在本地
make
make install
```

安装完成后会在prefix所指示目录下生成bin, include, lib, share几个文件夹，然后开始编辑`.bashrc`文件，让其可以访问到bin中的内容，添加以下内容：

```
export PATH="/home/*****/makeforffmpeg/bin:$PATH"
```

将`/home/*****/makeforffmpeg/bin`替换成安装的路径,然后运行命令`bash`进行刷新，然后可以尝试`ffmpeg -version`命令，得到以下显示内容：

```
ffmpeg version 1.2.6
built on Dec 14 2019 19:33:01 with gcc 5.4.0 (Ubuntu 5.4.0-6ubuntu1~16.04.10) 20160609
configuration: --disable-yasm --prefix='/install/makeforffmpeg'
libavutil      52. 18.100 / 52. 18.100
libavcodec     54. 92.100 / 54. 92.100
libavformat    54. 63.104 / 54. 63.104
libavdevice    54.  3.103 / 54.  3.103
libavfilter     3. 42.103 /  3. 42.103
libswscale      2.  2.100 /  2.  2.100
libswresample   0. 17.102 /  0. 17.102
```

就说明安装成功了，并且可以正常使用。

## 2. 常用的命令

1. 安装ffmpeg首要的需求就是压缩视频，在服务器上处理了视频以后，通常原来的视频有22M左右，最后竟然处理为600M,这是由于直接使用opencv进行视频的生成，没有进行压缩导致的占用空间过大。这时候就需要ffmpeg进行视频压缩，原先600M左右的视频通过使用以下命令可以压缩到10M左右。

```
ffmpeg -y -i demo.avi -r 10 -b:a 32k output.mp4
```

- -y 覆盖重名文件
- -i 输入文件
- -r 1秒10帧
-  -b:a 32k 表示音频1秒保存32kb，即1秒4kB
- output.mp4 输出文件名称

在python中使用以下命令比较合适

```
os.system("ffmpeg -y -i demo.avi -r 10 -b:a 32k %s.mp4" % (your_file_name))
```

2. 剪切视频

```
ffmpeg  -y -i C:/plutopr.mp4 -acodec copy 
		-vf scale=1280:720
		-ss 00:00:10 -t 15 C:/cutout1.mp4 
```

- -ss time_off set the start time offset 设置从视频的哪个时间点开始截取，上文从视频的第10s开始截取
- -to 截到视频的哪个时间点结束。上文到视频的第15s结束。截出的视频共5s.如果用-t 表示截取多长的时间如 上文-to 换位-t则是截取从视频的第10s开始，截取15s时长的视频。即截出来的视频共15s.
- -vcodec copy表示使用跟原视频一样的视频编解码器。
- -acodec copy表示使用跟原视频一样的音频编解码器。
- -i 表示源视频文件
- -y 表示如果输出文件已存在则覆盖。
- -vf 设置视频分辨率

3. 提取视频关键帧并组合为新的视频（方便标注）

    切割视频为关键帧：

```
ffmpeg -i .\part1_0015.mp4 -vf select='eq(pict_type\,I)' -vsync 2 -s 1280*720 -f image2 .\0015\%04d.jpg
```

将关键帧所在文件夹内容保存为新的视频

```
ffmpeg -threads 2 -y -r 10 -i .\0015\%04d.jpg  output.mp4
```

## References

解压：<https://www.cnblogs.com/0616--ataozhijia/p/3670893.html>

安装：<https://blog.csdn.net/tenebaul/article/details/31439377>

压缩：<https://cloud.tencent.com/developer/article/1331837>