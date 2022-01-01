---
title: 来自AlexeyAB的目标检测建议
date: 2020-01-09 22:28:15
tags:
---

# 来自AlexeyAB的目标检测建议【翻译】

> 前言: 来自俄国的AlexeyAB不断地更新darknet,不仅添加了darknet在window下的适配，还追踪最新的框架，通过阅读和理解AlexeyAB的建议，可以为我们带来很多启发。本文主要翻译有关目标检测的建议部分，而如何配置等相关内容可以在文末链接找到。

[TOC]

下图是CSPNet中统计的目前的State of the Art的目标检测模型。其中从csresnext50-panet-spp-optimal模型是CSPNet中提出来的，直接结合alexeyab版本的darknet就可以实现。

![](https://img-blog.csdnimg.cn/20200109223251119.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

## 0. 安装

### 0.1 依赖

- window系统，或者linux系统
- CMake版本高于3.8
- CUDA 10.0，cuDNN>=7.0
- OpenCV版本高于2.4
- Linux下需要GCC 或者Clang, Window下取药Visual Studio 15或17或19版

### 0.2 数据集获取

1. MS COCO数据集: 使用`get_coco_dataset.sh`来获取数据集
2. OpenImages数据集: 使用`get_openimages_dataset.py`获取数据集,并按照规定的格式重排训练集。
3. Pascal VOC数据集: 使用`voc_label.py`对数据集标注进行处理。
4. ILSVRC2012数据集(ImageNet Classification): 使用`get_imagenet_train.sh`获取数据集，使用`imagenet_label.sh`用于验证集。
5. German/Belgium/Russian/LISA/MASTIF 交通标志数据集
6. 其他数据集

## 1. 相比原版Darknet的改进

- 添加了对window下运行darknet的支持。
- 添加了SOTA模型： CSPNet, PRN, EfficientNet。
- 在原来基础上添加了新的层：[conv\_lstm], [scale_channels] SE/ASFF/BiFPN, [local_avgpool], [sam], [Gaussian_yolo], [reorg3d] (fixed [reorg]), fixed [batchnorm]
- 可以使用[conv\_lstm]层或者[crnn]层来实现针对视频的目标检测
- 添加数据增强: mixup=1, cutmix=1, mosaic=1 blur=1
- 添加了激活函数: SWISH, MISH, NORM\_CHAN, NORM\CHAN\_SOFTMAX
- 使用CPU RAM来进行GPU处理，可以提升训练阶段的batch size、提升模型的准确率。？？
- 提升了二值网络在CPU和GPU上的检测速度(原来的2-4倍)
- 通过将convolutional层和batch-norm层合并成一个层，提升了7%
- 如果在Makefile中使用CUDNN_HALF参数，可以让网络在TeslaV100 GeForce RTX系列的GPU上的检测速度提升两倍。
- 针对视频的检测进行了优化，对Full HD的视频可以提升1.2倍，对4k的视频可以提升2倍
- 数据增强部分使用opencv SSE/AVX等取代了原来的手写版的数据增强，数据增强速度提升为原来的3.5倍。
- 使用AVX来提升darknet使用Intel CPU训练和测试的速度。YOLOv3可达85%准确？？
- 提供了random=1选项，可以在网络多尺度训练的时候优化内存分配
- 优化了检测时的GPU初始化，这里从最开始使用batch=1而不是用batch=1进行重新初始化？？
- 添加了计算mAP,F1,IoU, Precision-Recall等指标的方法，只需要运行`darknet detector map`命令即可。

- 支持在训练的过程中画loss曲线和准确率曲线，只需要添加`-map`标志即可
- 提供了`-json_port`,`-mjpeg_port`选项，支持作为json 和mjpeg 服务器来在线获取的结果。可以使用你的编写的软件或者web浏览器与json和mjpeg服务器连接。
- 添加了anchor的计算功能，可以根据数据集来聚类得到合适的anchor。
- 添加了一些目标检测和目标跟踪的示例：<https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp>
- 在使用错误的cfg文件或者数据集的时候，添加了运行时的建议和警告

 ## 2. 命令行使用

Linux中使用./darknet，window下使用darknet.exe。

Linux中命令格式类似`./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights`

Linux中的可执行文件在根目录下，Window下则在`\build\darknet\x64`文件夹中。一下是不同情况下应该使用的命令：

- Yolo v3 COCO - **image**: `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -thresh 0.25`
- **Output coordinates** of objects: `darknet.exe detector test cfg/coco.data yolov3.cfg yolov3.weights -ext_output dog.jpg`
- Yolo v3 COCO - **video**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output test.mp4`
- Yolo v3 COCO - **WebCam 0**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights -c 0`
- Yolo v3 COCO for **net-videocam** - Smart WebCam: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights http://192.168.0.80:8080/video?dummy=param.mjpg`
- Yolo v3 - **save result videofile res.avi**: `darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights test.mp4 -out_filename res.avi`
- Yolo v3 **Tiny** COCO - video: `darknet.exe detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights test.mp4`
- **JSON and MJPEG server** that allows multiple connections from your soft or Web-browser `ip-address:8070` and 8090: `./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output`
- Yolo v3 Tiny **on GPU #1**: `darknet.exe detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -i 1 test.mp4`
- Alternative method Yolo v3 COCO - image: `darknet.exe detect cfg/yolov3.cfg yolov3.weights -i 0 -thresh 0.25`
- Train on **Amazon EC2**, to see mAP & Loss-chart using URL like: `http://ec2-35-160-228-91.us-west-2.compute.amazonaws.com:8090` in the Chrome/Firefox (**Darknet should be compiled with OpenCV**): `./darknet detector train cfg/coco.data yolov3.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map`
- 186 MB Yolo9000 - image: `darknet.exe detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights`
- Remeber to put data/9k.tree and data/coco9k.map under the same folder of your app if you use the cpp api to build an app
- To process a list of images `data/train.txt` and save results of detection to `result.json` file use: `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -ext_output -dont_show -out result.json < data/train.txt`
- To process a list of images `data/train.txt` and save results of detection to `result.txt` use:
    `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -dont_show -ext_output < data/train.txt > result.txt`
- Pseudo-lableing - to process a list of images `data/new_train.txt` and save results of detection in Yolo training format for each image as label `<image_name>.txt` (in this way you can increase the amount of training data) use: `darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights -thresh 0.25 -dont_show -save_labels < data/new_train.txt`
- To calculate anchors: `darknet.exe detector calc_anchors data/obj.data -num_of_clusters 9 -width 416 -height 416`
- To check accuracy mAP@IoU=50: `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights`
- To check accuracy mAP@IoU=75: `darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights -iou_thresh 0.75`

**使用Android手机与video-camera和mjpeg-stream**

1. 下载 mjpeg-stream APP: IP Webcam / Smart WebCam
    - Smart WebCam - preferably: <https://play.google.com/store/apps/details?id=com.acontech.android.SmartWebCam2>
    - IP Webcam: <https://play.google.com/store/apps/details?id=com.pas.webcam>
2. 将你的手机与电脑通过WIFI或者USB相连。
3. 开启手机中的Smart WebCam APP。
4. 将以下IP地址替换,在Smart WebCam APP中显示，并运行以下命令启动：

Yolo v3 COCO-model: `darknet.exe detector demo data/coco.data yolov3.cfg yolov3.weights http://192.168.0.80:8080/video?dummy=param.mjpg -i 0`

## 3. Linux下如何编译Darknet

### 3.1 使用cmake编译darknet

CMakeList.txt是一个尝试发现所有安装过的，可选的依赖项(比如CUDA，cuDnn, ZED)的配置文件，然后使用这些依赖项进行编译。他将创建一个可共享库来使用darknet进行代码开发。

在克隆了项目库以后按照以下命令进行执行：

```shell
mkdir build-release
cd build-release
cmake ..
make
make install
```

### 3.2 使用make编译darknet

在克隆了项目库以后，直接运行`make`命令，需要注意的是Makefile中有一些可选参数：

- GPU=1代表编译完成后将可以使用CUDA来进行GPU加速
- CUDNN=1代表通过cuDNN v5-v7进行编译，这样将可以加速使用GPU训练过程
- CUDNN_HALF=1代表在编译的过程中是否添加Tensor Cores, 编译完成后将可恶意将目标检测速度提升为原来的3倍，训练网络的速度提高为原来的2倍。
- OPENCV=1代表编译的过程中加入Opencv, 目前支持的版本有4.x/3.x/2.4.x， 编译结束后将允许对网络摄像头的视频流或者视频文件进行目标检测。
- DEBUG=1 代表是否开启YOLO的debug模式
- OPENMP=1代表编译过程将引入openmp,编译结束后将代表可以使用多核CPU对yolo进行加速。
- LIBSO=1 代表编译库darknet.so， ？？？
- ZED_CAMERA=1???

## 4. 如何在Window下编译Darknet

### 4.1 使用CMake-GUI进行编译

建议使用这种方法来完成window下darknet的编译，需要：Visual Studio 15/17/19, CUDA>10.0, cuDNN>7.0,OpenCV>2.4

使用CMake-GUI编译流程：

1. Configure
2. Optional platform for generator (Set: x64) 
3. Finish
4. Generate
5. Open Project
6. Set: x64 & Release
7. Build
8. Build solution


### 4.2 使用vcpkg进行编译

如果你已经安装了Visual Studio 15/17/19 、CUDA>10.0、 cuDNN>7.0、OpenCV>2.4, 那么推荐使用通过CMake-GUI进行编译。

否则按照以下步骤进行编译:

- 安装或更新Visual Studio到17+,确保已经对其进行全面修补。
- 安装CUDA和cuDNN。
- 安装git和cmake, 并将它们加入环境变量中。
- 安装vcpkg然后尝试安装一个测试库来确认安装是正确的，比如：`vcpkg install opengl`。
- 定义一个环境变量`VCPKG_ROOT`, 指向vcpkg的安装路径。
- 定义另一个环境变量`VCPKG_DEFAULT_TRIPLET`将其指向x64-windows。
- 打开Powershell然后输入以下命令：

```
PS \>                  cd $env:VCPKG_ROOT
PS Code\vcpkg>         .\vcpkg install pthreads opencv[ffmpeg] 
#replace with opencv[cuda,ffmpeg] in case you want to use cuda-accelerated openCV
```

- 打开Powershell, 切换到darknet文件夹，然后运行`.\build.ps1`进行编译。如果想要使用visual studio进行编译，这里提供了CMAKE提供的两个自定义解决方案，一个在`build_with_debug`中，另一个是在`build_win_release`中。

### 4.3 使用legacy way进行编译

- 如果你有CUDA10.0、cuDNN 7.4 和OpenCV 3.x , 那么打开`build\darknet\darknet.sln`, 设置x64和Release 然后运行Build， 进行darknet的编译，将cuDNN加入环境变量中。

    - 在`C:\opencv_3.0\opencv\build\x64\vc14\bin`找到`opencv_world320.dll`和`opencv_ffmpeg320_64.dll`, 然后将其复制到`darknet.exe`同级目录中。
    - 在`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0`中检查是否含有bin和include文件夹。如果没有这两个文件夹，那就将他们从CUDA安装的地方复制到这个地方。
    - 安装cuDNN 7.4.1 来匹配CUDA 10.0, 将cuDNN添加到环境变量`CUDNN`。将`cudnn64_7.dll`复制到`\build\darknet\x64`中。

- 如果你是用的是其他版本的CUDA（不是CUDA 10.0）, 那么使用Notepad打开`build\darknet\darknet.vxcproj`, 将其中的CUDA 10.0替换为你的CUDA的版本。然后打开`\darknet.sln`, 然后右击工程，点击属性properties, 选择CUDA C/C++, 然后选择Device , 然后移除`compute_75,sm_75`。之后从步骤1进行执行。

- 如果你没有GPU但是有OpenCV3.0， 那么打开`build\darknet\darknet_no_gpu.sln`, 设置x64和Release， 然后运行build -> build darknet_no_gpu。

- 如果你只安装了OpenCV 2.4.14，那你应该修改`\darknet.sln`中的路径。

    - (right click on project) -> properties -> C/C++ -> General -> Additional Include Directories: `C:\opencv_2.4.13\opencv\build\include`
    -  (right click on project) -> properties -> Linker -> General -> Additional Library Directories: `C:\opencv_2.4.13\opencv\build\x64\vc14\lib`

- 如果你的GPU有Tensor Cores(Nvidia Titan V/ Tesla V100/ DGX-2等)， 可以提升目标检测速度为原来的3倍，训练速度变为原来的2倍。`\darknet.sln` -> (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add here: `CUDNN_HALF;`

    **注意**：CUDA must be installed only after Visual Studio has been installed.

## 5. 如何训练

### 5.1 Pascal VOC dataset

1. 下载预训练模型 (154 MB): <http://pjreddie.com/media/files/darknet53.conv.74> 将其放在 `build\darknet\x64`文件夹中。

2. 下载pascal voc数据集并解压到 `build\darknet\x64\data\voc` 放在 `build\darknet\x64\data\voc\VOCdevkit\`文件夹中:

    - <http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar>
    - <http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar>
    - <http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar>

    2.1 下载 `voc_label.py` 到 `build\darknet\x64\data\voc`: <http://pjreddie.com/media/files/voc_label.py>

3. 下载并安装python: <https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe>

4. 运行命令: `python build\darknet\x64\data\voc\voc_label.py` (来生成文件: 2007_test.txt, 2007_train.txt, 2007_val.txt, 2012_train.txt, 2012_val.txt)

5. 运行命令: `type 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`

6. 设置 `batch=64` 和`subdivisions=8` 在 `yolov3-voc.cfg`文件中: [link](https://github.com/AlexeyAB/darknet/blob/ee38c6e1513fb089b35be4ffa692afd9b3f65747/cfg/yolov3-voc.cfg#L3-L4)

7. 使用 `train_voc.cmd` 开始训练或者使用以下命令行:

    `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74`

(**Note:** 如果想要停止loss显示，添加 `-dont_show`标志. 如果使用CPU运行, 用`darknet_no_gpu.exe` 代替 `darknet.exe`.)

如果想要该路径的话，请修改 `build\darknet\cfg\voc.data`文件。

**Note:** 在训练中如果你看到avg为nan，那证明训练出错。但是如果在其他部分出现nan，这属于正常现象，训练过程是正常的。

### 5.2 多GPU

1. 首先在一个GPU中训练大概1000个迭代: `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74`
2. 然后停下来，然后使用这个保存的模型 `/backup/yolov3-voc_1000.weights` 然后使用多GPU (最多4个GPU): `darknet.exe detector train cfg/voc.data cfg/yolov3-voc.cfg /backup/yolov3-voc_1000.weights -gpus 0,1,2,3`

在多GPU训练的时候，learning rate需要进行修改，比如单gpu使用0.001，那么多gpu应该使用0.001/GPUS。然后cfg中的burn_in参数和max_batches参数要设置为原来的GPUS倍。

<https://groups.google.com/d/msg/darknet/NbJqonJBTSY/Te5PfIpuCAAJ>

### 5.3 训练自定义数据集(重要)

(to train old Yolo v2 `yolov2-voc.cfg`, `yolov2-tiny-voc.cfg`, `yolo-voc.cfg`, `yolo-voc.2.0.cfg`, ... [click by the link](https://github.com/AlexeyAB/darknet/tree/47c7af1cea5bbdedf1184963355e6418cb8b1b4f#how-to-train-pascal-voc-data))

Training Yolo v3:

1. Create file `yolo-obj.cfg` with the same content as in `yolov3.cfg` (or copy `yolov3.cfg` to `yolo-obj.cfg)` and:

- change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)

- change line subdivisions to [`subdivisions=16`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)

- change line max_batches to (`classes*2000` but not less than `4000`), f.e. [`max_batches=6000`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) if you train for 3 classes

- change line steps to 80% and 90% of max_batches, f.e. [`steps=4800,5400`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L22)

- set network size `width=416 height=416` or any value multiple of 32: <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9>

- change line

     

    ```
    classes=80
    ```

     

    to your number of objects in each of 3

     

    ```
    [yolo]
    ```

    -layers:

    - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610>
    - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696>
    - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783>

- change [

    ```
    filters=255
    ```

    ] to filters=(classes + 5)x3 in the 3

     

    ```
    [convolutional]
    ```

     

    before each

     

    ```
    [yolo]
    ```

     

    layer

    - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603>
    - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689>
    - <https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776>

- when using`[Gaussian_yolo]`layers, change [

    ```
    filters=57
    ```

    ] filters=(classes + 9)x3 in the 3

     

    ```
    [convolutional]
    ```

     

    before each

     

    ```
    [Gaussian_yolo]
    ```

     

    layer

    - <https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L604>
    - <https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L696>
    - <https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L789>

So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.

**(Do not write in the cfg-file: filters=(classes + 5)x3)**

(Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

So for example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolov3.cfg` in such lines in each of **3** [yolo]-layers:

```
[convolutional]
filters=21

[region]
classes=2
```

1. Create file `obj.names` in the directory `build\darknet\x64\data\`, with objects names - each in new line
2. Create file `obj.data` in the directory `build\darknet\x64\data\`, containing (where **classes = number of objects**):

```
classes= 2
train  = data/train.txt
valid  = data/test.txt
names = data/obj.names
backup = backup/
```

1. Put image-files (.jpg) of your objects in the directory `build\darknet\x64\data\obj\`
2. You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files for Yolo v2 & v3: <https://github.com/AlexeyAB/Yolo_mark>

It will create `.txt`-file for each `.jpg`-image-file - in the same directory and with the same name, but with `.txt`-extension, and put to file: object number and object coordinates on this image, for each object in new line:

```
<object-class> <x_center> <y_center> <width> <height>
```

Where:

- `<object-class>` - integer object number from `0` to `(classes-1)`
- `<x_center> <y_center> <width> <height>` - float values **relative** to width and height of image, it can be equal from `(0.0 to 1.0]`
- for example: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
- atention: `<x_center> <y_center>` - are center of rectangle (are not top-left corner)

For example for `img1.jpg` you will be created `img1.txt` containing:

```
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```

1. Create file `train.txt` in directory `build\darknet\x64\data\`, with filenames of your images, each filename in new line, with path relative to `darknet.exe`, for example containing:

```
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```

1. Download pre-trained weights for the convolutional layers and put to the directory `build\darknet\x64`

    - for `csresnext50-panet-spp.cfg` (133 MB): [csresnext50-panet-spp.conv.112](https://drive.google.com/file/d/16yMYCLQTY_oDlCIZPfn_sab6KD3zgzGq/view?usp=sharing)
    - for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
    - for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)
    - for `enet-coco.cfg (EfficientNetB0-Yolov3)` (14 MB): [enetb0-coco.conv.132](https://drive.google.com/file/d/1uhh3D6RSn0ekgmsaTcl-ZW53WBaUDo6j/view?usp=sharing)

2. Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74`

    To train on Linux use command: `./darknet detector train data/obj.data yolo-obj.cfg darknet53.conv.74` (just use `./darknet` instead of `darknet.exe`)

    - (file `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
    - (file `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations)
    - (to disable Loss-Window use `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show`, if you train on computer without monitor like a cloud Amazon EC2)
    - (to see the mAP & Loss-chart during training on remote server without GUI, use command `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)

8.1. For training with mAP (mean average precisions) calculation for each 4 Epochs (set `valid=valid.txt` or `train.txt` in `obj.data` file) and run: `darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map`

1. After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

- After each 100 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`

    (in the original repository <https://github.com/pjreddie/darknet> the weights-file is saved only once every 10 000 iterations `if(iterations > 1000)`)

- Also you can get result earlier than all 45000 iterations.

**Note:** If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.

**Note:** If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.

**Note:** After training use such command for detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

**Note:** if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64: [link](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)



### 5.4 训练tiny-yolo



### 5.5 什么时候停止训练



### 5.6 如何在pascal voc2007数据集上计算mAP指标



## 6. 如何提升目标检测方法:







## 7. 如何标注以及创建标注文件？





## 8. 使用YOLO9000



## 9. 如何将YOLO作为DLL和SO库进行使用？



