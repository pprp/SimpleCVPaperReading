# DeepSort框架梳理

本文梳理的官方库，而不是众多衍生库。地址为：<https://github.com/nwojke/deep_sort>

不得不说，DeepSort提供的代码的注释是非常全面的，大部分工具都有很详细的输入输出描述，非常便于理解，对作者也是十分感谢和佩服。下边框架梳理主要是一下组织结构中大概的功能进行描述，之后再对某些部分进行详细解读。

DeepSort的组织结构：

```
application_util:
  image_viewer.py
  preprocessing.py
  visualization.py
deep_sort:
  detection.py
  kalman_filter.py
  linear_assignment.py
  iou_matching.py
  nn_matching.py
  track.py
  tracker.py
tools:
  freeze_model.py
  generate_detections.py
deep_sort_app.py
evaluate_motchallenage.py
generate_videos.py
show_results.py
```

## 1. application_util

主要是工具箱，提供了视频、图片的读取与保存，目标框可视化等方法和工具。

- image_viewer

    主要是提供了ImageViewer类，功能上非常强大，支持在图片中画矩形框、圆形、椭圆形高斯分布图、文字标注、点。

    还有一些简单的功能，比如判断ROI是否完全包含在整个图片中、获取图片ROI部分内容。

- prepreocessing

    提供了非极大抑制算法

- visualization

    支持视频上的操作和处理，在Visualization内部调用了ImageViewer类，用于视频的显示。

    还提供了生成唯一颜色的方法，通过改变HSV然后映射到RGB上获取。

## 2. deep_sort

主要是deep sort中的核心算法，如卡尔曼滤波、匈牙利算法、ReID等。

- detection

    实际上是bounding box的一个工具箱，保存了这个图片中对应ROI的feature(在ReID部分有用)。

    剩下功能主要是转换bounding box的表达形式，比如(xmin,ymin,xmax,ymax)到(top left, bottom right)

- iou_matching

    直译过来就是iou匹配，这个是sort算法中的核心匹配方法，主要是用于计算在匈牙利算法中前后两帧的代价，deep sort中也用到了，但是只是这个代价评价的一部分。

    提供方法如下：

    1. 计算一个bounding box和剩余候选框的iou
    2. 计算前后两帧得到的所有的 bounding box的iou距离metric

- kelman_filter

    卡尔曼滤波算法（chi2inv95是卡方分布变量），目标框有一个八维的状态（x,y,a,h,vx,vy,va,vh），分别是中心位置、长宽比a、高度h，用于速度建模。

- linear_assignment

    这是匈牙利算法的核心，scikit-learn库中有对应的实现，在这个模块中实现的是级联匹配、通过调用linear_assignment方法完成匈牙利算法。

- nn_matching

    提供计算每对点（bbox中心点）之间的平方距离、Cosine距离、最近邻距离度量。

- track

    主要是轨迹类，存储了了（x,y,a,h）的状态信息，并且在轨迹类中可以调用卡尔曼滤波算法完成对下一个状态的预测。

- tracker

    主要包含一个多目标跟踪器类，在这个类中完成轨迹初始化、轨迹和检测匹配、状态矩阵更新、轨迹更新等。

## 3. tools

主要提供了网络模型和检测部分的功能。

- freeze_model

    核心功能就是将tensorflow中的ckpt文件转化为pb文件。

- generate_detection

    包括图片预处理为一个patch, 模型文件的加载、feature的提取、检测框的提取。运行完这个以后，将会得到npy文件，其中存储的内容是feature。

## 4. other

- deep_sort_app

    该文件是核心的调用，将调用所有的部件，完成目标跟踪任务。

    主要功能有：

    1. 获取视频的信息和相关的检测信息，将所有的信息处理成一个字典，方便访问。

    2. 针对某一帧生成对应的检测结果。
    3. 调用所有的组件，完成整个目标跟踪流程的构建。

- evaluate_motchallenge

    本身没有方法，通过调用deep_sort_app中的run方法，分别生成对应多目标跟踪结果（txt文件），然后可以用该txt文件和gt.txt文件进行对比。具体可以使用官方提供的MATLAB代码，也可以使用py-motmetrics库，这两个库都可以实现MOT中的指标衡量。

- show_result

    用于展示多目标跟踪以后生成的结果，同样调用了visualization方法。

- generate_videos

    将检测结果保存为视频

---

以上主要是总结了各个模块作用，实际上他们互相交织，尤其是deep_sort整个跟踪的流程非常的复杂，先得到detection结果，然后抑制conf<0.7的结果，调用费极大抑制算法，然后进行级联匹配，IoU匹配，进行矩阵更新和后处理等。

另外发现作者非常喜欢给每个模块或者类写一个run方法，外部模块只需要调用run接口即可使用。

总体来讲，虽然看上去代码量不多，但是涉及的逻辑还是略微复杂，需要花点时间仔细理一下。

