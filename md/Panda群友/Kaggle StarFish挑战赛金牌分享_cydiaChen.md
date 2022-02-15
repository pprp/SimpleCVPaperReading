



# Public Leaderboard is not All you need - Kaggle Tensorflow StarFish挑战赛金牌分享

在去年，陆陆续续和队友@willer共同参加了一些计算机视觉竞赛（图像检测、分类竞赛），取得了一些不错的成绩。在年底，偶然得知Kaggle上有一个热度很高的还行检测的比赛，于是简单参考了一下。今天榜单揭晓，非常幸运，在比赛切换到了私榜后，分数大幅度提升，从初赛Public Leaderboard的1100多名直接上分到了Top10，很幸运的荣获了人生中的第一个Kaggle Gold。

## 1. 赛题解析

赛题数据为拍摄的水下视频影像，对截取的视频帧进行单分类任务，即识别出数据中的海星目标。

组委会对赛题数据进行了简单的预处理，提供了3份截取后的视频作为原始训练数据。

### 赛题评价指标：

比赛的评估指标为在不同IoU阈值（IoU0.3:0.8）下的F2 Score。根据F2分数的定义以及官方Evalutaion介绍：其更关心召回率。这可能和赛题背景有关：希望尽可能少的遗漏海星。
$$
F_2 = (1 + 2^2 ) * \frac{Precision * Recall}{2^2 * Precision + Recall}
$$
即评估过程中，大于IoU阈值的框且与Ground Truth匹配的即为正例（True Positive），多余的框都是误报（False Positive），任何漏报的目标框均为假阴性（False Negative）。

同时，评估过程中，一张图像中，所有检测框，根据预测的目标检测框的置信度得分进行降序排列后计算出最终的F2分数。最终的得分也是将各个IoU0.3:0.8阈值下的F2分数进行平均得到。

###  赛题难点：

1. 海星目标的高召回检出；
2. 海洋水下图像的画质劣化等问题；
3. 赛题提供视频帧，如果挖掘帧间时序关联信息。



## 2. 数据可视化

我们对赛题组委会提供的数据进行可视化，对画面中各种海星目标进行可视化，观察数据分布规律与特征。

### 2.1 数据总体可视化

1、视频内部色彩等差异可视化

![视频0可视化](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw450bcj30fe0flabt.jpg)

![视频1可视化](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw577aaj30fg0fg0u7.jpg)

![视频2可视化](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw5pb7dj30gp0fh40h.jpg)

可以看出：水下视频帧之间，伴随距离被摄目标的远近，水下图像中颜色存在一定差异。

### 2.2 目标相关可视化

![目标可视化图1](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw6agygj30l10id0xo.jpg)

![目标可视化图2](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw76berj30vz0fin3n.jpg)

![目标可视化图3](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw8jix8j30u10igafe.jpg)

通过对赛题数据进行可视化后发现：

1、不同画面中海星分布的规律存在差异，图片中海星分布数量不均衡；

2、大部分海星目标属于小目标，因此海星检测可以认为是一个小目标检测的问题；

3、海星目标相对于其他海洋生物，肉眼识别存在一定难度；且还行有可能成群出现，导致目标框间存在一定的重叠现象；

4、海星目标并不是很好辨识，并且视频帧间可能存在一些漏标情况。



 ## 4. 方案设计

综合赛题相关评价指标，数据可视化后发现的赛题数据分布特点集规律，我们进行了方案设计。

模型方面我们采用了Cascade RCNN的二阶段网络结构，作为Baseline模型。结合骨干网络优化、FPN网络优化、检测头优化等改进方式，形成了自研的Cascade RCNN++模型。我们发现：这些模型层面的改进，总是有效的。

1、Backbone: ResNet-50 -> Res2Net 50 -> Res2Net101 -> CB-Res2Net

2、FPN: FPN -> PAFPN

3、Detection Heads: Cascade RCNN Head -> Double Head Cascade RCNN

4、Loss Functions: Smooth L1 loss -> IoU Based loss; 

根据水下数据的特点，结合Discussion中各种大佬的意见，以及我们团队在其他比赛、项目中的经验，设计了以下的数据增强策略。

1、Weak Augmentations: Flip, RandomBrightnessContrast, RGBShift, HueSaturationValue, Noise, CLAHE, Affine, Rotate

2、Strong Augmentations: Copy-Paste, Mosaic, AutoAugmentation V1 policy, Mixup, Cutout

比赛数据中，测试数据均是以视频形式组织整理，并且以赛题的流式推理的测试过程，图片也是按照时序顺序输出的，并没有乱序，因此我们结合Kaggle热门Discussion中的技术方案，加入了目标追踪 (参考的是Norfair实现) 的方法。

![后处理流程图](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw8xoygj30j107r0sy.jpg)

同时，比赛的一开始，我们发现推理结果一直报错，结果定位是采用流式读取时，图像是RGB格式，而我们采用的MMdet的形式是BGR的格式。



## 5. 比赛总结

1、这次比赛在1月初，伴随大佬们研究YoloR的进度，我们的技术方案在Public Leader Board上疯狂掉分，一度掉到了后50%。当时由于接近年关，CVPR Rebuttal等其他情况，我们就并没有进一步对赛题进行优化。没想到大量在PB上表现极高的方法，在切换数据后，直接掉下去，我们也是在Kaggle中直接上升了1000多名进入了金牌区。

![Public Leaderboard](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw7k3svj30p20jzgmq.jpg)

![Private Leaderboard](https://tva1.sinaimg.cn/large/008i3skNgy1gzegw81wjgj30qb0hxdgy.jpg)

这也给了我们很大的启示：无论是打比赛还是做业务，验证集上，**甚至是初赛A、B榜，还是Public Leader Board或是Private Leader Board乃至决赛榜的表现都不能代表全部**。只有研究出一些通用型的方法，无论是在模型层面的改进，或是在数据层面合理构造数据，来符合现实场景下数据的特点，天道酬勤，必然会有一个较好的效果。

2、这里两位队友比我付出了更多的精力和努力，在此也特别感谢给力靠谱的队友。

3、同时，也非常感谢在Kaggle Discussion上的各位数据科学家们的热情discussion，这也给我们很多启发和思考。