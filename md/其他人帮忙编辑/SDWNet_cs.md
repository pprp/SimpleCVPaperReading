【GiantPandaCV导语】今天带来一篇熊猫人自己的工作，这篇文章主要是在年初NTIRE图像去模糊赛道的技术方案的基础上，对我们的方案进行了一个简单的extend，以及详尽的阐述。



## 1. 简单介绍

这篇文章主要围绕一些有趣的问题进行探讨：

1、在大量深度学习下游任务中，类似U-Net的结构进行反复的上下采样会引入最终结果的misalignment。而在低级视觉方法中，反复的上下采样是否有用？

2、能否有更好的办法来替代反复的上下采样？

3、现有的SOTA网络：比如MPRNet等方法，网络结构复杂，且需要极长时间的训练，因此能否有更简单结构的网络来达到相近的性能呢？

![常见的去模糊网络电路图示意，(d)为我们的方法](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09zy8lvj60mq0onmzg02.jpg)

围绕上述的几个问题，我们团队通过“**渐进空洞率的空洞卷积**”、“**小波重建模块**”、“**精细化的网络训练tricks**”三个方面，设计了SDWNet，尝试以一个新的视角来处理Image Debluring问题。

## 2. 网络设计

![SDWNet的网络结构](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09xhgw7j60xz0coq4m02.jpg)

1. **尽可能少的上下采样操作**：大部分图像去模糊网络都采用的U-Net作为backbone，然后通过修改其中的模块或者使用不同策略来获得卓越的性能，但是这样同时带来了参数过大以及反复上下采样的操作，这些方法在一定程度上存在训练缓慢，参数量大等问题。

   因此，为了避免这一情况，我们利用**不同空洞率的空洞卷积可以捕获不同感受野的特征来减少反复上下采样这一问题**。同时为了减少网络的计算量，我们只在特征提取的过程中使用了一次下采样，然后通过多个空洞卷积模块来融合不同感受野的特征信息，进而通过上采样回到原尺寸。中间我们添加了多个跳跃连接来避免这一过程丢失的信息。


![类金字塔型多空洞卷积模块](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09vplg1j60lq0k9tah02.jpg)

2. **渐进空洞卷积率**：我们考虑到仅仅使用单一空洞卷积来获得大的感受野特征十分困难，因此我们采用了**融合多个感受野特征**的方式来解决这一问题，同时我们采用了**并行的四个渐进空洞率**的设置，这样有效的捕获不同尺度下的感受野特征，从而更好的利用运动模糊在空间上的相似性，从而可以恢复出清晰的图像。


![小波重建模块](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09z1otjj60m00h6dhx02.jpg)

3. **小波重建模块**：为了进一步是的我们恢复的图像可以包含更多的纹理细节，我们利用小波变换将特征变换到频域中进行恢复并结合原有的空间域恢复出来的特征来得到最终的清晰图像。通过这一方式我们利用频域中恢复的信息来补充空间域，而不是在单个空间/频域中执行去模糊，使得恢复的图像包含更多的高频细节。同时，我们在实验中直接融合所有的小波子带信息会存在一定的性能下降，因此，我们发现不同频率小波子带之间的信息会存在相互干扰，使得网络生成的图像会产生一定的伪影。为了解决这一问题，我们采用了一个**共享的三层卷积**来分别恢复不同的频率子带。


## 3. 训练策略及实验结论

同时，我们在复现其他作者的工作时，发现以2020年霸榜的MPRNet工作，有一个很严重的问题，需要训练极长的训练轮数**3000**个epoch才可以达到最终论文中霸榜的数值指标。

但这一点在少GPU资源的环境中是很难对这些工作进行复现的。同时，在各种代码库对齐的过程中，我们也发现大家使用的训练策略都不尽相同。

### 3.1 Sliding Crop Matters!

因为MPRNet网络需要很长的训练周期数，并且GPU上的batch size也并不大，因此每次训练所需要的等效iteration是一个庞大的数量级。这很难说实际上的网络提升究竟是来源于余弦退火后在长时间训练后带来的亦或是网络本身结构的优越性带来的；

同时，我们还发现：在深度学习的图像重建任务中，大家都默认采用random crop的方式进行在线裁剪。

而在检测任务中，random crop往往会带来网络性能的下降。因此，我们的方法也探究了低级视觉任务中，使用图像的离线裁剪是否能够有效提升网络的鲁棒性。


![Sliding Crop方法示意图](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09tb1o7j60lw0dqq4p02.jpg)



受制于文章篇幅限制，我们的实验结果在补充材料中展示，在这里为大家提供一下展示：

![Image Patch尺寸大小与网络性能的关系](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09tsiqtj60ey05k0tb02.jpg)

在本文的实验中，我们探究了random crop和离线滑窗后模型训练的性能差异。我们可以看到，离线滑窗后的模型又可以有0.1个点的提升，**无痛涨点！**

### 3.2 Width might be significate. 

同时，由于我们的网络结构简单，其中复杂的电路图式的连接极少，因此我们还额外进行了网络深度、网络宽度对整体模型性能的影响实验。

![SDWNet网络深度与宽度对性能的影响](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09sjrr8j60zl0gj76j02.jpg)

从这里我们看到：本文提出的架构中，**网络的宽度比深度对低级视觉任务的增益更大**。更深的网络并不一定能带来性能的正向提升。

### 3.3 Ablation Study on Network Structure.

本文中，我们对模型结构层面的逐项改进，我们也进行了消融实验。

![消融实验](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09u4xekj60nh09tmy502.jpg)

从实验结果中，可以看出：使用更好的激活函数，**替换bicubic上采样为bilinear上采样**，设置不同的dilated rate和引入小波重建模块均能有效提升网络的PSNR指标数值。

### 3.4 图像主观质量展示

![图像主观质量评价](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09yg8t5j613x0chdiz02.jpg)

![image-20211013115512867](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09uuwulj613l0cpn0202.jpg)

![image-20211013120439402](https://tva1.sinaimg.cn/large/008i3skNgy1gvl09wmundj60tk0j5tce02.jpg)

我们可以重点关注一下，我们提出的方法和其他有大量反复上下采样的网络结构的网络相比，在恢复文字等具有强结构化特征的内容上，有了极大的性能提升，能够稳定提升文字辨识的质量。

## 4. 未来展望

1. 本文提出了一种方式来代替上下采样的方式得到更大的感受野。这种方法可能并不是一种唯一的方法。2021年以来，以Swin Transformer [2]为代表的Vision Transformer的方法展示出的融合locality和global的特性也不失是一种好的方式；
2. 本文提出的sliding crop的方法无伤涨点，能够有效提升图像重建网络的性能，并且也不需要漫长的模型训练时间。



## 参考文献

[1] Zamir S W, Arora A, Khan S, et al. Multi-stage progressive image restoration[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 14821-14831.

[2] Liu Z, Lin Y, Cao Y, et al. Swin transformer: Hierarchical vision transformer using shifted windows[J]. arXiv preprint arXiv:2103.14030, 2021.

