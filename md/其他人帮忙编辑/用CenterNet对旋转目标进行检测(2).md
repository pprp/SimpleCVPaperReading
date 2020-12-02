

# R-CenterNet——用CenterNet对旋转目标进行检测

<font color=#6495ED size=6><u>**GiantPandaCV导语**</u></font>

前段时间纯粹为了论文凑字数做的一个工作，本文不对CenterNet原理进行详细解读，如果对CenterNet原理不了解，建议简单读一下原论文然后对照本文代码理解（对原版CenterNet目标检测代码进行了极大程度精简）。

代码开源：https://github.com/ZeroE04/R-CenterNet

## demo
* **R-DLADCN(推荐)**
    * ![推荐](https://img-blog.csdnimg.cn/20201127231747944.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hcnVfTGlt,size_16,color_FFFFFF,t_70#pic_center)

* **R-ResDCN(主干网用的ResNet而不是DLA)**
    * ![主干网用的ResNet而不是DLA](https://img-blog.csdnimg.cn/20201127232026598.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hcnVfTGlt,size_16,color_FFFFFF,t_70#pic_center)

* **R-DLANet(未编译DCN的主干网)**
    * ![未编译DCN的主干网](https://img-blog.csdnimg.cn/20201127232130292.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hcnVfTGlt,size_16,color_FFFFFF,t_70#pic_center)

* **DLADCN(原始CenterNet)**
    * ![原始CenterNet](https://img-blog.csdnimg.cn/20201127232247887.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L01hcnVfTGlt,size_16,color_FFFFFF,t_70#pic_center)



## 前言

   基本想法就是直接修改CenterNet的head部分，但是是直接在长宽图上加一层通道表示角度，还是多引出一 路feature-map呢？实测是多引出一张feature map比较好，直接在长宽图上加一层通道很难收敛，具体原因我也是猜测，角度和尺度特征基本无共享，且会相互干扰（角度发生些许变化，目标的长宽可能就变了，如果角度是错的，长宽本来是对的呢？反之亦然）引出的feature-map只经历了一层卷积层就开始计算loss，对于这种复杂的关系表征能力不够，可能造成弄巧成拙。网络结构如下：
![image](https://img-blog.csdnimg.cn/img_convert/e4647fab85d5970c7cba99da3313aa54.png)
<center>R-CenterNet网络结构图</center>


## 代码说明
**代码主要分为五个部分：**
 ~~~
{R-CenterNet}
  |-- backbone
   -- |-- dlanet.py
   -- |-- dlanet_dcn.py
  |-- dataset.py
  |-- Loss.py
  |-- train.py
  |-- predict.py
 ~~~

- train.py：模型的训练
- predict.py：模型的前向推理
- backbone：模型的主干网，给了DLA和ResNet的DCN与普通版本，层数可以自定义
- loss.py：模型的损失函数
- dataset.py：模型的数据处理

**不是很重要：**
 ~~~
{R-CenterNet}
  |-- data/airplane
  |-- dcn
 ~~~
- data/airplane：示例训练数据与图片
- dcn:编译好的dcn，说明一下，这里与原版CenterNet编译dcn一样，直接文件夹复制过来即可，如果你不会编译dcn，就用backbone内的dlanet.py与resnet.py

1. 原版CenterNet代码较多，我只需要做目标检测，所以把各种3D检测等都删了，模型架构也拆了只保留了有用部分，并对代码架构进行了重构，方便自己阅读以及魔改。

2. 其次，因为只是加了一个角度检测，所以主要是修改了一下数据处理部分，用的还是VOC格式，只是在.josn文件里面加了一个角度信息，打标签的时候用[0,π]表示，后续在loss内添加了角度的feature-map损失，用的Smooth-L1 loss，打标签方法已在下方。


**2020.1021代码更新(不是很重要)**
 ~~~
{R-CenterNet}
  |-- labelGenerator
  |-- evaluation.py
  |-- imgs
 ~~~
- labelGenerator：生成自己的训练数据
- evaluation.py：性能指标计算
- imgs：性能指标计算示例图片

1. 鉴于一些同学想知道怎么对自己的数据打标签以及生成R-CenterNet可以训练的数据，所以更新一个labelGenerator文件夹，内包含转换函数以及用例。注意，这个文件夹以及其内部函数不是网络必须的，只是想训练自己打标签的数据时用的。

2. 鉴于一些同学想知道模型训练完毕，怎么对比性能，比如旋转框怎么计算mIOU等，所以更新一个evaluation.py以及对应的案例图片和文件夹imgs。注意，这个.py和imgs文件夹不是必须的，只是模型训练结束计算性能用的。
	- 注：每个label里面的目标五个数值：分别为目标中心点（x,y），以及宽度，长度，角度，角度是以12点钟为0°，顺时针旋转的角度，最大为179.99999°（旋转180°，相当于没转）

	![image](https://img-blog.csdnimg.cn/img_convert/6e013eba1e940c9c412e0ce404f9440e.png)

## 训练自己的多分类网络
- 打标签用labelGenerator文件夹里面的代码
- 修改代码中所有num_classes为你的类别数目
- 增加predict.py中方框颜色，我这里只检测单目标，所以只有红蓝框。
- 修改back_bone中hm的数目为你的类别数，如：

```python
def DlaNet(num_layers=34, heads = {'hm': your classes num, 'wh': 2, 'ang':1, 'reg': 2}, head_conv=256, plot=False)
```

## 环境
 * python3
 * 理论上torch >1.0即可，如果报了显存不足的问题就是torch版本低了
 * (可选)如何编译DCN以及环境需求, 与[CenterNet](https://github.com/xingyizhou/centernet) 原版保持一致，不会编译dcn就用backbone中的非dcn版本，性能相比dcn下降一个点左右，随着数据的增大逐渐缩小。

## 结束
- 有问题可以github提issue
- 后续有时间会将上面的工作工程化，C++落地


