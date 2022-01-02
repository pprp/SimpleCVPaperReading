# SimplePaperReading
分享公众号GiantPandaCV中的博客＆"神经网络架构搜索"中的博客列表。

- 📦 CSDN: https://blog.csdn.net/DD_PP_JJ
- 😃 博客园: https://www.cnblogs.com/pprp
- 😇 zhihu: https://www.zhihu.com/people/peijieDong

- :astonished: 简书：https://www.jianshu.com/u/d9ab1e1c8ba4

## 目录

- [神经网络结构搜索](#神经网络结构搜索)
- [注意力机制](#注意力机制)
- [Transformer](#Transformer)
- [目标检测](#目标检测)
  - [YOLOv3系列](#YOLOv3系列)
  - [CenterNet系列](#CenterNet系列)
  - [多目标跟踪](#多目标跟踪)
- [领域综述](#领域综述)
- [工具集](#工具集)



## 神经网络结构搜索

| 博客列表                                                     |
| ------------------------------------------------------------ |
| [Bag of Tricks for Neural Architecture Search](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/Bag%20of%20Tricks%20for%20NAS_pprp.md) |
| [ECCV20 BigNAS无需后处理直接部署](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/ECCV20%20BigNAS%E6%97%A0%E9%9C%80%E5%90%8E%E5%A4%84%E7%90%86%E7%9B%B4%E6%8E%A5.md) |
| [Microsoft NNI 有关NAS的核心类](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/Microsoft%20NNI%20%E6%9C%89%E5%85%B3%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E7%9A%84%E5%87%A0%E4%B8%AA%E6%A0%B8%E5%BF%83%E7%B1%BB.md) |
| [Microsoft NNI入门](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/Microsoft%20NNI%E5%85%A5%E9%97%A8.md) |
| [NAS的挑战和解决方案-一份全面的综述](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/NAS%E7%9A%84%E6%8C%91%E6%88%98%E5%92%8C%E5%87%BA%E8%B7%AF-%E4%B8%80%E4%BB%BD%E5%85%A8%E9%9D%A2%E7%BB%BC%E8%BF%B0.md) |
| [NetAug 韩松团队新作](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/NetAug%20%E9%9F%A9%E6%9D%BE%E5%9B%A2%E9%98%9F%E6%96%B0%E4%BD%9C_pprp.md) |
| [P-DARTS 渐进式搜索](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91%20P-DARTS%20%E6%B8%90%E8%BF%9B%E5%BC%8F%E6%90%9C%E7%B4%A2.md) |
| [CVPR2021 NAS相关论文链接](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91CVPR2021%20NAS%E7%9B%B8%E5%85%B3%E8%AE%BA%E6%96%87%E9%93%BE%E6%8E%A5.md) |
| [DARTS 可微分神经网络结构搜索开创者](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91DARTS%C2%A0.md) |
| [DNA: Block-wisely Supervised NAS with KD](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91DNA.md) |
| [【神经网络搜索】Efficient Neural Architecture Search](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91Efficient%20Neural%20Architecture%20Search.md) |
| [ICLR 2021 NAS 相关论文(包含Workshop)](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91ICLR%202021%20NAS%20%E7%9B%B8%E5%85%B3%E8%AE%BA%E6%96%87(%E5%8C%85%E5%90%ABWorkshop).md) |
| [NAS-RL（ICLR2017）](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NAS-RL.md) |
| [神经网络架构国内外发展现状](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NAS%E5%9B%BD%E5%86%85%E5%A4%96%E5%8F%91%E5%B1%95%E7%8E%B0%E7%8A%B6.md) |
| [【神经网络搜索】NAS总结](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NAS%E6%80%BB%E7%BB%93.md) |
| [【神经网络架构搜索】NAS-Bench-101: 可复现神经网络搜索](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NasBench101.md) |
| [【神经网络搜索】NasBench301 使用代理模型构建Benchmark](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NasBench301_pprp.md) |
| [【神经网络搜索】Once for all](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91Once%20for%20all.md) |
| [PC-DARTS 部分通道链接](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91PC-DARTS%20%E9%83%A8%E5%88%86%E9%80%9A%E9%81%93%E9%93%BE%E6%8E%A5.md) |
| [【神经网络搜索】ProxyLessNAS](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91ProxyLessNAS.md) |
| [【神经网络架构搜索】SMASH直接生成候选网络权重](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91SMASH.md) |
| [Semi-Supervised Neural Architecture Search](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91Semi-Supervised%20Neural%20Architecture%20Search.md) |
| [【神经网络搜索】Single Path One Shot](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91Single%20Path%20One%20Shot.md) |
| [你所需要知道的关于AutoML和NAS的知识点](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E4%BD%A0%E6%89%80%E9%9C%80%E8%A6%81%E7%9F%A5%E9%81%93%E7%9A%84%E5%85%B3%E4%BA%8EAutoML%E5%92%8CNAS%E7%9A%84%E7%9F%A5%E8%AF%86%E7%82%B9.md) |





## 注意力机制

| 博客列表                                                     |
| ------------------------------------------------------------ |
| [【CV中的Attention机制】Non-Local-neural-networks的理解与实现](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91Non-Local-neural-networks%E7%9A%84%E7%90%86%E8%A7%A3%E4%B8%8E%E5%AE%9E%E7%8E%B0.md) |
| [【CV中的Attention机制】BiSeNet中的FFM模块与ARM模块](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91BiSeNet%E4%B8%AD%E7%9A%84FFM%E6%A8%A1%E5%9D%97%E4%B8%8EARM%E6%A8%A1%E5%9D%97.md) |
| [【CV中的Attention机制】CBAM的姊妹篇-BAM模块](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91CBAM%E7%9A%84%E5%A7%8A%E5%A6%B9%E7%AF%87-BAM%E6%A8%A1%E5%9D%97.md) |
| [【CV中的Attention机制】DCANet解读](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91DCANet%E8%A7%A3%E8%AF%BB.md) |
| [【CV中的Attention机制】Selective-Kernel-Networks-SE进化版](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91Selective-Kernel-Networks-SE%E8%BF%9B%E5%8C%96%E7%89%88.md) |
| [【CV中的Attention机制】ShuffleAttention](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91ShuffleAttention.md) |
| [【CV中的Attention机制】易于集成的Convolutional-Block-Attention-Module-CBAM模块](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91%E6%98%93%E4%BA%8E%E9%9B%86%E6%88%90%E7%9A%84Convolutional-Block-Attention-Module-CBAM%E6%A8%A1%E5%9D%97.md) |
| [【CV中的Attention机制】融合Non-Local和SENet的GCNet](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91%E8%9E%8D%E5%90%88Non-Local%E5%92%8CSENet%E7%9A%84GCNet.md) |
| [【CV中的Attention机制】模块梳理合集](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91%E6%A8%A1%E5%9D%97%E6%A2%B3%E7%90%86%E5%90%88%E9%9B%86.md) |
| [【CV中的attention机制】语义分割中的scSE模块](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90CV%E4%B8%AD%E7%9A%84attention%E6%9C%BA%E5%88%B6%E3%80%91%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E4%B8%AD%E7%9A%84scSE%E6%A8%A1%E5%9D%97.md) |
| [【cv中的Attention机制】最简单最易实现的SE模块](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E3%80%90cv%E4%B8%AD%E7%9A%84Attention%E6%9C%BA%E5%88%B6%E3%80%91%E6%9C%80%E7%AE%80%E5%8D%95%E6%9C%80%E6%98%93%E5%AE%9E%E7%8E%B0%E7%9A%84SE%E6%A8%A1%E5%9D%97.md) |
| [卷积神经网络中的即插即用模块](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E5%8D%B3%E6%8F%92%E5%8D%B3%E7%94%A8%E6%A8%A1%E5%9D%97.md) |
| [神经网络加上注意力机制，精度反而下降，为什么会这样呢？](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8A%A0%E4%B8%8A%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%EF%BC%8C%E7%B2%BE%E5%BA%A6%E5%8F%8D%E8%80%8C%E4%B8%8B%E9%99%8D%EF%BC%9F.md) |



## Transformer

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/A%20Battle%20of%20Network%20Structure%20MLP%20Transformer%20CNN_pprp.md) |
| [BoTNet:Bottleneck Transformers for Visual Recognition](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/BoTNet_Bottleneck%20Transformer_pprp.md) |
| [CvT: 如何将卷积的优势融入Transformer](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/CvT_pprp.md) |
| [DeiT：使用Attention蒸馏Transformer](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/DeiT_pprp.md) |



## 目标检测

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [增强CNN学习能力的Backbone:CSPNet](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/FPN/CSPNet.md) |
| [打通多个视觉任务的全能Backbone:HRNet](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/FPN/HRNet.md) |
| [两阶段实时检测网络ThunderNet](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/FPN/ThunderNet.md) |
| [【CV中的特征金字塔】Feature Pyramid Network](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/FPN/%E3%80%90CV%E4%B8%AD%E7%9A%84%E7%89%B9%E5%BE%81%E9%87%91%E5%AD%97%E5%A1%94%E3%80%91FPN.md) |

### YOLOv3系列

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [【从零开始学习YOLOv3】1. YOLO cfg文件解析](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%911.%20YOLO%20cfg%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90.md) |
| [【从零开始学习YOLOv3】2. 代码配置和数据集处理](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%912.%20%E4%BB%A3%E7%A0%81%E9%85%8D%E7%BD%AE%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86.md) |
| [【从零开始学习YOLOv3】3. YOLOv3的数据组织与处理](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%913.%20YOLOv3%E7%9A%84%E6%95%B0%E6%8D%AE%E7%BB%84%E7%BB%87%E4%B8%8E%E5%A4%84%E7%90%86.md) |
| [【从零开始学习YOLOv3】4. YOLOv3中的参数进化](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%914.%20YOLOv3%E4%B8%AD%E7%9A%84%E5%8F%82%E6%95%B0%E8%BF%9B%E5%8C%96.md) |
| [【从零开始学习YOLOv3】5. 网络模型的构建](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%915.%20%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9E%84%E5%BB%BA.md) |
| [【从零开始学习YOLOv3】6. 模型构建中的YOLOLayer](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%916.%20%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA%E4%B8%AD%E7%9A%84YOLOLayer.md) |
| [【从零开始学习YOLOv3】7. 教你在YOLOv3模型中添加Attention机制](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%917.%20%E6%95%99%E4%BD%A0%E5%9C%A8YOLOv3%E6%A8%A1%E5%9E%8B%E4%B8%AD%E6%B7%BB%E5%8A%A0Attention%E6%9C%BA%E5%88%B6.md) |
| [【从零开始学习YOLOv3】8. YOLOv3中Loss部分计算](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E3%80%90%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AD%A6%E4%B9%A0YOLOv3%E3%80%918.%20YOLOv3%E4%B8%ADLoss%E9%83%A8%E5%88%86%E8%AE%A1%E7%AE%97.md) |
| [我们是如何改进YOLOv3进行红外小目标检测的？](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/%E5%A6%82%E4%BD%95%E6%94%B9%E8%BF%9Byolov3.md) |
| [YOLOv3和SSD有什么区别](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/yolo%E5%92%8Cssd%E7%9A%84%E5%8C%BA%E5%88%AB.md) |
| [一张图梳理YOLOv4论文](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/YOLOv3/YOLOv4.md) |



### CenterNet系列

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [CenterNet0-pytorch版CenterNet训练自己的数据集](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/Centernet0-%E6%95%B0%E6%8D%AE%E9%9B%86%E9%85%8D%E7%BD%AE.md) |
| [CenterNet1-数据加载解析](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/CenterNet1-%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E5%BB%BA.md) |
| [CenterNet2-骨干网络之hourglass](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/CenterNet2-%E9%AA%A8%E5%B9%B2%E7%BD%91%E7%BB%9C%E4%B9%8Bhourglass.md) |
| [CenterNet3-骨干网络之DLASeg](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/CenterNet3-%E9%AA%A8%E5%B9%B2%E7%BD%91%E7%BB%9C%E4%B9%8BDeepLayerAgregation.md) |
| [CenterNet4-loss计算代码解析](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/CenterNet4-Loss%E8%AE%A1%E7%AE%97.md) |
| [CenterNet5-测试推理过程](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/CenterNet5-%E6%B5%8B%E8%AF%95%E6%8E%A8%E7%90%86%E8%BF%87%E7%A8%8B.md) |
| [用关键点进行目标检测](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/%E5%85%B3%E9%94%AE%E7%82%B9%E5%AE%9A%E4%BD%8D%E7%BA%A2%E5%A4%96%E5%B0%8F%E7%9B%AE%E6%A0%87.md) |
| [CenterNet代码原理详解](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/CenterNet/CenterNet%E7%94%B5%E5%AD%90%E4%B9%A6.md) |

### 多目标跟踪

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [Deep SORT多目标跟踪算法代码解析(上)](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E8%B7%9F%E8%B8%AA/Deep%20Sort%E5%A4%9A%E7%9B%AE%E6%A0%87%E8%B7%9F%E8%B8%AA%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90(%E4%B8%8A).md) |
| [Deep SORT多目标跟踪算法代码解析(下)](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E8%B7%9F%E8%B8%AA/Deep%20Sort%E5%A4%9A%E7%9B%AE%E6%A0%87%E8%B7%9F%E8%B8%AA%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90(%E4%B8%8B)%20.md) |
| [Deep SORT多目标跟踪算法代码解析](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E8%B7%9F%E8%B8%AA/DeepSORT%E7%AE%97%E6%B3%95%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90(%E5%85%A8).md) |
| [DarkLabel转换MOT、ReID、VOC格式数据集脚本分享](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E8%B7%9F%E8%B8%AA/darklabel%E6%95%99%E7%A8%8B.md) |
| [Deep SORT论文阅读笔记](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E8%B7%9F%E8%B8%AA/deep%20sort%E8%AE%BA%E6%96%87.md) |



## 知识蒸馏

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [【Attention Transfer】paying more attention to attention](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/Attention%20Transfer_pprp.md) |
| [Towards Oracle Knowledge Distillation with NAS](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/Towards%20Oracle%20Knowledge%20Distillation%20with%20NAS_pprp.md) |
| [【知识蒸馏】Deep Mutual Learning](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/%E3%80%90%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F%E3%80%91Deep%20Mutual%20Learning_pprp.md) |
| [【知识蒸馏】Knowledge Review](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/%E3%80%90%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F%E3%80%91Knowledge%20Review_pprp.md) |
| [知识蒸馏综述: 知识的类型](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F%E7%BB%BC%E8%BF%B0-%E7%9F%A5%E8%AF%86%E7%9A%84%E7%B1%BB%E5%9E%8B_pprp.md) |
| [知识蒸馏综述:网络结构搜索应用](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F%E7%BB%BC%E8%BF%B0_%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2%E5%BA%94%E7%94%A8_pprp.md) |
| [知识蒸馏综述：代码整理](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F%E7%BB%BC%E8%BF%B0%EF%BC%9A%E4%BB%A3%E7%A0%81%E6%95%B4%E7%90%86_pprp.md) |
| [知识蒸馏综述：蒸馏机制](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F/%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F%E7%BB%BC%E8%BF%B0%EF%BC%9A%E8%92%B8%E9%A6%8F%E6%9C%BA%E5%88%B6_pprp.md) |

## 领域综述

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [Bag of Tricks for Object Detection](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/bag%20of%20trick%20for%20object%20detection.md) |
| [【科普】神经网络中的随机失活方法](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/Dropout%E6%A8%A1%E5%BC%8F%E5%92%8C%E5%8F%91%E5%B1%95.md) |
| [DeepSort框架梳理](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/deepsort%E6%A1%86%E6%9E%B6%E6%A2%B3%E7%90%86.md) |
| [precision和recall重新理解](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/precision%E5%92%8Crecall%E9%87%8D%E6%96%B0%E7%90%86%E8%A7%A3.md) |
| [PyTorch中模型的可复现性](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/pytorch%E4%B8%AD%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%8F%AF%E5%A4%8D%E7%8E%B0%E6%80%A7.md) |
| [目标检测和感受野的总结和想法](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/%E5%85%B3%E4%BA%8E%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%92%8C%E6%84%9F%E5%8F%97%E9%87%8E%E7%9A%84%E4%B8%80%E7%82%B9%E6%80%BB%E7%BB%93%E5%92%8C%E6%83%B3%E6%B3%95.md) |
| [【综述】神经网络中不同类型的卷积层](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E5%8D%B7%E7%A7%AF%E6%A0%B8%E6%A2%B3%E7%90%86.md) |
| [卷积神经网络中的各种池化操作](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E6%B1%A0%E5%8C%96%E6%93%8D%E4%BD%9C.md) |
| [多目标跟踪MOT16数据集和评价指0标](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/%E5%A4%9A%E7%9B%AE%E6%A0%87%E8%B7%9F%E8%B8%AA%E6%95%B0%E6%8D%AE%E9%9B%86%E8%A7%A3%E6%9E%90.md) |
| [【翻译】手把手教你用AlexeyAB版Darknet](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E9%A2%86%E5%9F%9F%E7%BB%BC%E8%BF%B0/%E6%9D%A5%E8%87%AAAlexeyAB%E7%9A%84%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E5%BB%BA%E8%AE%AE.md) |



## 工具集

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [Fixing the train-test resolution discrepancy](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/Fixing%20the%20train-test%20resolution_pprp.md) |
| [【论文阅读】Mixed Precision Training](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/Mixed%20Precision%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB.md) |
| [PyTorch Lightning工具学习](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/Pytorch_lightning%E5%B7%A5%E5%85%B7%E6%8E%A8%E8%8D%90.md) |
| [Tensorflow2.0 入门](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/Tensorflow2.0%20%E5%85%A5%E9%97%A8%EF%BC%88%E4%B8%80%EF%BC%89.md) |
| [Ubuntu16.04 Cuda11.1 Cudnn8.1 Tensorflow2.4环境配置](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/Tensorflow2.4%20Ubuntu16.04%E5%AE%89%E8%A3%85.md) |
| [DarkLabel转换MOT、ReID、VOC格式数据集脚本分享](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/darklabel%E6%95%99%E7%A8%8B.md) |
| [docker入门级使用教程](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/docker%E5%85%A5%E9%97%A8%E7%BA%A7%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B.md) |
| [深度学习应用的服务端部署](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/flask%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8%E7%9A%84%E6%9C%8D%E5%8A%A1%E7%AB%AF%E9%83%A8%E7%BD%B2.md) |
| [Python Yaml配置工具](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/python%20yaml%E9%85%8D%E7%BD%AE%E5%B7%A5%E5%85%B7.md) |
| [Sphinx 快速构建工程文档](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/sphinx%E5%BF%AB%E9%80%9F%E6%9E%84%E5%BB%BA%E5%B7%A5%E7%A8%8B%E6%96%87%E6%A1%A3.md) |
| [Tmux科研利器-更方便地管理实验](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/tmux%E4%BD%BF%E7%94%A8%E8%B0%83%E7%A0%94.md) |
| [**人脸轻量级**](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E4%BA%BA%E8%84%B8%E8%BD%BB%E9%87%8F%E7%BA%A7.md) |
| [如何使用logging生成日志](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8logging%E7%94%9F%E6%88%90%E6%97%A5%E5%BF%97.md) |
| [如何更好地调整学习率？](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E5%A6%82%E4%BD%95%E6%9B%B4%E5%A5%BD%E5%9C%B0%E8%B0%83%E6%95%B4%E5%AD%A6%E4%B9%A0%E7%8E%87%EF%BC%9F.md) |
| [如何阅读和学习深度学习项目代码](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E5%A6%82%E4%BD%95%E9%98%85%E8%AF%BB%E5%92%8C%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE%E4%BB%A3%E7%A0%81.md) |
| [快速入门使用tikz绘制深度学习网络图](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E5%BF%AB%E9%80%9F%E5%AD%A6%E4%B9%A0%E4%BD%BF%E7%94%A8tikz%E7%BB%98%E5%88%B6CNN%E7%A4%BA%E6%84%8F%E5%9B%BE.md) |
| [【译】科研敏感性锻炼](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E7%A7%91%E7%A0%94%E6%95%8F%E6%84%9F%E6%80%A7.md) |
| [简易关键点标注软件分享](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E7%AE%80%E6%98%93%E5%85%B3%E9%94%AE%E7%82%B9%E6%A0%87%E6%B3%A8%E8%BD%AF%E4%BB%B6.md) |
| [分布式训练框架Horovod初步学习](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E5%88%86%E5%B8%83/Horovod%E5%88%9D%E6%AD%A5.md) |
| [PyTorch消除训练瓶颈 提速技巧](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E5%B7%A5%E5%85%B7%E7%B1%BB/%E5%88%86%E5%B8%83/PyTorch%E8%AE%AD%E7%BB%83%E5%8A%A0%E9%80%9FDataLoader.md) |











