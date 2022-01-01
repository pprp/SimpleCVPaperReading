# SimplePaperReading
分享公众号GiantPandaCV中的博客＆"神经网络架构搜索"中的博客



## 目录

- [神经网络结构搜索](#神经网络结构搜索)
- [注意力机制](#注意力机制)
- [Transformer](#Transformer)
- [目标检测](#目标检测)
  - [YOLOv3系列](##YOLOv3系列)
  - [CenterNet系列](##CenterNet系列)
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
| [**【神经网络搜索】Efficient Neural Architecture Search**](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91Efficient%20Neural%20Architecture%20Search.md) |
| [ICLR 2021 NAS 相关论文(包含Workshop)](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91ICLR%202021%20NAS%20%E7%9B%B8%E5%85%B3%E8%AE%BA%E6%96%87(%E5%8C%85%E5%90%ABWorkshop).md) |
| [NAS-RL（ICLR2017）](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NAS-RL.md) |
| [神经网络架构国内外发展现状](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NAS%E5%9B%BD%E5%86%85%E5%A4%96%E5%8F%91%E5%B1%95%E7%8E%B0%E7%8A%B6.md) |
| [【神经网络搜索】NAS总结](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NAS%E6%80%BB%E7%BB%93.md) |
| [【神经网络架构搜索】NAS-Bench-101: 可复现神经网络搜索](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NasBench101.md) |
| [【神经网络搜索】NasBench301 使用代理模型构建Benchmark](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91NasBench301_pprp.md) |
| [【神经网络搜索】Once for all](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91Once%20for%20all.md) |
| [PC-DARTS 部分通道链接](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91PC-DARTS%20%E9%83%A8%E5%88%86%E9%80%9A%E9%81%93%E9%93%BE%E6%8E%A5.md) |
| [**【神经网络搜索】ProxyLessNAS**](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2/%E3%80%90%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%90%9C%E7%B4%A2%E3%80%91ProxyLessNAS.md) |
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
| [**卷积神经网络中的即插即用模块**](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E5%8D%B3%E6%8F%92%E5%8D%B3%E7%94%A8%E6%A8%A1%E5%9D%97.md) |
| [神经网络加上注意力机制，精度反而下降，为什么会这样呢？](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8A%A0%E4%B8%8A%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%EF%BC%8C%E7%B2%BE%E5%BA%A6%E5%8F%8D%E8%80%8C%E4%B8%8B%E9%99%8D%EF%BC%9F.md) |



## Transformer

| 博客题目                                                     |
| ------------------------------------------------------------ |
| [A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/A%20Battle%20of%20Network%20Structure%20MLP%20Transformer%20CNN_pprp.md) |
| [BoTNet:Bottleneck Transformers for Visual Recognition](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/BoTNet_Bottleneck%20Transformer_pprp.md) |
| [CvT: 如何将卷积的优势融入Transformer](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/CvT_pprp.md) |
| [DeiT：使用Attention蒸馏Transformer](https://github.com/pprp/SimpleCVPaperAbstractReading/blob/master/md/Transformer/DeiT_pprp.md) |



## 目标检测



### YOLOv3系列



### CenterNet系列





## 工具集













