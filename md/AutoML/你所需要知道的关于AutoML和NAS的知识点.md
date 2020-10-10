# 你所需要知道的关于AutoML和NAS的知识点

【GiantPandaCV导读】本文是笔者第一次进行翻译国外博客，第一次尝试，由于水平的限制，可能有的地方翻译表达的不够准确，在翻译过程中尽量还原作者的意思，如果需要解释的部分会在括号中添加，如有问题欢迎指正。本文翻译的是《Everything you need to know about AutoML and Neural Architecture Search》获得了4.8k的高赞，值得一读。

> 作者：George Seif
>
> 翻译：pprp
>
> 日期：2018/8/21

AutoML和NAS是深度学习领域的新秀。不需要过多的工作量，他们可以使用最暴力的方式让你的机器学习任务达到非常高的准确率。既简单又有效率。

那么AutoML和NAS是如何起作用的呢？如何使用这种工具？

## Neural Architecture Search

神经网络架构搜索，简称NAS。开发一个神经网络模型往往需要大量的工程架构方面的设计。有时候您可以通过迁移学习完成一个任务，但是如果想要有更好的性能，最好设计自己的网络。为自己的任务设计网络架构需要非常专业的技能，并且总体上非常有挑战性。我们很可能不知道当前最新技术的局限在哪里（STOA技术的瓶颈我们并不清楚），所以会进行很多试错，这样的话非常浪费时间和金钱。

为了解决这个问题，NAS被提出来了，这是一种可以搜索最好的神经网络结构的算法。大多数算法都是按照以下方式进行的：

1. 首先定义一个Building Blocks的集合，集合中元素代表的是可能用于神经网络搜索的基本单元。比如说NASNet中提出了以下Building Block。

![image-20201010204933174](%E4%BD%A0%E6%89%80%E9%9C%80%E8%A6%81%E7%9F%A5%E9%81%93%E7%9A%84%E5%85%B3%E4%BA%8EAutoML%E5%92%8CNAS%E7%9A%84%E7%9F%A5%E8%AF%86%E7%82%B9.assets/image-20201010204933174.png)









## 英文原文

链接：https://towardsdatascience.com/everything-you-need-to-know-about-automl-and-neural-architecture-search-8db1863682bf