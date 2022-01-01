# 神经网络架构国内外发展现状-信息检索

[TOC]

## [1] 信息检索语言

信息检索语言是用于描述信息系统中的信息的内容特征，常见的信息检索语言包括分类语言和主题语言。就神经网络架构搜索这个问题来说，最好选择主题语言，可以通过借助自然语言，更具有直观性和概念唯一性。而主题语言分为关键词语言和纯自然语言。

选用关键词语言就要挑选神经网络架构搜索的关键词，表征文献主题内容具有实质意义的词语，不要将冠词、介词、副词、连词作为查询的关键词。

## [2] 信息检索技术

采用布尔逻辑检索的方法：

- 使用逻辑运算符将检索词、短语、代码进行逻辑配置
- 指定文献命中条件和组配次序
- 是构造检索最基本的匹配模式，最高效的检索技术。

以神经网络搜索技术为例，搜索 `CNKI中国知网` 数据库，选择高级检索：

![](https://img-blog.csdnimg.cn/20210226092353243.png)

挑选关键词  神经网络架构搜索+NAS+架构搜索

![](https://img-blog.csdnimg.cn/20210226092731698.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_6,color_FFFFFF,t_70)

主题中的加号是高级检索匹配运算符, 规则如下：

> 高级检索支持使用运算符*、+、-、''、""、()进行同一检索项内多个检索词的组合运算，检索框内输入的内容不得超过120个字符。
>
> 输入运算符*(与)、+(或)、-(非)时，前后要空一个字节，优先级需用英文半角括号确定。
>
> 若检索词本身含空格或*、+、-、()、/、%、=等特殊符号，进行多词组合运算时，为避免歧义，须将检索词用英文半角单引号或英文半角双引号引起来。

检索结果：

![](https://img-blog.csdnimg.cn/20210226093027602.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)



在IEEE中检索尝试：

![](https://img-blog.csdnimg.cn/20210226095204965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

搜索非中国人发表文献：

![](https://img-blog.csdnimg.cn/20210226095835722.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

发现非中文发表文献就有5k之多，检索结果如下：

![](https://img-blog.csdnimg.cn/20210226100216300.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

查找中国发表的相关内容，只有300篇，看来国内在这个领域并没有处于领先地位。

![](https://img-blog.csdnimg.cn/20210226100548377.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)


- 截词检索

  - 一般用于引文信息的检索，为了避免检索式过长，一般会在信息检索时使用截词检索。
  - 一般有`*` 和 `?` ： * 代表无限个字符；？代表有限截断，一个字符
- 字段限制检索

  - 将检索词限定到一个或者多个字段中，来检索这些检索字段含有的信息，一般都是检索提名、主题或者全字段。
  - 有题名，篇名
  - 作者单位
  - 摘要
  - 关键词
  - 主题词
  - 全文
  - 作者
  - 书名
  - DOI: Igital Objects Identifier: 数字化对象标识符，通过DOI可以获取该文献对应的元数据、下载链接，进而可以获取文摘信息，全文。
  - 出版年
  - 问下按类型
  - 所有字段
- 聚类检索

  - 自动聚类用户所需要的相关信息，重点完成用户的特性查询，通过几次迭代查询，一般能找到目标的结果。
  - 比如：主题、发表年度、文献来源、机构、基金、文献类型

## [3] 信息检索工具

网络数据库和搜索引擎是最主要的检索工具，国际三大科技文献检索系统：

- 科学引文索引 Science Citation Index SCI
- 工程索引 Engineering Index EI
- 科技会议索引CPCI-S Conference Proceedings Citation Index-Science 

一般去Web of Science查询文献

## [4] 信息检索流程



4.1 分析问题

神经网络架构搜索是近些年兴起的领域，属于计算机科学领域的研究，时间设定可以是2012年-2021年，因为2012年是神经网络兴起的年代，再往前的文献参考价值不大。目前国内外都有研究，国外以谷歌、微软等巨头研究领先，国内也有百度、华为等公司在开展相关业务，所以语种设置可以不限，一般是英文或者中文。

4.2 选择检索工具

这里选择web of science作为检索工具

4.3 拟定检索词

Neural Architecture Search 

NAS

4.4. 编写检索式

TS=(Neural Architecture Search OR NAS)

4.5 获取原文或者文献线索

![](https://img-blog.csdnimg.cn/20210226103142927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

这样会存在问题，如果是多个词，需要用冒号包起来,结果如下：

![](https://img-blog.csdnimg.cn/20210226110856302.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70)

