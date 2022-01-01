# 如何使用logging生成日志

【GiantPandaCV导语】日志对程序执行情况的排查非常重要，通过日志文件，可以快速定位出现的问题。本文将简单介绍使用logging生成日志的方法。

## logging模块介绍

logging是python自带的包，一共有五个level:

- debug: 查看程序运行的信息，调试过程中需要使用。
- info: 程序是否如预期执行的信息。
- warn: 警告信息，但不影响程序执行。
- error: 出现错误，影响程序执行。
- critical: 严重错误

## logging用法

```python
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

logging.info("program start")
```

format参数设置了时间，规定了输出的格式。

```python
import logging
 #先声明一个 Logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
#然后指定其对应的 Handler 为 FileHandler 对象
handler = logging.FileHandler('Alibaba.log')
#然后 Handler 对象单独指定了 Formatter 对象单独配置输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
```

Filehandler是用于将日志写入到文件，如这里将所有日志输出到Alibaba.log文件夹中。



## 参考

https://zhuanlan.zhihu.com/p/56968001