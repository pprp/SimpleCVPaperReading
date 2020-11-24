# 【Numpy学习】第二节 Numpy随机抽样

## 0. 内容总结

![20201124113908393.png (2848×4123) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124113908393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)



## 1. 石油勘探

野外正在进行9（n=9）口石油勘探井的发掘工作，每一口井能够开发出油的概率是0.1（p=0.1）。请问，最终所有的勘探井都勘探失败的概率？

```ptyhon
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
x = np.random.binomial(9, 0.1, size=50000)
plt.hist(x)
plt.xlabel('随机变量：成功次数')
plt.ylabel('次数')
plt.show()
```

结果如下：![20201124102249441.png (717×529) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124102249441.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 2. 投硬币问题

模拟投硬币，投2次，请问两次都为正面的概率？

```python
import numpy as np 
import matplotlib.pyplot as plt  

np.random.seed(0)

x = np.random.binomial(2, 0.5, size=100000)

print(np.sum(x==2)/100000)

plt.hist(x)
plt.show()
```

结果如下：

![20201124102811142.png (745×531) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124102811142.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 3. 泊松分布

假定某航空公司预定票处平均每小时接到42次订票电话，那么10分钟内恰好接到6次电话的概率是多少？

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x = np.random.poisson(42/6, size=10000)
print(np.sum(x == 6)/10000)

plt.hist(x)
plt.show()
```

结果如下：0.1526

![20201124103711762.png (743×529) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124103711762.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 4. 超几何分布

一共20只动物里有7只是狗，抽取12只有3只狗的概率（无放回抽样）

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x = np.random.hypergeometric(7, 13, 12, size=1000000)
print(np.sum(x == 3)/1000000)

plt.hist(x, bins=8)
plt.show()
```

结果如下：

![20201124105518870.png (715×503) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124105518870.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 5. 均匀分布

代码：

```python
import numpy as np 
import matplotlib.pyplot as plt 
x = np.random.uniform(0,10,50000)
y = np.random.uniform(size=(4,3))
plt.hist(x,bins=10)
plt.show()
```

测试结果：

![20201124105809250.png (712×524) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124105809250.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 6. 正态分布

```python
import numpy as np
x = np.random.randn(5,5,5,5)
print(x.shape)
```

输出：

```
[5,5,5,5]
```

指定均值和标准差：

```python
import numpy as np
mu = 5
sigma=0.5
x = np.random.normal(mu, sigma, size=(2,4))
```

## 7. 指数分布

```python
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(0)
lam=7
x=np.random.exponential(1/lam, size=5000)
plt.hist(x)
plt.show()
```

结果如下：![20201124112315950.png (718×527) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124112315950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 8. 其他

**choice从序列中随机挑选**

```python
import numpy as np

x = np.random.choice(10, 5, replace=True)
print(x)

x = np.random.choice(4, 100, p=[0.1,0.2,0.3,0.4])
print(x)
```

**shuffle洗牌操作**

```python
import numpy as np
x = np.arange(10)
print(x)
np.random.shuffle(x)
print(x)

x = np.random.randn(3,2,2)
print(x)
np.random.shuffle(x)
print(x)

x = np.arange(15).reshape(3,5)
print(x)
y = np.random.permutation(x)
print(x)
print(y)
```

输出结果：

```txt
[0 1 2 3 4 5 6 7 8 9]
[3 1 2 9 4 6 7 8 0 5]
[[[ 0.32893857  0.8333537 ]
  [ 0.49781583 -1.96317531]]

 [[ 2.74954043 -1.15290717]
  [-0.65952234 -0.34419525]]

 [[-0.44873248 -0.97622593]
  [ 0.36281789  1.24704471]]]
[[[-0.44873248 -0.97622593]
  [ 0.36281789  1.24704471]]

 [[ 2.74954043 -1.15290717]
  [-0.65952234 -0.34419525]]

 [[ 0.32893857  0.8333537 ]
  [ 0.49781583 -1.96317531]]]
  
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
[[ 0  1  2  3  4]
 [10 11 12 13 14]
 [ 5  6  7  8  9]]
```

