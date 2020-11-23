# 【Numpy学习】第一节 Numpy输入输出

【前言】学习了保存numpy数组、文本文件、控制输出格式等函数。

## 0. 知识点总结

![20201123220310405.png (2671×3132) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201123220310405.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)



## 1. 测试save

```python
import numpy as np 
a = np.random.randint(0,10,30)
print(a)
np.save("./save_format.npy", a)
```

![用vscode打开结果](https://img-blog.csdnimg.cn/20201123204603339.png#pic_center)

由于npy是二进制文件，所以打开以后除了一些头部信息，其他都是乱码。

## 2. 测试load

```python
import numpy as np
a = np.load("./save_format.npy")
print(a)
```

![20201123204834714.png (643×181) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201123204834714.png#pic_center)

## 3. 测试savez

```python
import numpy as np
a = np.random.randint(0, 10, 5)
b = np.sin(a)
c = np.cos(a)
print("=="*30)
print(a, '\n', b, '\n', c)
print("=="*30)
np.savez("./savez_format.npz", var_a=a, var_b=b, var_c=c)

loaded_data = np.load("./savez_format.npz")
print(loaded_data.files)

print("=="*30)
print(loaded_data['var_a'], '\n', loaded_data['var_b'], '\n', loaded_data['var_c'])
print("=="*30)
```

测试savez保存为npz文件，并用winrar进行打开，结果如下：

![解压结果](https://img-blog.csdnimg.cn/20201123205904797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

整体输入输出如下：

![20201123205738225.png (669×277) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201123205738225.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

## 4. csv文件读写测试

令csv_format.csv文件内容如下：

```python
id,value1,value2,value3
1,123,1.4,23
2,110,0.5,18
3,164,2.1,19
```

测试代码如下：

```python
import numpy as np

csv_data = np.loadtxt('./csv_format.csv', delimiter=',', skiprows=(1))
print(csv_data)
print(np.loadtxt('./csv_format.csv', delimiter=',', skiprows=(1), usecols=(1, 2)))

v1, v2 = np.loadtxt('./csv_format.csv', delimiter=',',
                    skiprows=1, usecols=(1, 2), unpack=True)

print(v1, v2)
```

输出：

```python
[[  1.  123.    1.4  23. ]
 [  2.  110.    0.5  18. ]
 [  3.  164.    2.1  19. ]]
[[123.    1.4]
 [110.    0.5]
 [164.    2.1]]
[123. 110. 164.] [1.4 0.5 2.1]
```

## 5. genfromtxt测试

csv_format2.csv文件

```txt
id,v1,v2,v3
1,123,,23
2,,0.5,18
3,164,2.1,
```

测试代码：

```python
import numpy as np 
csv_data = np.genfromtxt('./csv_format2.csv',delimiter=',',names=True)
print(csv_data['id'])
print(csv_data['v1'])
print(csv_data['v2'])
print(csv_data['v3'])
```

输出结果：

```txt
[1. 2. 3.]
[123.  nan 164.]
[nan 0.5 2.1]
[23. 18. nan]
```

## 6. 文本格式测试

测试代码：

```python
import numpy as np
# test 1
np.set_printoptions(precision=4)
print(np.array([3.1415926]))
# test 2
np.set_printoptions(threshold=20)
print(np.arange(50))
# test 3
np.set_printoptions(threshold=np.iinfo(np.int).max)
print(np.arange(4)**2+np.finfo(float).eps)
# test 4
np.set_printoptions(precision=2,suppress=True,threshold=5)
print(np.linspace(0,1,20))
```

输出结果：

```
[3.1416]
[ 0  1  2 ... 47 48 49]
[2.2204e-16 1.0000e+00 4.0000e+00 9.0000e+00]
[0.   0.05 0.11 ... 0.89 0.95 1.  ]
```

## 7. 练习

No. 1 **只打印或显示numpy数组rand_arr的小数点后3位。**

- `rand_arr = np.random.random([5, 3])`

```python
# WORK 1
import numpy as np 
rand_arr = np.random.random([5,3])
np.set_printoptions(precision=3)
print(rand_arr)
```

输出结果：

```python
[[0.766 0.817 0.416]
 [0.059 0.408 0.651]
 [0.137 0.842 0.024]
 [0.462 0.377 0.925]
 [0.825 0.241 0.201]]
```

No.2 将numpy数组a中打印的项数限制为最多6个元素。

```python
# WORK 2
import numpy as np
np.set_printoptions(threshold=6)
print(np.random.randint(0,100,10))
```

输出结果：

```python
[22  1 87 ... 17 38 59]
```

No. 3 打印完整的numpy数组a而不中断。

```python
# WORK 3
import numpy as np 
np.set_printoptions(threshold=np.iinfo(np.int).max)
print(np.random.randint(0,100,20))
```

输出结果：

```
[12  5 91 24 66 72 65  8 13 36 78 27 81 42  3 61 81 62  7  8]
```

## 8. 参考

[numpy.genfromtxt的用法_终南小道的博客-CSDN博客](https://blog.csdn.net/weixin_41811657/article/details/84614818)

[numpy组队学习下-输入和输出 - 组队学习 / 编程实践（Numpy） - Datawhale](http://datawhale.club/t/topic/178)
