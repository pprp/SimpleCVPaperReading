# 【Numpy学习】3. Numpy线性代数相关

Numpy 定义了 `matrix` 类型，使用该 `matrix` 类型创建的是矩阵对象，它们的加减乘除运算缺省采用矩阵方式计算，因此用法和Matlab十分类似。但是官方并不推荐在程序中使用 `matrix`,所以仍然用 `ndarray` 来介绍。

[TOC]

## 1. np中的广播机制

参考:https://www.runoob.com/numpy/numpy-broadcast.html

**广播的规则:**

- 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐。
- 输出数组的形状是输入数组形状的各个维度上的最大值。
- 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
- 当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。

**简单理解：**对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：

- 数组拥有相同形状。
- 当前维度的值相等。
- 当前维度的值有一个是 1。

## 2. np.matmul和np.dot的区别

np.matmul: Matrix product of two arrays.

np.dot: alternative matrix product with different broadcasting rules.

主要有两个地方不同：

- np.matmul不允许出现矩阵和标量相乘的情况
- 广播规则不同：总体来说dot广播能力更强，matmul限制更多一点。

dot规则：If *a* is an N-D array and *b* is an M-D array (where `M>=2`), it is a sum product over the last axis of *a* and the second-to-last axis of *b*:

matmul规则： If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

第一个例子：dot和matmul都可以用：

```python
a = np.ones([9, 5, 7, 4])
c = np.ones([9, 5, 4, 3])
print(a.shape)
print(c.shape)

print(np.dot(a, c).shape)
print(np.matmul(a, c).shape)
```

输出：

```
(9, 5, 7, 4)
(9, 5, 4, 3)      
(9, 5, 7, 9, 5, 3)   # doc
(9, 5, 7, 3)    # matmul
```

这里doc规则是将a的最后一列找到，为4，然后在c中找到4在第三列，然后将其前面几维复制出来，得到最终结果（下面会再举几个例子）。

这里matmul在遇见多维的时候，主要关注a的最后一维和c的倒数第二维，这样和普通的矩阵乘法类似。

再来看几个dot可以使用，但matmul不可以使用的例子：

```python
a = np.arange(3*4*5*6).reshape((3,4,5,6))
b = np.arange(3*4*5*6)[::-1].reshape((4,5,1,6,3))
print(np.dot(a, b).shape)
```

输出为：

```
(3, 4, 5, 4, 5, 1, 3)
```

类似的再来看几个例子(这几个例子matmul都无法使用，不满足广播规则)：

```
(3,4,5,6) (5,4,6,3)-> (3,4,5,5,4,3)
(3,4,5,6) (4,5,6,3)-> (3,4,5,4,5,4)
```

相信你有了一定的感觉，遇到矩阵乘法，尽量还是用matmul，因为dot的广播要求太松，虽然可以得到结果，但是很容易和我们预期出现差异。

## 3. 奇异值分解

linalg这个模块中包含了线性代数很多函数

```python
A = np.random.randint(0, 100, size=12).reshape(3, 4)

u, s, v = np.linalg.svd(A, full_matrices=False)

print("A", A)
print("u", u)
print("s", s)
print("v", v)
print("A shape", A.shape)
print("u shape", u.shape)
print("s shape", s.shape)
print("v shape", v.shape)

print(np.matmul(np.matmul(u,np.diag(s)),v))
```

测试结果：

```
A [[ 8 24 55 80]
 [ 3 89 99 17]
 [48 84 35 49]]
u [[-0.45473138 -0.73517602 -0.50272815]
 [-0.68462754  0.64957664 -0.33065892]
 [-0.56965297 -0.19382055  0.79870463]]
s [184.09535094  65.7078928   53.50116434]
v [[-0.17944547 -0.65018618 -0.61232511 -0.41245025]
 [-0.20143797  0.36353579  0.26008576 -0.87156174]
 [ 0.62286533  0.4784395  -0.60616659 -0.12528561]]
A shape (3, 4)
u shape (3, 3)
s shape (3,)
v shape (3, 4)
[[ 8. 24. 55. 80.]
 [ 3. 89. 99. 17.]
 [48. 84. 35. 49.]]
```

## 4. QR分解

QR（正交三角）分解法是求一般矩阵全部特征值的最有效并广泛应用的方法，一般矩阵先经过正交相似变化成为Hessenberg矩阵，然后再应用QR方法求特征值和特征向量。它是将矩阵分解成一个正规正交矩阵Q与上三角形矩阵R，所以称为QR分解法，与此正规正交矩阵的通用符号Q有关。

如果实（复）非奇异矩阵A能够化成正交（酉）矩阵Q与实（复）非奇异上三角矩阵R的乘积，即A=QR，则称其为A的QR分解。

> ps: 奇异矩阵代表矩阵的秩不是满秩，非奇异矩阵代表满秩矩阵。
>
> 奇异矩阵AX=0有无穷解，AX=b有五穷街或者无解。
>
> 非奇异矩阵Ax=0只有唯一0解，AX=b有唯一解。

测试：

```python
A = np.array([[2, -2, 3], [1, 1, 1], [1, 3, -1]])
print(A)
# [[ 2 -2  3]
#  [ 1  1  1]
#  [ 1  3 -1]]

q, r = np.linalg.qr(A)
print(q.shape)
print(q)

print(r.shape)
print(r)

print(np.dot(q, r))

a = np.allclose(np.dot(q.T, q), np.eye(3))
print(a)
```

结果：

```
[[ 2 -2  3] 
 [ 1  1  1] 
 [ 1  3 -1]]
(3, 3)
[[-0.81649658  0.53452248  0.21821789] 
 [-0.40824829 -0.26726124 -0.87287156] 
 [-0.40824829 -0.80178373  0.43643578]]
(3, 3)
[[-2.44948974  0.         -2.44948974] 
 [ 0.         -3.74165739  2.13808994] 
 [ 0.          0.         -0.65465367]]
[[ 2. -2.  3.]
 [ 1.  1.  1.]
 [ 1.  3. -1.]]
True
```

## 5. Cholesky 分解

Cholesky 分解是把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。它要求矩阵的所有特征值必须大于零，故分解的下三角的对角元也是大于零的。Cholesky分解法又称平方根法，是当A为实对称正定矩阵时，LU三角分解法的变形。

测试：

```python
A = np.array([[1, 1, 1, 1], [1, 3, 3, 3],
              [1, 3, 5, 5], [1, 3, 5, 7]])
print(np.linalg.eigvals(A)) # 特征值全部非0
L = np.linalg.cholesky(A)
print(L.shape)
print(np.dot(L,L.T))
```

结果：

```
[13.13707118  1.6199144   0.51978306  0.72323135]
(4, 4)
[[1. 1. 1. 1.]
 [1. 3. 3. 3.]
 [1. 3. 5. 5.]
 [1. 3. 5. 7.]]
```

## 6. 线代中几个性质

### 6.1 范数

- `numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)` 计算向量或者矩阵的范数。

  ![](http://datawhale.club/uploads/default/original/1X/3848231829634ed07c89fd313297a9358916a226.png)

测试：

```python
import numpy as np

x = np.array([1, 2, 3, 4])

print(np.linalg.norm(x, ord=1)) 
# 10.0
print(np.sum(np.abs(x)))  
# 10

print(np.linalg.norm(x, ord=2))  
# 5.477225575051661
print(np.sum(np.abs(x) ** 2) ** 0.5)  
# 5.477225575051661

print(np.linalg.norm(x, ord=-np.inf))  
# 1.0
print(np.min(np.abs(x)))  
# 1

print(np.linalg.norm(x, ord=np.inf))  
# 4.0
print(np.max(np.abs(x)))  
# 4

A = np.array([[1, 2, 3, 4], [2, 3, 5, 8],
              [1, 3, 5, 7], [3, 4, 7, 11]])

print(A)
# [[ 1  2  3  4]
#  [ 2  3  5  8]
#  [ 1  3  5  7]
#  [ 3  4  7 11]]

print(np.linalg.norm(A, ord=1))  # 30.0
print(np.max(np.sum(A, axis=0)))  # 30

print(np.linalg.norm(A, ord=2))  
# 20.24345358700576
print(np.max(np.linalg.svd(A, compute_uv=False)))  
# 20.24345358700576

print(np.linalg.norm(A, ord=np.inf))  # 25.0
print(np.max(np.sum(A, axis=1)))  # 25

print(np.linalg.norm(A, ord='fro'))  
# 20.273134932713294
print(np.sqrt(np.trace(np.dot(A.T, A))))  
# 20.273134932713294
```

### 6.2 行列式

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
# [[1 2]
#  [3 4]]

print(np.linalg.det(x))
```

### 6.3 秩

```python
import numpy as np

I = np.eye(3)  # 先创建一个单位阵
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

r = np.linalg.matrix_rank(I)
print(r)  # 3

I[1, 1] = 0  # 将该元素置为0
print(I)
# [[1. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 1.]]

r = np.linalg.matrix_rank(I)  # 此时秩变成2
print(r)  # 2
```

### 6.4 迹

方阵的迹就是主对角元素之和。

```python
import numpy as np

x = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
print(x)
# [[1 2 3]
#  [3 4 5]
#  [6 7 8]]

y = np.array([[5, 4, 2], [1, 7, 9], [0, 4, 5]])
print(y)
# [[5 4 2]
#  [1 7 9]
#  [0 4 5]]

print(np.trace(x))  # A的迹等于A.T的迹
# 13
print(np.trace(np.transpose(x)))
# 13

print(np.trace(x + y))  # 和的迹 等于 迹的和
# 30
print(np.trace(x) + np.trace(y))
# 30
```

### 6.5 逆矩阵

设 A 是数域上的一个 n 阶矩阵，若在相同数域上存在另一个 n 阶矩阵 B，使得：`AB=BA=E`（E 为单位矩阵），则我们称 B 是 A 的逆矩阵，而 A 则被称为可逆矩阵。

```python
A = np.array([[1, -2, 1], [0, 2, -1], [1, 1, -2]])
print(A)
# [[ 1 -2  1]
#  [ 0  2 -1]
#  [ 1  1 -2]]

# 求A的行列式，不为零则存在逆矩阵
A_det = np.linalg.det(A)  
print(A_det)
# -2.9999999999999996

A_inverse = np.linalg.inv(A)  # 求A的逆矩阵
print(A_inverse)
# [[ 1.00000000e+00  1.00000000e+00 -1.11022302e-16]
#  [ 3.33333333e-01  1.00000000e+00 -3.33333333e-01]
#  [ 6.66666667e-01  1.00000000e+00 -6.66666667e-01]]

x = np.allclose(np.dot(A, A_inverse), np.eye(3))
print(x)  # True
x = np.allclose(np.dot(A_inverse, A), np.eye(3))
print(x)  # True

A_companion = A_inverse * A_det  # 求A的伴随矩阵
print(A_companion)
# [[-3.00000000e+00 -3.00000000e+00  3.33066907e-16]
#  [-1.00000000e+00 -3.00000000e+00  1.00000000e+00]
#  [-2.00000000e+00 -3.00000000e+00  2.00000000e+00]]
```

### 6.6 特征值&特征向量

$$
Ax=\lambda x
$$

其中$\lambda$是特征值，对应的x是特征向量。

- `numpy.linalg.eig(a)` 计算方阵的特征值和特征向量。
- `numpy.linalg.eigvals(a)` 计算方阵的特征值。

简单例子：

```python
x = np.diag((1, 2, 3))  
print(x)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

print(np.linalg.eigvals(x))
# [1. 2. 3.]

a, b = np.linalg.eig(x)  
# 特征值保存在a中，特征向量保存在b中
print(a)
# [1. 2. 3.]
print(b)
```

其他方法计算：

**幂法**

```python
a = [[4,-1,1],[-1,3,-2],[1,-2,3]]#系数矩阵
A = np.mat(a)
N = len(A)
# U  = np.mat([1]*N).T
# V  = np.mat([1]*N).T
U  = np.mat([0,1,1]).T
V  = np.mat([0,1,1]).T
sigma = 0.000001#精度
M = 1000#最大迭代次数
m = 1
k = 0
before = m
while(k<M):
    before = m
    V = A*U
    m = max(V)#[0,0]
    U = V/m
    if(abs(m-before)<sigma):
        print("特征值为",(m),sep='\n')
        print("特征向量为",(U),sep='\n')
        break
    k += 1
if k==M:
    print('计算失败，k=',k)
```

**反幂法**

```python
import numpy as np
a = [[4,-1,1],[-1,3,-2],[1,-2,3]]#系数矩阵
N = len(a)

A = np.mat(a)
U  = np.mat([1]*N).T
V  = np.mat([1]*N).T
sigma = 0.000001#精度
M = 1000#最大迭代次数
m = 1
k = 0
before = m
while(k<M):
    before = m
    V = A.I*U
    m = max(V)[0,0]
    U = V/m
    if(abs(m-before)<sigma):
        print("特征值为",(1/m),sep='\n')
        print("特征向量为",(U),sep='\n')
        break
    k += 1
if k==M:
    print('计算失败，k=',k)
```

**原点平移加速法**

```python
import numpy as np
# a = [[4,-1,1],[-1,3,-2],[1,-2,3]]#系数矩阵
a = [[4,-1,1],[-1,3,-2],[1,-2,3]]#系数矩阵
N = len(a)
#原点加速法:
step = 2.5
for i in range(N):
    a[i][i] -=step
A = np.mat(a)

U  = np.mat([1]*N).T
V  = np.mat([1]*N).T
sigma = 0.000001#精度
M = 1000#最大迭代次数
m = 1
k = 0
before = m
while(k<M):
    before = m
    V = A*U
    m = max(V)[0,0]
    U = V/m
    if(abs(m-before)<sigma):
        print("特征值为",(1/m),sep='\n')
        print("特征向量为",(U),sep='\n')
        break
    k += 1
if k==M:
    print('计算失败，k=',k)
```




## 7. 求解方程组

```python
#  x + 2y +  z = 7
# 2x -  y + 3z = 7
# 3x +  y + 2z =18

import numpy as np

A = np.array([[1, 2, 1], [2, -1, 3], [3, 1, 2]])
b = np.array([7, 7, 18])
x = np.linalg.solve(A, b)
print(x)  # [ 7.  1. -2.]

x = np.linalg.inv(A).dot(b)
print(x)  # [ 7.  1. -2.]

y = np.allclose(np.dot(A, x), b)
print(y)  # True
```

## 8. 练习

提供三个例子进行练习

```python
# TEST 1 计算两个数组a和数组b之间的欧氏距离。
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])
print(np.sqrt(np.sum((a-b)**2)))
print(np.linalg.norm(a-b, ord=2))

# TEST 2 给定矩阵A和数组b，求解线性方程组：
A = np.array([[1, -2, 1], [0, 2, -8], [-4, 5, 9]])
b = np.array([0, 8, -9])
x = np.linalg.solve(A, b)
print(x)
print(np.dot(A, x))
print(np.matmul(np.linalg.inv(A), b))


print('='*50)
# TEST 3 给定矩阵A[[4,-1,1],[-1,3,-2],[1,-2,3]]计算特征值和特征向量
A = np.array([[4, -1, 1],
              [-1, 3, -2],
              [1, -2, 3]])
lambda_x,v = np.linalg.eig(A)
print(lambda_x)
print(v)
```









