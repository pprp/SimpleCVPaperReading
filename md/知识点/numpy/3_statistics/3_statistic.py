import numpy as np

# TEST 1
# x = np.random.randint(0,100,20).reshape(4,5)
# print(x)
# print(np.amin(x))# 数组所有元素中最小值
# print(np.amin(x,axis=0))# 数组沿着0轴找到每一行最小
# print(np.amin(x,axis=1))# 数组沿着1轴找到每一列最小

# x = np.random.randint(0, 10, 3*4*5).reshape(3, 4, 5)
# print(x)
# print('-'*50)
# print(np.amin(x, axis=0))
# print('-'*50)
# print(np.amin(x, axis=1))
# print('-'*50)
# print(np.amin(x, axis=2))


# TEST 2
# import numpy as np

# x = np.array([[11, 12, 13, 14, 15],
#               [16, 17, 18, 19, 20],
#               [21, 22, 23, 24, 25],
#               [26, 27, 28, 29, 30],
#               [31, 32, 33, 34, 35]])
# print(x.size)
# print(np.mean(x))
# print(np.var(x))
# print(np.mean((x-np.mean(x))**2))
# # 无偏估计
# print(np.sum((x-np.mean(x))**2)/(x.size-1))
# print(np.var(x, ddof=1))
# # axis测试
# print(np.var(x,axis=0))
# print(np.var(x,axis=1))


# TEST 3
# x = np.array([[11, 12, 13, 14, 15],
#               [16, 17, 18, 19, 20],
#               [21, 22, 23, 24, 25],
#               [26, 27, 28, 29, 30],
#               [31, 32, 33, 34, 35]])

# print(np.std(x))
# print(np.sqrt(np.var(x)))
# print(np.std(x,axis=0))
# print(np.std(x,axis=1))

# TEST 4
# import numpy as np

# x = np.random.randint(0, 20, size=[4, 5])
# print(x)

# print(np.ptp(x))
# print(np.ptp(x, axis=0))
# print(np.ptp(x, axis=1))


# TEST 5
# x = np.random.randint(0,20,[4,5])
# print(x)
# print(np.percentile(x, [25,50]))
# x = x.reshape(-1)
# print(x)
# print(np.sort(x))
# print(np.percentile(x, [25,50]))

# TEST 6
# x = np.random.randint(0, 100, [3, 7])
# print(np.sort(x))
# print(np.median(x))
# print(np.mean(x))
# print(np.average(x))
# print(np.mean(x, axis=0))
# print(np.average(x, axis=0))

# w = np.arange(1, 22).reshape(3, 7)
# print(np.average(x, weights=w))

# TEST 7
# x = np.arange(1, 8)
# y = np.arange(8, 15)
# print(x, y)
# print('-'*50)
# print(np.var(x))
# print(np.cov(x))
# print(np.var(x, ddof=1))
# print('-'*50)
# print(np.var(y))
# print(np.cov(y))
# print(np.var(y, ddof=1))
# print('-'*50)
# print(np.cov(x, y))
# print('-'*50)
# z = np.mean((x - np.mean(x)) * (y - np.mean(y)))  # 协方差
# print(z)

# z = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)  # 样本协方差
# print(z)

# z = np.dot(x - np.mean(x), y - np.mean(y)) / (len(x) - 1)  # 样本协方差
# print(z)

# TEST 8
# import numpy as np

# x, y = np.random.randint(0, 20, size=(2, 4))

# print(x)  
# print(y)  

# z = np.corrcoef(x, y)
# print(z)

# a = np.dot(x - np.mean(x), y - np.mean(y))
# b = np.sqrt(np.dot(x - np.mean(x), x - np.mean(x)))
# c = np.sqrt(np.dot(y - np.mean(y), y - np.mean(y)))
# print(a / (b * c)) 

# TEST 9
# x = np.array([0.2, 6.4, 3.0, 1.6])

# bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

# inds = np.digitize(x, bins)

# print(inds)  # [1 4 3 2]
# for n in range(x.size):
#     print(bins[inds[n] - 1], "<=", x[n], "<", bins[inds[n]])

# WORK 1
a = np.random.randint(1,10,[5,3])
print(a)
print(np.amax(a, axis=1))