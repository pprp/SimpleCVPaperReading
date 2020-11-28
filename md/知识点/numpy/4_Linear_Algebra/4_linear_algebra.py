import numpy as np

# x1 = np.linspace(1, 11, 6).reshape(3, 2)
# x2 = np.linspace(2, 12, 6).reshape(3, 2)
# y1 = np.linspace(1, 11, 9).reshape(3, 3)
# y2 = np.linspace(2, 12, 9).reshape(3, 3)
# print(x1, '\n -----------\n', x2)
# print('-'*50)
# print(x1*x2)
# print(np.multiply(x1, x2))
# print('-'*50)
# print(y1, '\n -----------\n', y2)
# print('-'*50)
# print(np.dot(y1, y2))
# print(np.matmul(y1, y2))
# print(y1@y2)

# TEST 2
# a = np.ones([9, 5, 7, 4])
# c = np.ones([9, 5, 4, 3])
# print(a.shape)
# print(c.shape)

# print(np.dot(a, c).shape)
# print(np.matmul(a, c).shape)

# a = np.arange(3*4*5*6).reshape((3,4,5,6))
# b = np.arange(3*4*5*6)[::-1].reshape((4,5,1,6,3))
# print(np.dot(a, b).shape)
# print(np.matmul(a, b).shape)

# TEST 3 SVD
# A = np.random.randint(0, 100, size=12).reshape(3, 4)

# u, s, v = np.linalg.svd(A, full_matrices=False)

# print("A", A)
# print("u", u)
# print("s", s)
# print("v", v)
# print("A shape", A.shape)
# print("u shape", u.shape)
# print("s shape", s.shape)
# print("v shape", v.shape)

# print(np.matmul(np.matmul(u,np.diag(s)),v))

# TEST 4 QR

# A = np.array([[2, -2, 3], [1, 1, 1], [1, 3, -1]])
# print(A)
# # [[ 2 -2  3]
# #  [ 1  1  1]
# #  [ 1  3 -1]]

# q, r = np.linalg.qr(A)
# print(q.shape)
# print(q)

# print(r.shape)
# print(r)

# print(np.dot(q, r))

# a = np.allclose(np.dot(q.T, q), np.eye(3))
# print(a)

# TEST 5 Cholesky分解
# A = np.array([[1, 1, 1, 1], [1, 3, 3, 3],
#               [1, 3, 5, 5], [1, 3, 5, 7]])
# print(np.linalg.eigvals(A)) # 特征值全部非0
# L = np.linalg.cholesky(A)
# print(L.shape)
# print(np.dot(L,L.T))

# TEST 6 范数

# import numpy as np

# x = np.array([1, 2, 3, 4])

# print(np.linalg.norm(x, ord=1))
# # 10.0
# print(np.sum(np.abs(x)))
# # 10

# print(np.linalg.norm(x, ord=2))
# # 5.477225575051661
# print(np.sum(np.abs(x) ** 2) ** 0.5)
# # 5.477225575051661

# print(np.linalg.norm(x, ord=-np.inf))
# # 1.0
# print(np.min(np.abs(x)))
# # 1

# print(np.linalg.norm(x, ord=np.inf))
# # 4.0
# print(np.max(np.abs(x)))
# # 4

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


