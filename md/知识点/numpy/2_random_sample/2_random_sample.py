# SAMPLE 1
# import matplotlib.pyplot as plt
# import numpy as np
# np.random.seed(0)
# x = np.random.binomial(9, 0.1, size=50000)
# plt.hist(x)
# plt.xlabel('随机变量：成功次数')
# plt.ylabel('次数')
# plt.show()

# SAMPLE 2
# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(0)

# x = np.random.binomial(2, 0.5, size=100000)

# print(np.sum(x==2)/100000)

# plt.hist(x)
# plt.show()

# SAMPLE 3
# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(0)
# x = np.random.poisson(42/6, size=10000)
# print(np.sum(x == 6)/10000)

# plt.hist(x)
# plt.show()

# SAMPLE 4
# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(0)
# x = np.random.hypergeometric(7, 13, 12, size=1000000)
# print(np.sum(x == 3)/1000000)

# plt.hist(x, bins=8)
# plt.show()

# SAMPLE 4
# import numpy as np 
# import matplotlib.pyplot as plt 
# x = np.random.uniform(0,10,50000)
# plt.hist(x,bins=10)
# plt.show()

# SAMPLE 5
# import numpy as np
# x = np.random.randn(5,5,5,5)
# print(x.shape)

# SAMPLE 6
# import numpy as np
# import matplotlib.pyplot as plt 
# np.random.seed(0)
# lam=7
# x=np.random.exponential(1/lam, size=5000)
# plt.hist(x)
# plt.show()

# SAMPLE 7
# import numpy as np

# x = np.random.choice(10, 5, replace=True)
# print(x)

# x = np.random.choice(4, 100, p=[0.1,0.2,0.3,0.4])
# print(x)

# SAMPLE 8
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