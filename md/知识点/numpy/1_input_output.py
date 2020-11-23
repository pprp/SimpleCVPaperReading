# TASK 1
# import numpy as np
# a = np.random.randint(0,10,30)
# print(a)
# np.save("./save_format.npy", a)

# TASK 2
# import numpy as np
# a = np.load("./save_format.npy")
# print(a)

# TASK 3
# import numpy as np
# a = np.random.randint(0, 10, 5)
# b = np.sin(a)
# c = np.cos(a)
# print("=="*30)
# print(a, '\n', b, '\n', c)
# print("=="*30)
# np.savez("./savez_format.npz", var_a=a, var_b=b, var_c=c)

# loaded_data = np.load("./savez_format.npz")
# print(loaded_data.files)

# print("=="*30)
# print(loaded_data['var_a'], '\n', loaded_data['var_b'], '\n', loaded_data['var_c'])
# print("=="*30)


# TASK 4
# import numpy as np

# csv_data = np.loadtxt('./csv_format.csv', delimiter=',', skiprows=(1))
# print(csv_data)
# print(np.loadtxt('./csv_format.csv', delimiter=',', skiprows=(1), usecols=(1, 2)))

# v1, v2 = np.loadtxt('./csv_format.csv', delimiter=',',
#                     skiprows=1, usecols=(1, 2), unpack=True)

# print(v1, v2)

# TASK 5
# import numpy as np 
# csv_data = np.genfromtxt('./csv_format2.csv',delimiter=',',names=True)
# print(csv_data['id'])
# print(csv_data['v1'])
# print(csv_data['v2'])
# print(csv_data['v3'])

# TASK 6
# import numpy as np
# # test 1
# np.set_printoptions(precision=4)
# print(np.array([3.1415926]))
# # test 2
# np.set_printoptions(threshold=20)
# print(np.arange(50))
# # test 3
# np.set_printoptions(threshold=np.iinfo(np.int).max)
# print(np.arange(4)**2+np.finfo(float).eps)
# # test 4
# np.set_printoptions(precision=2,suppress=True,threshold=5)
# print(np.linspace(0,1,20))

# WORK 1
# import numpy as np 
# rand_arr = np.random.random([5,3])
# np.set_printoptions(precision=3)
# print(rand_arr)

# # WORK 2
# import numpy as np
# np.set_printoptions(threshold=6)
# print(np.random.randint(0,100,10))

# WORK 3
import numpy as np 
np.set_printoptions(threshold=np.iinfo(np.int).max)
print(np.random.randint(0,100,20))