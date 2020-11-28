from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# T1 导入鸢尾属植物数据集，保持文本不变。
data = load_iris()
print(dir(data))

frame = pd.DataFrame(data=data.data, columns=data.feature_names)
print(frame)
print(frame.loc[0])
print(frame['sepal length (cm)'][0])

# T2 求出鸢尾属植物萼片长度的平均值、中位数和标准差（第1列，sepallength）
data_t2 = frame['sepal length (cm)']
print("均值：", np.mean(data_t2))
print("中位数：", np.median(data_t2))
print("标准差：", np.std(data_t2))

# T3 创建一种标准化形式的鸢尾属植物萼片长度，
# 其值正好介于0和1之间，这样最小值为0，最大值为1（第1列，sepallength）
data_min = np.min(data_t2)
data_max = np.max(data_t2)
shifted_data = (data_t2-data_min)/(data_max-data_min)
print(shifted_data)

# T4 找到鸢尾属植物萼片长度的第5和第95百分位数
print(np.percentile(data_t2, [5, 95]))
print('='*50)


# T5 把iris_data数据集中的20个随机位置修改为np.nan值。
# len_data_t2 = len(data_t2)
# choice_array = np.arange(len_data_t2)
# selected = np.random.choice(choice_array, 20)
# print("selected:", selected)
# data_t2[selected] = np.nan
iris_data = data.data
i, j = iris_data.shape
iris_data[np.random.choice(i, size=20), np.random.choice(j, size=20)] = np.nan
print(iris_data[:20])
print('='*50)

# T6 在iris_data的sepallength中查找缺失值的个数和位置（第1列）。
sepallength = iris_data[:, 0]
judge = np.isnan(sepallength)
print(judge)
print("缺失值个数：", len(sepallength[judge]))
print("位置：", np.where(judge))
print('='*50)

# T7 筛选具有 sepallength（第1列）< 5.0
# 并且 petallength（第3列）> 1.5 的 iris_data行
sepallength = iris_data[:, 0]
petallength = iris_data[:, 2]
# selected = np.where(sepallength < 5.0 and petallength > 1.5)
selected = np.where(np.logical_and(sepallength < 5.0, petallength > 1.5))
print(selected)
print(iris_data[selected])

print('='*50)

# T8 选择没有任何 nan 值的 iris_data行。
print(iris_data[np.sum(np.isnan(iris_data), axis=1) == 0])
iris_data = iris_data[np.sum(np.isnan(iris_data), axis=1) == 0]
print('='*50)

# T9 计算 iris_data 中sepalLength（第1列）和petalLength（第3列）之间的相关系数
sepallength = iris_data[:, 0]
petallength = iris_data[:, 2]
print("corrcoef:", np.corrcoef(sepallength, petallength))
print('='*50)

# T10 找出iris_data是否有任何缺失值。
print("缺失值是否存在：")
print(np.sum(np.isnan(iris_data)) > 1)
print('='*50)

# T11 在numpy数组中将所有出现的nan替换为0
# add some nan
i, j = iris_data.shape
iris_data[np.random.choice(i, size=20), np.random.choice(j, size=20)] = np.nan
print(iris_data[:20])
# iris_data[np.where(iris_data == np.nan)] = 0
iris_data[np.isnan(iris_data)] = 0
print(iris_data[:20])
print('='*50)

# T12 找出鸢尾属植物物种中的唯一值和唯一值出现的数量
x = np.unique(iris_data, return_counts=True)
print(x)
print('='*50)

# T13 将 iris_data 的花瓣长度（第3列）以形成分类变量的形式显示。
# 定义：Less than 3 --> ‘small’；3-5 --> ‘medium’；’>=5 --> ‘large’。


def classifier(petallength):
    if petallength < 3:
        return 'small'
    elif petallength < 5:
        return 'medium'
    else:
        return 'large'


petal_all = iris_data[:, 2]

out = [classifier(petal_all[i]) for i in range(len(petal_all))]
print(out)

# T14 在 iris_data 中创建一个新列，
# 其中 volume 是 (pi x petallength x sepallength ^ 2）/ 3。
# TODO
sepallength = iris_data[:, 0]
petallength = iris_data[:, 2]
new_volume = (np.pi*petallength*sepallength**2)/3
print(new_volume.shape)
# create new axis
new_volume = new_volume[:, np.newaxis]
iris_data = np.concatenate([iris_data, new_volume], axis=1)
print(iris_data.shape)

# T15 随机抽鸢尾属植物的种类，
# 使得Iris-setosa的数量是Iris-versicolor和Iris-virginica数量的两倍。

species = np.array(['Iris‐setosa', 'Iris‐versicolor', 'Iris‐virginica'])
species_out = np.random.choice(species, len(iris_data), p=[0.5, 0.25, 0.25])
print(np.unique(species_out, return_counts=True))

# T16 根据 sepallength 列对数据集进行排序。
sepallength = iris_data[:, 0]
index = np.argsort(sepallength)
iris_data = iris_data[index]
print(iris_data)

# T17 在鸢尾属植物数据集中找到最常见的花瓣长度值（第3列）
petallength = iris_data[:, 2]
vals, counts = np.unique(petallength, return_counts=True)
print(vals[np.argmax(counts)])

# T18 在鸢尾花数据集的 petalwidth（第4列）中查找第一次出现的值大于1.0的位置。
petalwidth = iris_data[:,3]
index = np.where(petalwidth>1.0)
print(index)
print(index[0].shape)
print(index[0][0])

