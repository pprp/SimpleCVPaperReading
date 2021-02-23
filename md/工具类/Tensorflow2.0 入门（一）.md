## Tensorflow2.0 入门

**计算图分为动态图和静态图**

- 静态图框架：Tensorflow1.x和Caffe，预先定义好计算图，运行的时候反复使用，不能改变。
  - 特点：速度快，适合大规模部署、适合嵌入式平台。
  - 定义静态图需要使用新的语法，不能使用if,while,for-loop等结构，将所有分支都考虑到，所以静态图可能会很庞大。

- 动态图框架：Tensorflow2.x和PyTorch，可以对预定义计算图进行修改。
  - 特点：灵活、便于debug、学习成本低。
  - 兼容python的各种逻辑控制语法，最终创建的图取决于每次运行时的条件分支选择。





**Eager Execution:**

Tensorflow 1.x 是传统执行模式，先绘制好计算图，利用会话**session**进行运行计算图，执行过程看不到中间结果，只能得到最终的输出结果，不利于调试。

Tensorflow 2.x 是Eager Execution, 依次执行程序中语句，可以看到程序中间结果，方便调试（和PyTorch有点类似）





**AutoGraph机制**

TF2.0开始，默认使用动态图模型，希望用autograph机制将tf2.0转化为tf1.0，用于部署。

- 给python函数加上@tf.function装饰器
- 这时会调用autograph，将python函数转化为等价的计算图。

自己找资料额外了解





**基本编程模型：**

数据准备：Numpy, tf.data 

```python
dataset = tf.data.Dataset.from_tensor_slices((x,y))
for x, y in dataset:
    print(x,y)
```

tf.tensor转numpy: xxx.numpy()

```python
(train_data, train_label), (_, _) = tf.keras.datasets.
mnist.load_data()
train_data = np.expand_dims(train_data.astype(np.
float32) / 255.0, axis=−1) # [60000, 28, 28, 1]
mnist_dataset = tf.data.Dataset.from_tensor_slices((
train_data, train_label))
```

数据预处理：tf.data.Dataset.map(f)

```python
def rot90(image, label):
image = tf.image.rot90(image)
return image, label
mnist_dataset = mnist_dataset.map(rot90)
```

tf.data.Dataset.shuffle(bufferSize) 将数据集打乱，从缓冲区中随机采样，采样后数据被替换。

tf.data.Dataset.batch(batchSize) 将数据集分为多个batch

```python
mnist_dataset = mnist_dataset.shuffle(buffer_size=10000).batch(4)
for images, labels in mnist_dataset:
    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        axs[i].set_title(labels.numpy()[i])
        axs[i].imshow(images.numpy()[i, :, :, 0])
    plt.show()
```









模型构建：tf.keras 

有几种方式构建计算图：

- 低阶API实现
  - 简单直观
  - 复杂模型的实现太过复杂

- 高阶keras接口
  - 便于用户使用
  - 模块化和组合
  - 易于扩展
  - 但是比较慢，封装后不够灵活
- 定义模型类 自定义层，支持更多自定义操作



tf.GradientTape() 代表这个上下文中的梯度需要被记录。

tf.keras是Tensorflow对Keras API规范实现的子类，能够让tf更易于使用，且不失灵活性和性能。

- 序列模型：
  -  tf.keras.Sequential():  Keras定义序列模型，利用model.add追加层。
  - 难以共享层和定义分支
  - 难以实现多输入多输出

```
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=4))
model.summary()
```

- functional API:
  - 可以解决序列模型的不足
  - 调用Keras中相关接口定义输入层和全连接层
  - 利用tf.keras.Model将定义的层组织起来

```
inputs = tf.keras.Input(shape=(4,), name='data')
outputs = tf.keras.layers.Dense(units=1, input_dim=4)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')
```

- 自定义模型类
  - 封装更多更复杂，便于复用
  - 继承tf.keras.Model类， 实现call()函数
  - 也可以自定义层，重写build,call两个函数

```python
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            input_dim=4,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output
```







训练：eager execution 

利用Keras高阶API进行训练：

- tf.keras.Model.compile  配置模型训练需要的参数，优化器、损失函数、评估指标
- tf.keras.Model.fit  训练模型

自定义循环：

- 使用低阶方法定义模型，必须要用这种循环，不能用高阶API
- 显示定义模型训练内部的循环，内部工作流程，可以引入灵活的自定义操作。





模型保存：savedmodel















