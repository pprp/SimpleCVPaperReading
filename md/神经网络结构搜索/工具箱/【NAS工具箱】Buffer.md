# 【NAS工具箱】Pytorch中的Buffer&Parameter

**Parameter** ： 模型中的一种可以被反向传播更新的参数。

第一种：

- 直接通过成员变量nn.Parameter()进行创建，会自动注册到parameter中。

```python
def __init__(self):
    super(MyModel, self).__init__()
    self.param = nn.Parameter(torch.randn(3, 3))  # 模型的成员变量
```

或者：

- 通过nn.Parameter() 创建普通对象
- 通过register_parameter()进行注册
- 可以通过model.parameters()返回

```python
def __init__(self):
    super(MyModel, self).__init__()
    param = nn.Parameter(torch.randn(3, 3))  # 普通 Parameter 对象
    self.register_parameter("param", param)
```



**Buffer** : 模型中不能被反向传播算法更新的参数。

- 创建tensor
- 将tensor通过register_buffer进行注册
- 可以通过model.buffers()返回

```python
def __init__(self):
    super(MyModel, self).__init__()
    buffer = torch.randn(2, 3)  # tensor
    self.register_buffer('my_buffer', buffer)
    self.param = nn.Parameter(torch.randn(3, 3))  # 模型的成员变量
```



总结：

- 模型参数=parameter+buffer; optimizer只能更新parameter，不能更新buffer，buffer只能通过forward进行更新。
- 模型保存的参数 model.state_dict() 返回一个OrderDict