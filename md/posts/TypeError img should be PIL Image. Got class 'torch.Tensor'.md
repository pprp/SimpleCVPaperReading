---
title: "TypeError img should be PIL Image. Got <class 'torch.Tensor'>"
date: 2019-09-09 14:54:46
tags: 
- pytorch
- error
categories:
- 深度学习
---




在pytorch中使用MNIST数据集，进行可视化，代码如下：

```
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# part 1: 数据集的获取，torch中提供了数据集的相关API
mnist_train_dataset = datasets.MNIST(root="./data/",
                                      train=True,
                                      download=True,
                                      transform=
                                        transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5]),transforms.Resize((28,28))])
                                    )
                                        
mnist_test_dataset = datasets.MNIST(root="./data/",
                                      train=False,
                                      download=True,
                                      transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28,28)))
                  )

# part 2: 数据装载， dataloader
data_loader_train = torch.utils.data.DataLoader(
    dataset=mnist_train_dataset,
    batch_size=128,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=mnist_test_dataset,
    batch_size = 1,
    shuffle=True
)


# part 3: 数据可视化，检查数据
images,labels = next(iter(data_loader_train))
# TypeError: img should be PIL Image. Got <class 'torch.Tensor'>
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std=mean=[0.5,0.5,0.5]
img = img * std + mean
# 直接imshow会报错：Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
# 意思是需要归一化处理
print([int(labels[i].numpy()) for i,label in enumerate(labels)])
plt.imshow(img)
plt.show()
```

运行会出现以下报错：

```
Traceback (most recent call last):
  File "d:/GitHub/studyNote/pytorch基础/mnist.torch.py", line 45, in <module>
    images,labels = next(iter(data_loader_train))
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torch\utils\data\dataloader.py", line 560, in __next__
    batch = self.collate_fn([self.dataset[i] for i in indices])
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torch\utils\data\dataloader.py", line 560, in <listcomp>
    batch = self.collate_fn([self.dataset[i] for i in indices])
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\datasets\mnist.py", line 95, in __getitem__
    img = self.transform(img)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\transforms.py", line 61, in __call__
    img = t(img)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\transforms.py", line 196, in __call__
    return F.resize(img, self.size, self.interpolation)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\functional.py", line 229, in resize
    raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
TypeError: img should be PIL Image. Got <class 'torch.Tensor'>
```

## 思考：

他需要PIL格式的图片，正好transforms中有一个方法为：`transforms.ToPILImage()`, 然后就变成了

```python
transform=transforms.Compose([
    						transforms.ToTensor(),                               
                              transforms.Normalize(mean=[0.5],std=[0.5
                              transforms.Resize([28,28]),
                              transforms.ToPILImage()
                             ])
```

但是还是报错：

```
Traceback (most recent call last):
  File "d:/GitHub/studyNote/pytorch基础/mnist.torch.py", line 45, in <module>
    images,labels = next(iter(data_loader_train))
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torch\utils\data\dataloader.py", line 560, in __next__
    batch = self.collate_fn([self.dataset[i] for i in indices])
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torch\utils\data\dataloader.py", line 560, in <listcomp>
    batch = self.collate_fn([self.dataset[i] for i in indices])
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\datasets\mnist.py", line 95, in __getitem__
    img = self.transform(img)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\transforms.py", line 61, in __call__
    img = t(img)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\transforms.py", line 196, in __call__
    return F.resize(img, self.size, self.interpolation)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\functional.py", line 229, in resize
    raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
TypeError: img should be PIL Image. Got <class 'torch.Tensor'>
```

去bing上查询到stackoverflow上发现一个类似的错误：

> ```python
> train_transforms = transforms.Compose(
> [transforms.Resize(255), 
> transforms.CenterCrop(224), 
> transforms.ToTensor(), 
> transforms.RandomHorizontalFlip(), 
> transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
> ```
>
> TypeError: img should be PIL Image. Got <class 'torch.Tensor'>

-- from <https://stackoverflow.com/questions/57079219/img-should-be-pil-image-got-class-torch-tensor>

下边大神的解决方案是：

> `transforms.RandomHorizontalFlip()` works on `PIL.Images`, not `torch.Tensor`. In your code above, you are applying `transforms.ToTensor()` prior to `transforms.RandomHorizontalFlip()`, which results in tensor.
>
> `transforms.RandomHorizontalFlip()` works on `PIL.Images`, not `torch.Tensor`. In your code above, you are applying `transforms.ToTensor()` prior to `transforms.RandomHorizontalFlip()`, which results in tensor.
>
> But, as per the official pytorch documentation [here](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomHorizontalFlip),
>
> > transforms.RandomHorizontalFlip() horizontally flip the given PIL Image randomly with a given probability.
>
> So, just change the order of your transformation in above code, like below:
>
> ```
> train_transforms = transforms.Compose([transforms.Resize(255), 
>                                        transforms.CenterCrop(224),  
>                                        transforms.RandomHorizontalFlip(),
>                                        transforms.ToTensor(), 
>                                        transforms.Normalize([0.485, 0.456, 0.406], 										[0.229, 0.224, 0.225])])
> ```

发现是顺序问题，需要调换，将ToTensor放在RandomHorizontalFlip之后。


## 解决：


这个问题我们也采用相同方法尝试。

从

```python
transform=transforms.Compose([
    						transforms.ToTensor(),                               
                              transforms.Normalize(mean=[0.5],std=[0.5
                              transforms.Resize([28,28]),
                              transforms.ToPILImage()
                             ])
```

改为：

```python
transform=transforms.Compose([
    						transforms.Resize([28,28]),
    						transforms.ToTensor(),                               
                              transforms.Normalize(mean=[0.5],std=[0.5])
                              # transforms.ToPILImage()
                             ])
```



对这个顺序比较好奇,又尝试了一下：

```python
transform=transforms.Compose([
                              transforms.Scale([28,28]),
                              transforms.Normalize(mean=[0.5],std=[0.5])，
                              transforms.ToTensor()
                             ])
```

发现报错：

```
  File "d:/GitHub/studyNote/pytorch基础/mnist.torch.py", line 47, in <module>
    images,labels = next(iter(data_loader_train))
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torch\utils\data\dataloader.py", line 560, in __next__
    batch = self.collate_fn([self.dataset[i] for i in indices])
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torch\utils\data\dataloader.py", line 560, in <listcomp>
    batch = self.collate_fn([self.dataset[i] for i in indices])
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\datasets\mnist.py", line 95, in __getitem__
    img = self.transform(img)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\transforms.py", line 61, in __call__
    img = t(img)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\transforms.py", line 164, in __call__
    return F.normalize(tensor, self.mean, self.std, self.inplace)
  File "E:\ProgramData\Miniconda3\envs\pytorch\lib\site-packages\torchvision\transforms\functional.py", line 201, in normalize
    raise TypeError('tensor is not a torch image.')
TypeError: tensor is not a torch image.
```

看来ToTensor需要在Normalize之前才行。

大家如果有新的发现可以在评论补充 