# 【Pytorch基础】Torchvision中transform的脚本化

Transforms从torch1.7开始新增了该特性，之前transform进行数据增强的方式是如下的，i.e. 使用compose的方式：

```python
default_configure = T.Compose([
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((32, 32)),  
            T.RandomRotation(15)
        ])
```

现在Transforms支持以下方式：

```python
import torch
import torchvision.transforms as T

# to fix random seed, use torch.manual_seed
# instead of random.seed
torch.manual_seed(12)

transforms = torch.nn.Sequential(
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.3),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
)
scripted_transforms = torch.jit.script(transforms)

tensor_image = torch.randint(0, 256, size=(3, 256, 256), dtype=torch.uint8)
# works directly on Tensors
out_image1 = transforms(tensor_image)
# on the GPU
out_image1_cuda = transforms(tensor_image.cuda())
# with batches
batched_image = torch.randint(0, 256, size=(4, 3, 256, 256), dtype=torch.uint8)
out_image_batched = transforms(batched_image)
# and has torchscript support
out_image2 = scripted_transforms(tensor_image)

```

Compose和脚本化的合作也是可行的：

```python
Note: we can similarly use T.Compose to define transforms
transforms = T.Compose([...]) and 
scripted_transforms = torch.jit.script(torch.nn.Sequential(*transforms.transforms))
```

以上方法有几点特征：

- 数据增强可以支持GPU加速
- batch化 transformation，视频处理中使用更方便。
- 可以支持多channel的tensor增强，而不仅仅是3通道或者4通道的tensor。







