# CeiT：训练更快的多层特征抽取ViT

【GiantPandaCV导语】来自商汤和南洋理工的工作，也是使用卷积来增强模型提出low-level特征的能力，增强模型获取局部性的能力，核心贡献是LCA模块，可以用于捕获多层特征表示。



## 引言

针对先前Transformer架构需要大量额外数据或者额外的监督(Deit)，才能获得与卷积神经网络结构相当的性能，为了克服这种缺陷，提出结合CNN来弥补Transformer的缺陷，提出了CeiT:

（1）设计Image-to-Tokens模块来从low-level特征中得到embedding。

（2）将Transformer中的Feed Forward模块替换为Locally-enhanced Feed-Forward(LeFF)模块，增加了相邻token之间的相关性。

（3）使用Layer-wise Class Token Attention（LCA）捕获多层的特征表示。

经过以上修改，可以发现模型效率方面以及泛化能力得到了提升，收敛性也有所改善，如下图所示：

![](https://img-blog.csdnimg.cn/9780a125da4047b98f28b4b6370f8550.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_8,color_FFFFFF,t_70,g_se,x_16)

## 方法



### 1. Image-to-Tokens

![](https://img-blog.csdnimg.cn/088299a0e1164687a5fecd687a6ca335.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)

使用卷积+池化来取代原先ViT中7x7的大型patch。

$$
\mathbf{x}^{\prime}=\mathrm{I} 2 \mathrm{~T}(\mathbf{x})=\operatorname{MaxPool}(\operatorname{BN}(\operatorname{Conv}(\mathbf{x})))
$$


### 2. LeFF

![](https://img-blog.csdnimg.cn/3e3e6cc8be1742a1a76864d37f0aa973.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_8,color_FFFFFF,t_70,g_se,x_16)

将tokens重新拼成feature map，然后使用深度可分离卷积添加局部性的处理，然后再使用一个Linear层映射至tokens。

$$
\begin{aligned}
\mathbf{x}_{c}^{h}, \mathbf{x}_{p}^{h} &=\operatorname{Split}\left(\mathbf{x}_{t}^{h}\right) \\
\mathbf{x}_{p}^{l_{1}} &=\operatorname{GELU}\left(\operatorname{BN}\left(\operatorname{Linear}\left(\left(\mathbf{x}_{p}^{h}\right)\right)\right)\right.\\
\mathbf{x}_{p}^{s} &=\operatorname{SpatialRestore}\left(\mathbf{x}_{p}^{l_{1}}\right) \\
\mathbf{x}_{p}^{d} &=\operatorname{GELU}\left(\operatorname{BN}\left(\operatorname{DWConv}\left(\mathbf{x}_{p}^{s}\right)\right)\right) \\
\mathbf{x}_{p}^{f} &=\operatorname{Flatten}\left(\mathbf{x}_{p}^{d}\right) \\
\mathbf{x}_{p}^{l_{2}} &=\operatorname{GELU}\left(\operatorname{BN}\left(\operatorname{Linear} 2\left(\mathbf{x}_{p}^{f}\right)\right)\right) \\
\mathbf{x}_{t}^{h+1} &=\operatorname{Concat}\left(\mathbf{x}_{c}^{h}, \mathbf{x}_{p}^{l_{2}}\right)
\end{aligned}
$$


### 3. LCA

前两个都比较常规，最后一个比较有特色，经过所有Transformer层以后使用的Layer-wise Class-token Attention，如下图所示：

![](https://img-blog.csdnimg.cn/cc00ba533fe746edaa2064a1f306482e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_10,color_FFFFFF,t_70,g_se,x_16)

LCA模块会将所有Transformer Block中得到的class token作为输入，然后再在其基础上使用一个MSA+FFN得到最终的logits输出。作者认为这样可以获取多尺度的表征。



## 实验

SOTA比较：

![](https://img-blog.csdnimg.cn/c1e75dde51044e2f9db47f7a1b2d967c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_2,color_FFFFFF,t_70,g_se,x_16)

I2T消融实验：

![](https://img-blog.csdnimg.cn/9b570241dcc84ab688cc36088a07e415.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_9,color_FFFFFF,t_70,g_se,x_16)

LeFF消融实验：

![](https://img-blog.csdnimg.cn/d4f2f16ff1374e33a341d2fdf86d20d0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_9,color_FFFFFF,t_70,g_se,x_16)

LCA有效性比较：

![](https://img-blog.csdnimg.cn/74c2fade4ed74d27a8081279fc7d4357.png)

收敛速度比较：

![](https://img-blog.csdnimg.cn/b7716eaa1a5a43338a424d8abd441136.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_2,color_FFFFFF,t_70,g_se,x_16)



## 代码

**模块1：I2T** Image-to-Token

```Python
  # IoT
  self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, conv_kernel, stride, 4),
      nn.BatchNorm2d(out_channels),
      nn.MaxPool2d(pool_kernel, stride)    
  )
  
  feature_size = image_size // 4

  assert feature_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
  num_patches = (feature_size // patch_size) ** 2
  patch_dim = out_channels * patch_size ** 2
  self.to_patch_embedding = nn.Sequential(
      Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
      nn.Linear(patch_dim, dim),
  )
```


**模块2：LeFF**

```Python
class LeFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        
    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
        
class TransformerLeFF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, scale = 4, depth_kernel = 3, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, LeFF(dim, scale, depth_kernel)))
            ]))
    def forward(self, x):
        c = list()
        for attn, leff in self.layers:
            x = attn(x)
            cls_tokens = x[:, 0]
            c.append(cls_tokens)
            x = leff(x[:, 1:])
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1) 
        return x, torch.stack(c).transpose(0, 1)
```


**模块3：LCA**

```Python
class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class LCA(nn.Module):
    # I remove Residual connection from here, in paper author didn't explicitly mentioned to use Residual connection, 
    # so I removed it, althougth with Residual connection also this code will work.
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.ModuleList([
                PreNorm(dim, LCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x[:, -1].unsqueeze(1)
            x = x[:, -1].unsqueeze(1) + ff(x)
        return x
```




## 参考

[https://arxiv.org/abs/2103.11816](https://arxiv.org/abs/2103.11816)

[https://github.com/rishikksh20/CeiT-pytorch/blob/master/ceit.py](https://github.com/rishikksh20/CeiT-pytorch/blob/master/ceit.py)



