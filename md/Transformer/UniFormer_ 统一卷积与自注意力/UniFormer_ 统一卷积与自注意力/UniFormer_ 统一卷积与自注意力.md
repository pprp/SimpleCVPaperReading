# UniFormer: 统一卷积与自注意力

【GiantPandaCV导语】 ICLR2022的Uniformer提出一种无缝集成卷积与自注意力的方法，仅仅在ImageNet-1k上就可以达到86.3的top1精度，文末附代码实现。



## 引言

视觉问题中想要学到有鉴别性的表征需要解决两个难点：

- Local Redundancy: 由于局部数据具有相似性，会带来局部冗余性
- Global Dependency: 为了解决长距离依赖问题，需要将不同区域中目标的关系进行建模。

因此，本文提出Uniformer来统一CNN和Transformer架构，同时解决以上两个问题。

## 方法 

![](https://img-blog.csdnimg.cn/e9b9443e89a14d1ca337402bc6468f6c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)

整体架构上follow传统CNN的架构，划分为4个stage，每个stage空间分辨率减半。

Uniformer中block由三部分构成：

- DPE: 动态位置编码
- MHRA: 多头关系聚合
- FFN: 前馈神经网络

## 1. Dynamic Position Embedding(DPE)

具体来说，使用的是一个Depth Wise Convolution来完成局部建模能力，能够让模型隐式编码位置信息，这样不仅可以让Transformer灵活处理不同输入分辨率还能提升识别性能。

$$
\operatorname{DPE}\left(\mathbf{X}_{i n}\right)=\operatorname{DWConv}\left(\mathbf{X}_{i n}\right)
$$


## 2. Multi-Head Relation Attention(MHRA)

本文将关系聚合器设计为多个头，每个头部单独处理一组channel的信息，具体来说：

- 每组卷积先生成上下文token V(类似Transformer中的MLP)
- 然后再token affinity A作用下对上下文进行聚合(CBlock与SABlock两种方式)

$$
\begin{aligned}
\mathrm{R}_{n}(\mathrm{X}) &=\mathrm{A}_{n} \mathrm{~V}_{n}(\mathrm{X}) \\
\operatorname{MHRA}(\mathrm{X}) &=\operatorname{Concat}\left(\mathrm{R}_{1}(\mathrm{X}) ; \mathrm{R}_{2}(\mathrm{X}) ; \cdots ; \mathrm{R}_{N}(\mathrm{X})\right) \mathrm{U}
\end{aligned}
$$


相关性R计算方式如上式所示，最终MHRA将n个头concate到一起，并使用U来集成N个头部。

**局部性获取：**（对应代码CBlock)

$$
\mathrm{A}_{n}^{\text {local }}\left(\mathbf{X}_{i}, \mathbf{X}_{j}\right)=a_{n}^{i-j}, \text { where } j \in \Omega_{i}^{t \times h \times w},
$$


限定一定区域，进行关系的建模。

**全局性获取：**(对应代码SABlock)

$$
\mathrm{A}_{n}^{\text {global }}\left(\mathbf{X}_{i}, \mathbf{X}_{j}\right)=\frac{e^{Q_{n}\left(\mathbf{X}_{i}\right)^{T} K_{n}\left(\mathbf{X}_{j}\right)}}{\sum_{j^{\prime} \in \Omega_{T \times H \times W}} e^{Q_{n}\left(\mathbf{X}_{i}\right)^{T} K_{n}\left(\mathbf{X}_{j^{\prime}}\right)}}
$$


从全局视角设计token相关性矩阵，即self attention.



## 3. Feed Forward Network

FFN也具有全局的建模能力，Follow了Transformer传统的结构。





## 实验

下图展示了各个任务中Uniformer的设计，大致上符合浅层使用局部性建模，深层使用全局性建模。

![](https://img-blog.csdnimg.cn/bb39bb333a924c258e8cad65dfabc7b1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)



图像分类任务：

![](https://img-blog.csdnimg.cn/72a35bc52e8444ad9b81569155ff4b26.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)



目标检测任务：

![](https://img-blog.csdnimg.cn/34deb41719fb4b91894eea77bd87a34c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)





消融实验：

![](https://img-blog.csdnimg.cn/5de6c24aa3c948ef9b7c11035f616706.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAKnBwcnAq,size_20,color_FFFFFF,t_70,g_se,x_16)



## 实现

```Python
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

```


```Python
class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x        
```


```Python
class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```


通过CBlock和SABlock构建的Uniformer的4个block如下：

```Python
  self.blocks1 = nn.ModuleList([
        CBlock(
            dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        for i in range(depth[0])])
    self.blocks2 = nn.ModuleList([
        CBlock(
            dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
        for i in range(depth[1])])
    self.blocks3 = nn.ModuleList([
        SABlock(
            dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
        for i in range(depth[2])])
    self.blocks4 = nn.ModuleList([
        SABlock(
            dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
    for i in range(depth[3])])
```



