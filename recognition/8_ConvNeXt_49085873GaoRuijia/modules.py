from typing import List
import torch
import torch.nn as nn

# === utility layer: stochastic depth ===========================================================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

# === utility layer: LayerNorm for NCHW ===========================================================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: [N, C, H, W]; normalize across channel dimension
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight[:, None, None] + self.bias[:, None, None]
        

# === ConvNeXt Basic Block ===========================================================================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim))) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma[:, None, None]
        x = self.drop_path(x)
        return x + shortcut

# === ConvNeXt Backbone ===========================================================================================================

class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 2, 
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.2,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        #stem：4×4, stride=4 convolution for early downsampling
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        #downsampling layers: LN + 2×2 stride=2 convolution
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        #stages: ConvNeXt blocks
        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        #head: global average pooling + LN + dropout + linear layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  #channels_last
        self.dropout = nn.Dropout(p=dropout_rate) 
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        #initialize weights
        self.apply(self._init_weights)
        #scale classifier head
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        #extract features through ConvNeXt stages
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # GlobalAvgPool
        x = x.mean([-2, -1])  #NCHW -> mean over H and W
        x = self.norm(x) #final normalization
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x) 
        x = self.head(x) 
        return x



