# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return (p, semseg)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# class FASA(nn.Module):
#     """
#     FASA — Feature-Adaptive Statistical Attention.

#     Statistics-driven attention. Channel attention is an ECA-style 1-D
#     interaction over a learned convex mix of {mean, std, CV}; spatial
#     attention is a multi-scale, directionally-factorized map over a
#     {mean, max, var} channel-pooled descriptor; the two are combined by a
#     per-channel gate predicted from global feature statistics.

#     Novel term: CV = σ / (|μ| + σ + ε) ∈ [0, 1)  — a bounded, scale-invariant
#     dispersion descriptor for the large object-scale variation of aerial data.

#     Numerical guards (eps, fp32 stats, dtype casts) are implementation details,
#     not part of the formulation.
#     """

#     def __init__(self, c1, c2=None, *args, **kwargs):
#         super().__init__()
#         C = c1
#         k = int(abs((math.log2(C) / 2) + 0.5))
#         k = k if k % 2 else k + 1
#         self.eps = 1e-6

#         # channel: convex mix of 3 stats -> ECA 1-D conv
#         self.moment_logits = nn.Parameter(torch.zeros(3))
#         self.channel_conv  = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

#         # spatial: {mean,max,var} -> 1ch, then 3 directional scales (3,5,7)
#         self.pool_fuse = nn.Conv2d(3, 1, 1, bias=False)
#         self.ca_h = nn.ModuleList(
#             [nn.Conv2d(1, 1, (kk, 1), padding=(kk // 2, 0), bias=False) for kk in (3, 5, 7)]
#         )
#         self.ca_w = nn.ModuleList(
#             [nn.Conv2d(1, 1, (1, kk), padding=(0, kk // 2), bias=False) for kk in (3, 5, 7)]
#         )
#         self.scale_logits = nn.Parameter(torch.zeros(3))

#         # per-channel fusion gate from 2 global statistics
#         self.gate = nn.Sequential(nn.Linear(2, 16), nn.ReLU(inplace=True), nn.Linear(16, C))

#         self.alpha = nn.Parameter(torch.tensor(0.15))

#     def forward(self, x):
#         dt = x.dtype
#         B, C, H, W = x.shape
#         xf = x.float()  # stats in fp32

#         # ---- channel ----
#         mu_c  = xf.mean(dim=(2, 3))                                   # (B,C)
#         var_c = xf.var(dim=(2, 3), correction=0)                     # (B,C)
#         sd_c  = (var_c + self.eps).sqrt()
#         cv_c  = sd_c / (mu_c.abs() + sd_c + self.eps)                # (B,C), in [0,1)

#         w   = F.softmax(self.moment_logits, dim=0)
#         d_c = (w[0] * mu_c + w[1] * sd_c + w[2] * cv_c).to(dt)       # (B,C)
#         a_c = torch.sigmoid(
#             self.channel_conv(d_c.unsqueeze(1)).squeeze(1)
#         ).view(B, C, 1, 1)

#         # ---- spatial ----
#         sp_mean = xf.mean(dim=1, keepdim=True)
#         sp_max  = xf.amax(dim=1, keepdim=True)
#         sp_var  = xf.var(dim=1, keepdim=True, correction=0) + self.eps
#         p = self.pool_fuse(torch.cat([sp_mean, sp_max, sp_var], dim=1).to(dt))

#         sw  = F.softmax(self.scale_logits, dim=0)
#         a_s = sum(
#             sw[i] * (torch.sigmoid(self.ca_h[i](p)) * torch.sigmoid(self.ca_w[i](p)))
#             for i in range(3)
#         )                                                            # (B,1,H,W)

#         # ---- gate ----
#         sp_disp    = sp_var.mean(dim=(2, 3))                         # (B,1)
#         ch_disp    = var_c.mean(dim=1, keepdim=True)                 # (B,1)
#         disp_ratio = torch.log1p(sp_disp / (ch_disp + self.eps))
#         sp_var_n   = sp_var / (sp_var.mean(dim=(2, 3), keepdim=True) + self.eps)
#         peak       = sp_var_n.amax(dim=(2, 3))                       # (B,1)
#         g = torch.sigmoid(
#             self.gate(torch.cat([disp_ratio, peak], dim=1).to(dt))
#         ).view(B, C, 1, 1)

#         # ---- fuse + bounded residual ----
#         a     = a_c * (g * a_s + (1 - g))
#         alpha = self.alpha.to(dt)
#         return x * (1 + alpha * (2 * a - 1))


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FASA(nn.Module):
    r"""FASA — Feature-Adaptive Statistical Attention.

    A lightweight, statistics-driven attention block intended for tiny-object
    detection (TOD), where features carry large object-scale variation.

    Three cooperating paths:
      * Channel  : ECA-style 1-D interaction over a convex mix of standardized
                   channel statistics {mean, std, CV}.
      * Spatial  : multi-scale, directionally-factorized map over a {mean,max,var}
                   channel-pooled descriptor.
      * Gate     : per-channel blend of the two, predicted from two global
                   dispersion statistics.

    Novelty term:
        CV = σ / (|μ| + σ + ε) ∈ [0, 1)
      a bounded, scale-invariant dispersion descriptor. Because the raw moments
      live on incomparable scales, all three are z-scored across the channel axis
      *before* the convex mix, so the learned softmax weights — not raw magnitude —
      decide each term's contribution. This is what makes the CV ablation honest.

    Identity at init:
        out = x * (1 + alpha * (2a - 1)),   alpha initialized to 0
      The block is the exact identity at step 0 (ReZero/LayerScale style) and
      learns its modulation strength and sign from zero, so it can be dropped into
      a pretrained backbone without perturbing it.

    Args:
        c1 (int): input channels.
        c2 (int, optional): unused; present for Ultralytics yaml compatibility.
        use_cv (bool): include the CV term in the channel mix. Set False for the
            "minus-CV" ablation row.
    """

    def __init__(self, c1: int, c2: int = None, *args, use_cv: bool = True, **kwargs) -> None:
        super().__init__()
        C = c1
        self.eps = 1e-6
        self.use_cv = use_cv
        self.n_moments = 3 if use_cv else 2

        # --- channel: convex mix of standardized stats -> ECA 1-D conv ---
        k = int(abs((math.log2(C) / 2) + 0.5))
        k = k if k % 2 else k + 1
        self.moment_logits = nn.Parameter(torch.zeros(self.n_moments))
        self.channel_conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

        # --- spatial: {mean,max,var} -> 1ch, then 3 directional scales ---
        self.pool_fuse = nn.Conv2d(3, 1, 1, bias=False)
        self.ca_h = nn.ModuleList(
            [nn.Conv2d(1, 1, (kk, 1), padding=(kk // 2, 0), bias=False) for kk in (3, 5, 7)]
        )
        self.ca_w = nn.ModuleList(
            [nn.Conv2d(1, 1, (1, kk), padding=(0, kk // 2), bias=False) for kk in (3, 5, 7)]
        )
        self.scale_logits = nn.Parameter(torch.zeros(3))

        # --- per-channel fusion gate from 2 global dispersion statistics ---
        self.gate = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(inplace=True), nn.Linear(16, C)
        )

        # --- bounded residual scale; 0 -> exact identity at init ---
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _zscore(t: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
        """Standardize across `dim` so heterogeneous statistics become comparable."""
        m = t.mean(dim=dim, keepdim=True)
        s = t.std(dim=dim, keepdim=True)
        return (t - m) / (s + eps)

    def _channel_attention(self, xf: torch.Tensor, dt: torch.dtype):
        """ECA over a standardized convex mix of channel statistics."""
        B, C = xf.shape[:2]
        mu_c = xf.mean(dim=(2, 3))                              # (B,C)
        var_c = xf.var(dim=(2, 3), correction=0)               # (B,C)
        sd_c = (var_c + self.eps).sqrt()

        moments = [self._zscore(mu_c, 1, self.eps),
                   self._zscore(sd_c, 1, self.eps)]
        if self.use_cv:
            cv_c = sd_c / (mu_c.abs() + sd_c + self.eps)        # (B,C) in [0,1)
            moments.append(self._zscore(cv_c, 1, self.eps))

        w = F.softmax(self.moment_logits, dim=0)
        d_c = sum(w[i] * moments[i] for i in range(self.n_moments)).to(dt)  # (B,C)
        a_c = torch.sigmoid(
            self.channel_conv(d_c.unsqueeze(1)).squeeze(1)
        ).view(B, C, 1, 1)
        return a_c, var_c

    def _spatial_attention(self, xf: torch.Tensor, dt: torch.dtype):
        """Multi-scale directionally-factorized spatial map."""
        sp_mean = xf.mean(dim=1, keepdim=True)
        sp_max = xf.amax(dim=1, keepdim=True)
        sp_var = xf.var(dim=1, keepdim=True, correction=0) + self.eps
        p = self.pool_fuse(torch.cat([sp_mean, sp_max, sp_var], dim=1).to(dt))

        sw = F.softmax(self.scale_logits, dim=0)
        a_s = sum(
            sw[i] * (torch.sigmoid(self.ca_h[i](p)) * torch.sigmoid(self.ca_w[i](p)))
            for i in range(3)
        )                                                       # (B,1,H,W)
        return a_s, sp_var

    def _fusion_gate(self, var_c: torch.Tensor, sp_var: torch.Tensor, dt: torch.dtype):
        """Per-channel blend weight from two global dispersion statistics."""
        B, C = var_c.shape
        sp_disp = sp_var.mean(dim=(2, 3))                       # (B,1)
        ch_disp = var_c.mean(dim=1, keepdim=True)               # (B,1)
        disp_ratio = torch.log1p(sp_disp / (ch_disp + self.eps))
        sp_var_n = sp_var / (sp_var.mean(dim=(2, 3), keepdim=True) + self.eps)
        peak = sp_var_n.amax(dim=(2, 3))                        # (B,1)
        g = torch.sigmoid(
            self.gate(torch.cat([disp_ratio, peak], dim=1).to(dt))
        ).view(B, C, 1, 1)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()  # all statistics computed in fp32 for numerical stability

        a_c, var_c = self._channel_attention(xf, dt)
        a_s, sp_var = self._spatial_attention(xf, dt)
        g = self._fusion_gate(var_c, sp_var, dt)

        # fuse: g blends spatial map with a unit (pass-through) prior, per channel
        a = a_c * (g * a_s + (1.0 - g))                         # (B,C,H,W), in (0,1)

        # bounded, identity-at-init residual
        alpha = self.alpha.to(dt)
        return x * (1.0 + alpha * (2.0 * a - 1.0))

class CoordinationAttention(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super().__init__()
        assert c1 == c2, f"CoordinationAttention requires c1==c2, got {c1} vs {c2}"

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Orthonormal Haar DWT / IDWT as fixed grouped convs.
# Orthonormal => synthesis is the exact transpose of analysis, so the
# DWT->IDWT round trip is lossless (reconstruction is exact). Implemented
# as conv2d / conv_transpose2d (stride 2) so it stays ONNX/TensorRT
# exportable -- important for YOLO26's edge story. Swap in
# `pytorch_wavelets` (DWTForward/DWTInverse) if you prefer a vetted lib.
# ----------------------------------------------------------------------
def _haar2d():
    s = 0.5  # (1/sqrt2)^2, orthonormal separable Haar
    ll = torch.tensor([[ s,  s], [ s,  s]])
    lh = torch.tensor([[ s,  s], [-s, -s]])
    hl = torch.tensor([[ s, -s], [ s, -s]])
    hh = torch.tensor([[ s, -s], [-s,  s]])
    return torch.stack([ll, lh, hl, hh], 0)  # (4, 2, 2)


class HaarDWT(nn.Module):
    def __init__(self, ch):
        super().__init__()
        w = _haar2d().unsqueeze(1).repeat(ch, 1, 1, 1)  # (4*ch, 1, 2, 2)
        self.register_buffer("w", w)
        self.ch = ch

    def forward(self, x):
        y = F.conv2d(x, self.w.to(x.dtype), stride=2, groups=self.ch)
        B, _, H, W = y.shape
        y = y.view(B, self.ch, 4, H, W)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]  # LL, LH, HL, HH


class HaarIDWT(nn.Module):
    def __init__(self, ch):
        super().__init__()
        w = _haar2d().unsqueeze(1).repeat(ch, 1, 1, 1)  # (4*ch, 1, 2, 2)
        self.register_buffer("w", w)
        self.ch = ch

    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        y = torch.stack([ll, lh, hl, hh], 2).view(B, C * 4, H, W)
        return F.conv_transpose2d(y, self.w.to(y.dtype), stride=2, groups=self.ch)

class LKA(nn.Module):
    """Decomposed large-kernel attention. Channel-preserving."""
    def __init__(self, c1, c2=None, k=5, dk=7, d=3):
        super().__init__()
        ch = c1
        self.dw = nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch)
        self.dwd = nn.Conv2d(ch, ch, dk, padding=(dk // 2) * d, groups=ch, dilation=d)
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x * self.pw(self.dwd(self.dw(x)))


class WGCA(nn.Module):
    """Wavelet-Gated Context Attention. Channel-preserving.
    YAML args after the channel drive ablations:
      [c]               -> context=True,  gate=True
      [c, False, False] -> lossless wavelet only
      [c, True, False]  -> context, no gating
    """
    def __init__(self, c1, c2=None, context=True, gate=True):
        super().__init__()
        ch = c1
        self.use_gate = gate
        self.dwt = HaarDWT(ch)
        self.idwt = HaarIDWT(ch)
        self.ctx = LKA(ch) if context else nn.Identity()
        self.gate_conv = nn.Conv2d(ch, 3 * ch, 1) if gate else None
        self.proj = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        ctx = self.ctx(ll)
        if self.use_gate:
            gl, gh, ghh = torch.sigmoid(self.gate_conv(ctx)).chunk(3, 1)
            lh, hl, hh = lh * gl, hl * gh, hh * ghh
        return x + self.gamma * self.proj(self.idwt(ctx, lh, hl, hh))
    
class WCA(nn.Module):
    """Wavelet Context Attention (gate-free). Lossless Haar split -> LKA context
    on the half-res LL band -> exact IDWT -> LayerScale residual. Detail subbands
    pass through unmodulated (gating ablated out after gate-inertness analysis)."""
    def __init__(self, c1, c2=None):
        super().__init__()
        ch = c1
        self.dwt = HaarDWT(ch)
        self.idwt = HaarIDWT(ch)
        self.ctx = LKA(ch)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        ctx = self.ctx(ll)
        return x + self.gamma * self.proj(self.idwt(ctx, lh, hl, hh))


class ECA(nn.Module):
    """Efficient channel attention (no FC, ~0 params)."""
    def __init__(self, ch, k=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)                       # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class MDC(nn.Module):
    """Multi-scale Dilated Context — receptive-field expansion at P4.
    Parallel depthwise dilated branches aggregate multi-scale context; ECA
    reweights channels; LayerScale residual (gamma init 0, matches WCA / stable
    on the end2end one-to-one head). Channel-preserving."""
    def __init__(self, c1, c2=None, dilations=(1, 3, 5), k=3):
        super().__init__()
        ch = c1
        self.branches = nn.ModuleList(
            nn.Conv2d(ch, ch, k, padding=(k // 2) * d, dilation=d, groups=ch)
            for d in dilations
        )
        self.fuse = nn.Conv2d(ch * len(dilations), ch, 1)
        self.eca = ECA(ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = torch.cat([b(x) for b in self.branches], 1)
        y = self.eca(self.fuse(y))
        return x + self.gamma * y

class SAKA(nn.Module):
    """Scale-Adaptive Kernel Attention. LKA uses one fixed dilated depthwise
    conv for its large receptive field; SAKA replaces it with several dilated
    branches whose per-location mixing weights are predicted from local content,
    so the effective receptive field adapts to object scale (targets VisDrone's
    extreme scale variance). Output form matches LKA (x * attn) for a clean swap.
      adaptive=True  -> content-routed branch weights (full module)
      adaptive=False -> fixed equal weights (multi-dilation control)
    """
    def __init__(self, c1, c2=None, adaptive=False, k=5, dilations=(1, 3, 5)):
        super().__init__()
        ch = c1
        self.adaptive = adaptive
        self.n = len(dilations)
        self.local = nn.Conv2d(ch, ch, k, padding=k // 2, groups=ch)
        self.branches = nn.ModuleList(
            nn.Conv2d(ch, ch, k, padding=(k // 2) * d, dilation=d, groups=ch)
            for d in dilations
        )
        self.router = nn.Conv2d(ch, self.n, 1) if adaptive else None
        self.pw = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        loc = self.local(x)
        outs = [b(loc) for b in self.branches]              # n x (B,C,H,W)
        if self.adaptive:
            w = torch.softmax(self.router(loc), dim=1)      # (B,n,H,W)
            ctx = sum(outs[i] * w[:, i:i + 1] for i in range(self.n))
        else:
            ctx = sum(outs) / self.n
        return x * self.pw(ctx)

import torch
import torch.nn as nn
import torch.nn.functional as F

class HFLKA(nn.Module):
    """High-Frequency Large Kernel Attention.
    Designed specifically for Tiny Object Detection (VisDrone).
    1. Dense Asymmetric Convolutions: Fixes the 'grid artifact' of dilated convs.
    2. High-Frequency Gate: Suppresses massive background noise.
    """
    def __init__(self, c1, c2, k=7):
        super().__init__()
        self.ch = c1
        
        # 1. Local dense feature extraction (matches LKA's first stage)
        self.local = nn.Conv2d(self.ch, self.ch, 5, padding=2, groups=self.ch)
        
        # 2. Dense Asymmetric Large Kernel (Cross-shaped, no holes)
        self.dw_h = nn.Conv2d(self.ch, self.ch, (1, k), padding=(0, k // 2), groups=self.ch)
        self.dw_v = nn.Conv2d(self.ch, self.ch, (k, 1), padding=(k // 2, 0), groups=self.ch)
        
        # 3. High-Frequency Gate (Parameter-free Laplacian)
        # Extracts edges/specks (tiny objects) and suppresses flat backgrounds
        hp = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)
        self.register_buffer('hp', hp.repeat(self.ch, 1, 1, 1))
        
        # 4. Projection
        self.pw = nn.Conv2d(self.ch, self.ch, 1)

    def forward(self, x):
        # Extract high-frequency mask 
        # (Tiny objects = high response, Background = low response)
        hf = F.conv2d(x, self.hp.to(x.dtype), padding=1, groups=self.ch)
        
        # Soft attention mask
        gate = torch.sigmoid(torch.abs(hf)) 
        
        # Dense cross-shaped large kernel context
        ctx = self.dw_v(self.dw_h(self.local(x)))
        
        # Gate the context: suppress background, amplify tiny objects
        ctx = ctx * gate
        
        # Output matches LKA format (x * attn) for a clean swap
        return x * self.pw(ctx)

class HRGA(nn.Module):
    """High-Resolution Gated Attention. Inputs [F3 (stride 8, main),
    H2 (stride 4, detail source)]. F3 generates a semantic gate (WHERE objects
    are); H2 carries the DETAIL; gate selects relevant detail; pixel-unshuffle
    brings it losslessly to stride 8; LayerScale residual into F3.
      gate=True  -> semantic gating (full module)
      gate=False -> naive injection (control: is the gate doing anything?)
    """
    def __init__(self, c1, c2, gate=True):
        super().__init__()                 # c1 = F3 ch (output), c2 = H2 ch (source)
        self.gate = gate
        self.proj = nn.Conv2d(c2, c1, 1)
        self.gate_conv = nn.Conv2d(c1, 1, 1) if gate else None
        self.fuse = nn.Conv2d(c1 * 4, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f3, h2 = x                                       # (B,C,H,W), (B,c2,2H,2W)
        d = self.proj(h2)                                # (B,C,2H,2W)
        if self.gate:
            g = torch.sigmoid(self.gate_conv(f3))        # (B,1,H,W)
            g = F.interpolate(g, scale_factor=2, mode="nearest")
            d = d * g                                    # keep detail near objects
        d = F.pixel_unshuffle(d, 2)                      # (B,4C,H,W) lossless
        return f3 + self.gamma * self.fuse(d)            # (B,C,H,W)

class LKA_HFGate(nn.Module):
    def __init__(self, c1, c2=None, k=5, dk=7, d=3):
        super().__init__(); ch = c1
        self.dw  = nn.Conv2d(ch, ch, k, padding=k//2, groups=ch)
        self.dwd = nn.Conv2d(ch, ch, dk, padding=(dk//2)*d, groups=ch, dilation=d)
        self.pw  = nn.Conv2d(ch, ch, 1)
        hp = torch.tensor([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]], dtype=torch.float32)
        self.register_buffer("hp", hp.repeat(ch, 1, 1, 1))
    def forward(self, x):
        gate = torch.sigmoid(torch.abs(F.conv2d(x, self.hp.to(x.dtype), padding=1, groups=x.shape[1])))
        ctx = self.pw(self.dwd(self.dw(x)) * gate)   # LKA context, HF-gated
        return x * ctx

import torch
import torch.nn as nn

class RepLKA(nn.Module):
    """
    Train-time multi-branch Large Kernel Attention (LKA).
    Merges to plain LKA (k=5 DW + 7x7 d=3 DW + 1x1) at inference.
    """
    def __init__(self, c1, c2, k=5, dk=7, d=3, deploy=False):
        super().__init__()
        
        # In attention blocks, input channels (c1) usually equal output channels (c2).
        # We use c2 for all convolutions to ensure consistency.
        self.ch, self.k, self.dk, self.d, self.deploy = c2, k, dk, d, deploy
        
        if deploy:
            # Inference-mode: Single unified convolution branches
            self.dw  = nn.Conv2d(c2, c2, k,  padding=k//2, groups=c2)
            self.dwd = nn.Conv2d(c2, c2, dk, padding=(dk//2)*d, groups=c2, dilation=d)
        else:
            # Training-mode: Multi-branch
            # Stage 1 (dense, d=1): 5x5 + 3x3 parallel depthwise
            self.dw_k = nn.Conv2d(c2, c2, k, padding=k//2, groups=c2)
            self.dw_s = nn.Conv2d(c2, c2, 3, padding=1,   groups=c2)
            
            # Stage 2 (dilated, d=3): 7x7 + 5x5, BOTH at dilation d
            self.dwd_k = nn.Conv2d(c2, c2, dk, padding=(dk//2)*d, groups=c2, dilation=d)
            self.dwd_s = nn.Conv2d(c2, c2, 5,  padding=(5//2)*d,  groups=c2, dilation=d)
            
        # Pointwise convolution (1x1)
        self.pw = nn.Conv2d(c2, c2, 1)

    def _stage1(self, x): 
        return self.dw(x) if self.deploy else self.dw_k(x) + self.dw_s(x)
        
    def _stage2(self, x): 
        return self.dwd(x) if self.deploy else self.dwd_k(x) + self.dwd_s(x)

    def forward(self, x):
        # Attention scaling: x * Pointwise(Dilated_DW(Dense_DW(x)))
        return x * self.pw(self._stage2(self._stage1(x)))
    
import torch
import torch.nn as nn

class CSCA(nn.Module):
    """
    Center-Surround Contrastive Attention (CSCA).
    Resolves the Activation Spread paradox in YOLO26's NMS-Free One-to-One head.
    Pins activations to the exact center of tiny objects (VisDrone) and inhibits edges/background.
    """
    def __init__(self, c1, c2=None, k_center=3, k_surround=7):
        super().__init__()
        ch = c1
        
        # 1. Center Branch (Excitatory): Captures the exact core of tiny objects
        self.center = nn.Conv2d(ch, ch, k_center, padding=k_center // 2, groups=ch)
        
        # 2. Surround Branch (Inhibitory): Captures local background/context
        self.surround = nn.Conv2d(ch, ch, k_surround, padding=k_surround // 2, groups=ch)
        
        # 3. Learnable Channel-wise Inhibition Factor (gamma)
        # Initialized to 1.0. Allows the network to learn the optimal suppression ratio.
        self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1))
        
        # 4. Channel Projection
        self.proj = nn.Conv2d(ch, ch, 1)
        
        # 5. LayerScale for stable integration with YOLO26's Progressive Loss
        # Initialized to 1e-4 (Identity at start), slowly grows during training.
        self.layer_scale = nn.Parameter(torch.ones(1, ch, 1, 1) * 1e-4)

    def forward(self, x):
        # Extract center and surround features
        f_c = self.center(x)
        f_s = self.surround(x)
        
        # Contrastive Gating: 
        # If Center > Surround (isolated tiny object) -> Sigmoid > 0.5 (Amplify)
        # If Surround >= Center (edges, clutter, large objects) -> Sigmoid <= 0.5 (Suppress)
        contrast = f_c - (self.gamma * f_s)
        attn = torch.sigmoid(contrast)
        
        # Generate attention map
        attn_map = self.proj(attn)
        
        # Apply LayerScale residual (matches YOLO26 training stability)
        return x + self.layer_scale * (x * attn_map)


import torch
import torch.nn as nn

class LCSAv2(nn.Module):
    """Local Contrast & Surround Attention (MuSGD-Stable Version).
    Mathematically bounded and initialized to identity to ensure stable 
    convergence under YOLO26's native MuSGD optimizer.
    """
    def __init__(self, c1, c2=None, kc=3, ks=9, eps=1e-3):
        super().__init__()
        ch = c1
        self.center   = nn.Conv2d(ch, ch, kc, padding=kc // 2, groups=ch)
        self.surround = nn.Conv2d(ch, ch, ks, padding=ks // 2, groups=ch)
        
        # Learnable scale (temperature) parameter. Initialized to 0.
        self.scale = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.pw    = nn.Conv2d(ch, ch, 1)
        self.eps   = eps

        # Sensible init: center = delta, surround = box blur
        with torch.no_grad():
            self.center.weight.zero_();  self.center.weight[:, :, kc // 2, kc // 2] = 1.0
            self.surround.weight.fill_(1.0 / (ks * ks))
            if self.center.bias is not None:   self.center.bias.zero_()
            if self.surround.bias is not None: self.surround.bias.zero_()
            
            # CRITICAL FIX: Zero-init projection so module starts as exact Identity
            self.pw.weight.zero_()
            if self.pw.bias is not None: self.pw.bias.zero_()

    def forward(self, x):
        c = self.center(x)
        s = self.surround(x)
        
        # FIX 1: Safe Normalized Contrast.
        # Dividing by |s| explodes on background pixels. Dividing by sum of magnitudes 
        # strictly bounds the contrast to [-1, 1], preventing gradient explosion.
        contrast = (c - s) / (torch.abs(c) + torch.abs(s) + self.eps)
        
        # FIX 2: Bounded Gating.
        # Because contrast is bounded, Sigmoid won't instantly saturate to 0 or 1.
        # Scale starts at 0, so sigmoid(0) = 0.5.
        gate = torch.sigmoid(self.scale * contrast)
        
        # FIX 3: Additive Residual Connection.
        # Multiplicative attention attenuates gradients. Additive preserves MuSGD momentum.
        # Because pw is zero-init, the module acts as pure identity at epoch 0.
        return x + self.pw(c * gate)
    
class LCSA(nn.Module):
    """Local Contrast & Surround Attention.
    Built for YOLO26's NMS-free one-to-one head on dense tiny objects.
    Instead of aggregating context (which smooths the spatial response and
    blurs adjacent objects), it sharpens each location against its local
    surround via *learnable divisive normalization*, increasing
    center-vs-neighbor contrast so the one-to-one head can separate crowded
    objects and the DFL-free head gets sharper localization features.
    Channel-preserving. Multiplicative output (no LayerScale -> no dead gamma).
    """
    def __init__(self, c1, c2=None, kc=3, ks=9, eps=1e-3):
        super().__init__()
        ch = c1
        self.center   = nn.Conv2d(ch, ch, kc, padding=kc // 2, groups=ch)  # local detail
        self.surround = nn.Conv2d(ch, ch, ks, padding=ks // 2, groups=ch)  # local context
        self.alpha    = nn.Parameter(torch.ones(1, ch, 1, 1))  # per-channel contrast gain
        self.pw       = nn.Conv2d(ch, ch, 1)
        self.eps      = eps

        # Sensible init: center = identity (delta), surround = box blur.
        # => at step 0 the module computes a normalized high-pass contrast gate
        #    (a tiny-object / edge emphasizer) and then adapts per channel.
        with torch.no_grad():
            self.center.weight.zero_();  self.center.weight[:, :, kc // 2, kc // 2] = 1.0
            self.surround.weight.fill_(1.0 / (ks * ks))
            if self.center.bias is not None:   self.center.bias.zero_()
            if self.surround.bias is not None: self.surround.bias.zero_()

    def forward(self, x):
        c = self.center(x)
        s = self.surround(x)
        contrast = (c - s) / torch.sqrt(s * s + self.eps)   # illumination-invariant local contrast
        gate = torch.sigmoid(self.alpha * contrast)          # emphasize standout locations, damp flat surround
        return x * self.pw(c * gate)                         # LKA-style multiplicative coupling
    


class SE(nn.Module):
    """Squeeze-and-Excitation. Channel attention (the canonical baseline)."""
    def __init__(self, c1, c2=None, r=16):
        super().__init__()
        ch = c1
        rd = max(ch // r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, rd, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(rd, ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)
    
class _ChannelAttn(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        rd = max(ch // r, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, rd, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rd, ch, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class _SpatialAttn(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""
    def __init__(self, c1, c2=None, r=16, k=7):
        super().__init__()
        self.ca = _ChannelAttn(c1, r)
        self.sa = _SpatialAttn(k)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    

class CoordAtt(nn.Module):
    """Coordinate Attention. Position-aware channel attention (common in aerial)."""
    def __init__(self, c1, c2=None, r=32):
        super().__init__()
        ch = c1
        rd = max(ch // r, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(ch, rd, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(rd)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(rd, ch, 1, bias=False)
        self.conv_w = nn.Conv2d(rd, ch, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        xh = self.pool_h(x)                       # b,c,h,1
        xw = self.pool_w(x).permute(0, 1, 3, 2)   # b,c,w,1
        y = torch.cat([xh, xw], dim=2)            # b,c,h+w,1
        y = self.act(self.bn1(self.conv1(y)))
        xh, xw = torch.split(y, [h, w], dim=2)
        xw = xw.permute(0, 1, 3, 2)
        ah = torch.sigmoid(self.conv_h(xh))
        aw = torch.sigmoid(self.conv_w(xw))
        return x * ah * aw

class SimAM(nn.Module):
    """SimAM — parameter-free attention. Include it to show a no-param comparator."""
    def __init__(self, c1=None, c2=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mu = x.mean([2, 3], keepdim=True)
        d = (x - mu).pow(2)
        v = d.sum([2, 3], keepdim=True) / n
        e = d / (4 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(e)
    
class EMA(nn.Module):
    """Efficient Multi-scale Attention with cross-spatial learning.
    `factor` = number of groups; channels must be divisible by it and
    (channels // factor) must be > 0. At P3 in YOLO26n channels are small,
    so use a small factor (e.g. 8 or 16), NOT the paper's default 32."""
    def __init__(self, c1, c2=None, factor=8):
        super().__init__()
        ch = c1
        self.groups = factor
        assert ch % self.groups == 0 and ch // self.groups > 0, \
            f"channels {ch} not compatible with factor {factor}"
        g = ch // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(g, g)
        self.conv1x1 = nn.Conv2d(g, g, 1)
        self.conv3x3 = nn.Conv2d(g, g, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, -1, h, w)          # b*g, c//g, h, w
        xh = self.pool_h(gx)                                # ., ., h, 1
        xw = self.pool_w(gx).permute(0, 1, 3, 2)            # ., ., w, 1
        hw = self.conv1x1(torch.cat([xh, xw], dim=2))
        xh, xw = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * xh.sigmoid() * xw.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)


# ultralytics/nn/modules/block.py  (append near C2PSA)
# =====================================================================
# MS-DPRA : Multi-Scale Deformable Pyramid Routing Attention
# Co-designed with MuSGD: every expressive weight is a 2-D Conv/Linear
# so Newton-Schulz orthogonalization fires on the parts that matter.
# LayerNorm / biases / 1-D scalars fall back to the SGD path of MuSGD.
# =====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


def _topk_routing(q_r, k_r, topk):
    """BiFormer-style region-level routing.
    q_r, k_r : (B, N_r, C) region tokens.
    Returns index tensor I : (B, N_r, topk)."""
    aff = torch.einsum("bqc,bkc->bqk", q_r, k_r) / (q_r.shape[-1] ** 0.5)
    _, idx = aff.topk(topk, dim=-1)          # (B, N_r, topk)
    return idx
def _deform_sample(feat, offset, dilation, n_pts):
    """Deformable sampling via grid_sample (ONNX-exportable)."""
    B, C, H, W = feat.shape
    heads = offset.shape[1]
    head_dim = C // heads  # <-- Calculate head_dim here
    
    device = feat.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=feat.dtype),
        torch.arange(W, device=device, dtype=feat.dtype), indexing="ij")
    base = torch.stack([xs, ys], dim=0).unsqueeze(0)            
    
    dy = offset[:, :, :n_pts] * dilation                        
    dx = offset[:, :, n_pts:] * dilation
    
    g = []
    for p in range(n_pts):
        gx = (base[0, 0] + dx[:, :, p]) / max(W - 1, 1) * 2 - 1
        gy = (base[0, 1] + dy[:, :, p]) / max(H - 1, 1) * 2 - 1
        g.append(torch.stack([gx, gy], dim=-1))                 
        
    grid = torch.stack(g, dim=2)                                
    grid = grid.view(B * heads, n_pts, H * W, 2)
    
    # ---------------------------------------------------------
    # FIXED: Split feature channels across heads instead of duplicating them!
    # Old: feat_rep = feat.unsqueeze(1).expand(...).reshape(B * heads, C, H, W)
    # New: Group the batch and heads, and isolate the head_dim
    feat_split = feat.view(B * heads, head_dim, H, W)
    # ---------------------------------------------------------

    sampled = F.grid_sample(feat_split, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
                            
    # sampled is now (B * heads, head_dim, n_pts, H * W)
    # Reshape and permute to match the (B, h, n_pts, head_dim, H, W) expectation in forward()
    sampled = sampled.view(B, heads, head_dim, n_pts, H, W)
    sampled = sampled.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    return sampled
class MSDPRA(nn.Module):
    """Multi-Scale Deformable Pyramid Routing Attention.

    Drop-in replacement for `Attention` inside PSABlock.
    Args mirror Attention(c, attn_ratio, num_heads) and add routing/deform knobs.
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4,
                 region_size=7, topk=4, n_pts=4,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        assert c % num_heads == 0, f"channels {c} not divisible by heads {num_heads}"
        self.c, self.h = c, num_heads
        self.region_size, self.topk, self.n_pts = region_size, topk, n_pts
        self.dilations = tuple(dilations)
        self.M = len(self.dilations)
        self.head_dim = c // num_heads

        # ---- 2-D weights -> Muon path in MuSGD ----
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.off = nn.Conv2d(c, num_heads * 2 * n_pts, 1, bias=False)
        self.gate = nn.Conv2d(c, self.M, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        # ---- 1-D params -> SGD path in MuSGD ----
        self.scale = nn.Parameter(torch.ones(self.M))

    def forward(self, x):
        B, C, H, W = x.shape
        h, M, n_pts = self.h, self.M, self.n_pts
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)                           # each (B,C,H,W)
        rs = self.region_size

        # --- 1) region-level routing (BiFormer bi-level) ---
        # pad H,W to multiple of region_size
        pad_h = (rs - H % rs) % rs
        pad_w = (rs - W % rs) % rs
        if pad_h or pad_w:
            q_r = F.pad(q, (0, pad_w, 0, pad_h))
            k_r = F.pad(k, (0, pad_w, 0, pad_h))
        else:
            q_r, k_r = q, k
        Hr, Wr = q_r.shape[-2:]
        Nr = (Hr // rs) * (Wr // rs)
        # region average pool -> (B, C, Nr)
        q_rg = F.avg_pool2d(q_r, rs).flatten(2)                 # (B,C,Nr)
        k_rg = F.avg_pool2d(k_r, rs).flatten(2)
        # region routing in channel-reshaped space (cheap proxy)
        with torch.no_grad():
            aff = q_rg.transpose(1, 2) @ k_rg                   # (B,Nr,Nr)
            idx = aff.topk(self.topk, dim=-1).indices           # (B,Nr,topk)

        # --- 2) deformable multi-scale sampling ---
        off = self.off(x)                                       # (B, h*2*n_pts, H, W)
        off = off.view(B, h, 2 * n_pts, H, W)

        # --- 3) routed + deformed token attention per scale ---
        q_flat = q.view(B, h, self.head_dim, H * W)             # (B,h,hd,HW)
        outs = []
        for m, d in enumerate(self.dilations):
            samp = _deform_sample(v, off, d, n_pts)             # (B,h,n_pts,C,H,W)
            # treat each (head, point) as a key/value token stream
            kv = samp.permute(0, 1, 2, 3, 4, 5).reshape(
                B, h * n_pts, self.head_dim * (C // self.head_dim // self.head_dim + 1)
            ) if False else samp.reshape(B, h * n_pts, C, H, W)
            # simpler & stable: per-head attention against sampled keys
            k_m = samp.reshape(B, h, n_pts, self.head_dim, H * W)  # (B,h,n_pts,hd,HW)
            k_m = k_m.permute(0, 1, 2, 4, 3)                       # (B,h,n_pts,HW,hd)
            attn = torch.einsum("bhdQ,bhnpQ->bhdnp", q_flat, k_m) / (self.head_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            v_m = samp.reshape(B, h, n_pts, self.head_dim, H * W).permute(0, 1, 2, 4, 3)
            y_m = torch.einsum("bhdnp,bhnpQ->bhdQ", attn, v_m)     # (B,h,hd,HW)
            outs.append(y_m)
        # --- 4) scale-aware fusion (gate is 2-D conv -> Muon) ---
        y = torch.stack(outs, dim=2)                              # (B,h,M,hd,HW)
        g = self.gate(x).softmax(dim=1)                           # (B,M,H,W)
        g = g.unsqueeze(1).unsqueeze(3)                           # (B,1,M,1,HW)
        y = (y * g).sum(dim=2)                                    # (B,h,hd,HW)
        y = y.reshape(B, C, H, W)
        # --- 5) output projection (2-D conv -> Muon) ---
        return self.proj(y)


class MSDPRABlock(nn.Module):
    """Drop-in replacement for PSABlock: MSDPRA + FFN + residual.
    Mirrors PSABlock(c, attn_ratio, num_heads, shortcut) signature."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True,
                 region_size=7, topk=4, n_pts=4, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.attn = MSDPRA(c, attn_ratio=attn_ratio, num_heads=num_heads,
                           region_size=region_size, topk=topk,
                           n_pts=n_pts, dilations=dilations)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2MSDPRA(nn.Module):
    """C2-style wrapper around MSDPRABlock — drop-in replacement for C2PSA."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        # 1. REMOVE the assert c1 == c2
        
        # 2. Base the hidden channels on c2 (the output), not c1
        self.c = int(c2 * e)  
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # 3. Ensure final projection goes to c2
        self.cv2 = Conv(2 * self.c, c2, 1)  
        
        self.m = nn.Sequential(*(MSDPRABlock(self.c, attn_ratio=0.5,
                                             num_heads=max(self.c // 64, 1))
                                 for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))











import torch
import torch.nn as nn
import torch.nn.functional as F

class TSMA(nn.Module):
    """Tensor-Spectral Micro-Attention for MuSGD & Tiny Objects."""
    def __init__(self, c1):
        super().__init__()
        # 1. Fixed High-Pass Spectral Isolation (Laplacian/DoG)
        # Non-optimizable 3D/4D tensor buffer (ignored by MuSGD)
        k = 3
        # Laplacian kernel for edge detection (isolates sub-10px boundaries)
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('fixed_hp_kernel', laplacian_kernel.view(1, 1, k, k))
        
        # 2. MuSGD-Compliant Orthogonal Tensor Projection (4D Conv)
        # Optimizable >2D parameters, NO bias, NO BatchNorm
        self.proj = nn.Conv2d(c1, c1, 3, 1, 1, bias=False)
        self.gate = nn.Conv2d(c1, c1, 1, 1, 0, bias=False) 

    def forward(self, x):
        # Isolate high-frequency spatial peaks (sub-10px edges) using Depthwise Conv
        hp = F.conv2d(x, self.fixed_hp_kernel.expand(x.shape[1], -1, -1, -1), padding=1, groups=x.shape[1])
        
        # 4D Tensor Projection (Orthogonalized by MuSGD)
        y = self.proj(hp)
        y = self.gate(y) 
        
        # Additive Isometric Injection (Preserves signal magnitude)
        return x + y

class C2TSMA(nn.Module):
    """CSP Bottleneck with TSMA for YOLO26 Tiny Object Detection.
    Replaces standard C3k2/C2f blocks to preserve microscopic features.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, g=g) for _ in range(n)))
        # Apply TSMA on the concatenated features before final projection
        self.tsma = TSMA(2 * c_)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        # Fused spatial features are enhanced by TSMA
        return self.cv3(self.tsma(torch.cat((y1, y2), 1)))
    














import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv  # Conv2d + BN + SiLU, matches rest of YOLO26


class SGCU(nn.Module):
    """Spectral-Gated Channel Unit. All ops are Conv (ndim>=2) -> every
    weight is auto-routed to MuSGD's Muon+SGD blend. No free 1D gates."""
    def __init__(self, c, r=8):
        super().__init__()
        cr = max(c // r, 8)
        self.dw = Conv(c, c, k=3, s=1, g=c, act=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_l1, self.fc_l2 = Conv(c, cr, k=1, act=True), Conv(cr, c, k=1, act=False)
        self.fc_g1, self.fc_g2 = Conv(c, cr, k=1, act=True), Conv(cr, c, k=1, act=False)

    def forward(self, x):
        l = self.fc_l2(self.fc_l1(self.pool(self.dw(x))))
        g = self.fc_g2(self.fc_g1(self.pool(x)))
        return x * torch.sigmoid(l + g)


class StripAttention(nn.Module):
    """Row/column-decomposed self-attention: O(HW(H+W)C) not O((HW)^2C).
    The only way attention is computationally viable at P2/P3 resolution."""
    def __init__(self, c, heads=None):
        super().__init__()
        self.heads = heads or max(c // 64, 1)  # same head-sizing convention as C2PSA
        self.qkv = Conv(c, 3 * c, k=1, act=False)
        self.proj = Conv(c, c, k=1, act=False)

    @staticmethod
    def _sdpa(q, k, v):
        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        return attn.softmax(dim=-1) @ v

    def forward(self, x):
        B, C, H, W = x.shape
        h, dh = self.heads, C // self.heads
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = (t.reshape(B, h, dh, H, W) for t in (q, k, v))

        qr, kr, vr = (t.permute(0, 3, 1, 4, 2).reshape(B * H, h, W, dh) for t in (q, k, v))
        row = self._sdpa(qr, kr, vr).reshape(B, H, h, W, dh).permute(0, 2, 4, 1, 3).reshape(B, C, H, W)

        qc, kc, vc = (t.permute(0, 4, 1, 3, 2).reshape(B * W, h, H, dh) for t in (q, k, v))
        col = self._sdpa(qc, kc, vc).reshape(B, W, h, H, dh).permute(0, 2, 4, 3, 1).reshape(B, C, H, W)

        return self.proj(row + col)


class CSSA(nn.Module):
    """Cross-Strip Spatial Attention: dilated local context + strip attention."""
    def __init__(self, c, k=7, d=3):
        super().__init__()
        self.local = Conv(c, c, k=k, s=1, g=c, d=d, act=True)
        self.strip = StripAttention(c)
        self.merge = Conv(2 * c, c, k=1, act=False)

    def forward(self, x):
        return x + self.merge(torch.cat([self.local(x), self.strip(x)], dim=1))


class MuTOA(nn.Module):
    """Muon-synchronized Tiny-Object Attention.
    YAML args: [c2, r]  where c2 = channels (in==out), r = channel reduction (default 8).
    """
    def __init__(self, c1, r=8):
        super().__init__()
        self.channel = SGCU(c1, r)
        self.spatial = CSSA(c1)

    def forward(self, x):
        return self.spatial(self.channel(x))
    




import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class MSCA(nn.Module):
    """Multi-Scale Spectral-Context Attention.
    MuSGD-aligned: separate Wq,Wk,Wv,Wo (2-D, Newton-Schulz applies);
    windowed softmax (well-conditioned grads); minimal 1-D params."""
    def __init__(self, c, reduction=2, window=4):
        super().__init__()
        ch = c // 2
        self.c, self.ch, self.window = c, ch, window
        
        # --- FSE branch ---
        # FIX 1: padding must be (21-1)*3//2 = 30
        self.low = nn.Conv2d(ch, ch, 21, padding=30, dilation=3, groups=ch)
        self.gate = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1))  # only 1-D param (routed to SGD)
        
        # --- DCA branch (SEPARATE 2-D projections -> Muon-friendly) ---
        self.wq = nn.Conv2d(ch, ch, 1, bias=False)
        self.wk = nn.Conv2d(ch, ch, 1, bias=False)
        self.wv = nn.Conv2d(ch, ch, 1, bias=False)
        self.wo = nn.Conv2d(ch, ch, 1, bias=False)
        self.dwq = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.dwk = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.dwv = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        
        # FIX 2: offset must have 2 * kH * kW channels = 18 for a 3x3 deformable conv
        self.offset = nn.Conv2d(ch, 2 * 3 * 3, 1)
        self.deform_conv = DeformConv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        
        # --- GSA ---
        # FIX 4: sp concatenation outputs 2*c channels
        self.scale = nn.Conv2d(c * 2, 1, 1)
        # FIX 5: fuse takes the combined c channels + 1 channel for sg
        self.fuse  = nn.Conv2d(c + 1, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        lo, hi = x[:, :self.ch], x[:, self.ch:]
        
        # --- FSE ---
        low = self.low(lo)
        high = lo - low
        gs = torch.sigmoid(self.gate(high))
        wh = self.gamma * (high * gs - high.mean((2,3),keepdim=True)) / (high.std((2,3),keepdim=True) + 1e-5)
        
        # --- DCA ---
        q = self.dwq(self.wq(hi))
        k = self.dwk(self.wk(hi))
        v = self.dwv(self.wv(hi))
        off = self.offset(q)
        
        # Depthwise deformable convolution (groups=ch)
        vd = self.deform_conv(v, off)
        
        # Pad if H or W is not divisible by window size to prevent silent pixel dropping
        pad_h = (self.window - H % self.window) % self.window
        pad_w = (self.window - W % self.window) % self.window
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            vd = F.pad(vd, (0, pad_w, 0, pad_h))
            
        Hp, Wp = q.shape[-2], q.shape[-1]
        
        # windowed attention (4x4) -- well-conditioned softmax
        qf = F.unfold(q, self.window, stride=self.window)            # (B, ch*w*w, L)
        kf = F.unfold(k, self.window, stride=self.window)
        vf = F.unfold(vd, self.window, stride=self.window)
        
        L = qf.shape[-1]
        P = self.window * self.window
        
        qf = qf.transpose(1, 2).reshape(B, L, self.ch, P)
        kf = kf.transpose(1, 2).reshape(B, L, self.ch, P)
        vf = vf.transpose(1, 2).reshape(B, L, self.ch, P)
        
        # FIX 3: Correct einsum to compute pixel-to-pixel attention matrix over channels
        A = torch.einsum('blcp,blcq->blpq', qf, kf) / (self.ch**0.5)
        A = A.softmax(-1)
        
        # Apply attention to values -> (B, L, ch, P)
        ctx = torch.einsum('blpq,blcq->blcp', A, vf)
        
        # Fold back to spatial dimensions
        ctx = ctx.permute(0, 1, 3, 2).reshape(B, self.ch * P, L)
        ctx = F.fold(ctx, (Hp, Wp), self.window, stride=self.window)
        
        # Crop back to original H, W if we padded
        ctx = ctx[:, :, :H, :W]
        ctx = self.wo(ctx)
        
        # --- GSA & Residual ---
        sp = torch.cat([F.avg_pool2d(x, 2), F.max_pool2d(x, 2)], 1)
        sp = F.interpolate(sp, size=(H, W), mode='nearest')
        sg = F.softplus(self.scale(sp)) + 0.5
        
        # FIX 5: Recombine wh and ctx back to C channels before residual addition
        combined = torch.cat([wh, ctx], dim=1)
        alpha = torch.sigmoid(self.fuse(torch.cat([combined, sg], dim=1)))
        
        return x + alpha * combined


class HFRA(nn.Module):
    """High-Frequency Resonance Attention (HFRA) with MoL fusion.
    Uses a frozen Laplacian branch to isolate high-frequency tiny object textures.
    Uses nn.Linear (2D matrix) for fusion so MuSGD applies Muon orthogonalization.
    """
    # FIX: Added 'n: int = 1' as the 3rd argument to intercept the Ultralytics parser
    def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, s: int = 1):
        super().__init__()
        c_ = c2
        
        # Base Semantic Branch
        self.base_branch = nn.Sequential(
            DWConv(c1, c_, k, s), 
            Conv(c_, c_, 1, 1)
        )
        
        # Frozen High-Pass Resonance Branch
        self.res_branch = nn.Conv2d(c1, c_, k, s, k//2, bias=False)
        with torch.no_grad():
            lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            w = lap.view(1, 1, 3, 3).repeat(c_, c1, 1, 1)
            self.res_branch.weight.data.copy_(w)
            self.res_branch.weight.requires_grad = False

        # MoL Fusion: nn.Linear ensures MuSGD routes this to Muon optimizer
        self.fusion = nn.Linear(2 * c_, c2)
        self.act = nn.Sigmoid()
        self.add = c1 == c2 and s == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_base = self.base_branch(x)
        x_res = self.res_branch(x)
        
        # Concatenate and permute for Linear layer [B, H, W, C]
        x_cat = torch.cat([x_base, x_res], dim=1).permute(0, 2, 3, 1)
        
        # Attention generation
        attn = self.act(self.fusion(x_cat)).permute(0, 3, 1, 2)
        
        out = x_base * attn
        return out + x if self.add else out

class HighFreqInject(nn.Module):
    """Injects high-frequency spatial details from a higher-resolution feature map (e.g., P2) into a lower-resolution map (e.g., P3)."""
    def __init__(self, c1, c2):
        super().__init__()
        # Laplacian kernel for edge detection (high-pass filter)
        self.laplacian = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        kernel = torch.tensor([[[[0., 1., 0.],
                                 [1., -4., 1.],
                                 [0., 1., 0.]]]], dtype=torch.float32)
        self.laplacian.weight.data = kernel.repeat(c1, 1, 1, 1)
        self.laplacian.requires_grad_(False) # Fixed high-pass filter
        
        # Projection to match target channels
        self.proj = Conv(c1, c2, 1, 1)

    def forward(self, x):
        # x is a list: [target_features (P3), source_features (P2)]
        target, source = x[0], x[1]
        edges = self.laplacian(source)
        return target + self.proj(edges)