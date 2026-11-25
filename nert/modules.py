import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules import Conv, C2f
from ultralytics.nn.modules.head import v10Detect


def conv_bn(c1, c2, kernel, groups=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel, padding=dilation * (kernel // 2),
                  groups=groups, dilation=dilation, bias=False),
        nn.BatchNorm2d(c2),
    )


class RepStage(nn.Module):
    """Sum compatible linear branches, then ReLU; fuse exactly in eval mode."""
    def __init__(self, c1, c2, kernel, groups, k=3, dilation=1):
        super().__init__()
        if k < 1 or dilation < 1:
            raise ValueError("k and dilation must be positive")
        self.c1, self.c2 = c1, c2
        self.kernel, self.groups, self.dilation = kernel, groups, dilation
        self.branches = nn.ModuleList(conv_bn(c1, c2, kernel, groups, dilation) for _ in range(k))
        self.scale = conv_bn(c1, c2, 1, groups) if kernel > 1 else None
        self.identity = nn.BatchNorm2d(c1) if c1 == c2 else None

    def forward(self, x):
        if hasattr(self, "reparam"):
            return F.relu(self.reparam(x))
        y = sum(branch(x) for branch in self.branches)
        if self.scale is not None:
            y = y + self.scale(x)
        if self.identity is not None:
            y = y + self.identity(x)
        return F.relu(y)

    @staticmethod
    def _fold(w, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return w * scale[:, None, None, None], bn.bias - bn.running_mean * scale

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, "reparam"):
            return self
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm running statistics")
        weights, biases = zip(*(self._fold(b[0].weight, b[1]) for b in self.branches))
        w, bias = sum(weights), sum(biases)
        if self.scale is not None:
            sw, sb = self._fold(self.scale[0].weight, self.scale[1])
            w = w + F.pad(sw, [self.kernel // 2] * 4)
            bias = bias + sb
        if self.identity is not None:
            identity = torch.zeros_like(w)
            indices = torch.arange(self.c2, device=w.device)
            identity[indices, indices % (self.c1 // self.groups), self.kernel // 2, self.kernel // 2] = 1
            iw, ib = self._fold(identity, self.identity)
            w, bias = w + iw, bias + ib
        self.reparam = nn.Conv2d(
            self.c1, self.c2, self.kernel, padding=self.dilation * (self.kernel // 2),
            dilation=self.dilation, groups=self.groups, bias=True,
        ).to(device=w.device, dtype=w.dtype)
        self.reparam.weight.copy_(w)
        self.reparam.bias.copy_(bias)
        del self.branches, self.scale, self.identity
        return self


class DMO(nn.Module):
    def __init__(self, channels, k=3, dilation=1):
        super().__init__()
        self.spatial = RepStage(channels, channels, 3, channels, k, dilation)
        self.pointwise = RepStage(channels, channels, 1, 1, k)

    def forward(self, x):
        return self.pointwise(self.spatial(x))


class C3kDMO(nn.Module):
    def __init__(self, c, k, dilation):
        super().__init__()
        hidden = max(1, c // 2)
        self.left, self.right = Conv(c, hidden, 1), Conv(c, hidden, 1)
        self.blocks = nn.Sequential(DMO(hidden, k, dilation), DMO(hidden, k, dilation))
        self.out = Conv(2 * hidden, c, 1)

    def forward(self, x):
        return self.out(torch.cat((self.blocks(self.left(x)), self.right(x)), dim=1))


class C3k2DMO(C2f):
    """C3k2 feature aggregation with two DMO units per internal C3k block."""
    def __init__(self, c1, c2, n=1, c3k=False, e=.5, g=1, shortcut=True, k=3, dilation=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3kDMO(self.c, k, dilation) if c3k else DMO(self.c, k, dilation)
            for _ in range(n)
        )


class LSConv(nn.Module):
    """7x7 context -> group-specific local weights -> weighted neighborhoods.

Weights are softmax-normalized across the Ks*Ks offsets.
Each group's channels share a position-specific kernel. Native unfold is portable
and allocates a neighborhood tensor during aggregation.
"""
    def __init__(self, channels, groups=4, small_kernel=3):
        super().__init__()
        if groups < 1 or channels % groups or small_kernel < 1 or small_kernel % 2 != 1:
            raise ValueError("groups must divide channels and small_kernel must be positive and odd")
        self.groups, self.ks = groups, small_kernel
        hidden = max(8, channels // 2)
        self.context = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, hidden, 7, g=hidden),
                                     nn.Conv2d(hidden, groups * small_kernel ** 2, 1))
        self.out = Conv(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        weights = self.context(x).reshape(b, self.groups, 1, self.ks ** 2, h * w).softmax(dim=3)
        patches = F.unfold(x, self.ks, padding=self.ks // 2)
        patches = patches.reshape(b, self.groups, c // self.groups, self.ks ** 2, h * w)
        return self.out((patches * weights).sum(dim=3).reshape(b, c, h, w))


class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(1, c // 4), 1),
                                nn.ReLU(), nn.Conv2d(max(1, c // 4), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


class LSBlock(nn.Module):
    def __init__(self, c, groups=4, small_kernel=3):
        super().__init__()
        self.dw = Conv(c, c, 3, g=c, act=False)
        self.ffn1 = nn.Sequential(Conv(c, 2 * c, 1), Conv(2 * c, c, 1, act=False))
        self.spatial = nn.Sequential(LSConv(c, groups, small_kernel), SE(c))
        self.ffn2 = nn.Sequential(Conv(c, 2 * c, 1), Conv(2 * c, c, 1, act=False))

    def forward(self, x):
        x = x + self.dw(x)
        x = x + self.ffn1(x)
        x = x + self.spatial(x)
        return x + self.ffn2(x)


class C2LS(nn.Module):
    def __init__(self, c1, c2, n=1, e=.5, groups=4, small_kernel=3):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, 2 * hidden, 1)
        self.blocks = nn.Sequential(*(LSBlock(hidden, groups, small_kernel) for _ in range(n)))
        self.cv2 = Conv(2 * hidden, c2, 1)

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat((a, self.blocks(b)), dim=1))


class NFEH(v10Detect):
    """YOLOv10 dual-assignment head; inference computes only one-to-one branches."""
    def forward(self, x):
        if self.training or getattr(self, "validation_loss", False):
            return super().forward(x)
        one = [torch.cat((self.one2one_cv2[i](f), self.one2one_cv3[i](f)), 1)
               for i, f in enumerate(x)]
        decoded = self._inference(one)
        output = self.postprocess(decoded.permute(0, 2, 1), self.max_det, self.nc)
        return output if self.export else (output, {"one2one": one})

    def bias_init(self):
        # The base helper initializes both training and inference branches.
        super().bias_init()


@torch.no_grad()
def reparameterize(model):
    model.eval()
    for module in list(model.modules()):
        if isinstance(module, RepStage):
            module.switch_to_deploy()
    return model
