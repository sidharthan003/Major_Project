import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


def _window_partition(x: torch.Tensor, ws: int):
    b, c, h, w = x.shape
    ph = (ws - h % ws) % ws
    pw = (ws - w % ws) % ws
    if ph or pw:
        x = nn.functional.pad(x, (0, pw, 0, ph))
    hp, wp = x.shape[2], x.shape[3]
    x = x.view(b, c, hp // ws, ws, wp // ws, ws).permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, ws * ws, c), hp, wp


def _window_reverse(windows: torch.Tensor, ws: int, b: int, c: int, hp: int, wp: int):
    nh, nw = hp // ws, wp // ws
    x = windows.view(b, nh, nw, ws, ws, c).permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, hp, wp)


class SwinWindowBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, ws: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.ws = ws
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        windows, hp, wp = _window_partition(x, self.ws)
        y = self.n1(windows)
        y, _ = self.attn(y, y, y, need_weights=False)
        windows = windows + y
        y = self.n2(windows)
        windows = windows + self.mlp(y)
        x = _window_reverse(windows, self.ws, b, c, hp, wp)
        return x[:, :, :h, :w]


class C2f_Swin(nn.Module):
    # Compatible with YOLO YAML parser params
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, heads=4, ws=8):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(SwinWindowBlock(self.c, heads=heads, ws=ws) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, 1))