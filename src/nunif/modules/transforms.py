import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import lru_cache
from . replication_pad2d import replication_pad2d_naive
from . reflection_pad2d import reflection_pad2d_naive

# differentiable transforms for loss function


def rotate_grid(batch, height, width, angle, device, dtype):
    with torch.no_grad():
        angle = math.radians(angle)
        py, px = torch.meshgrid(torch.linspace(-1, 1, height, device=device, dtype=dtype),
                                torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij")
        mesh_x = px * math.cos(angle) - py * math.sin(angle)
        mesh_y = px * math.sin(angle) + py * math.cos(angle)
        grid = torch.stack((mesh_x, mesh_y), 2).unsqueeze(0).repeat(batch, 1, 1, 1).contiguous().detach()
    return grid


@lru_cache
def rotate_grid_cache(batch, height, width, angle, device, dtype):
    return rotate_grid(batch, height, width, angle, device, dtype)


PAD_MODE_NN = {
    "zeros": "constant",
    "reflection": "reflect",
    "border": "replicate",
}


def _pad(input, pad, mode="constant", value=0):
    if mode == "reflect":
        return reflection_pad2d_naive(input, pad, detach=True)
    elif mode == "replicate":
        return replication_pad2d_naive(input, pad, detach=True)
    else:
        return F.pad(input, pad, mode=mode, value=value).contiguous()


def diff_rotate(x, angle, mode="bilinear", padding_mode="zeros", align_corners=False, expand=False, cache=True):
    if expand and padding_mode not in {"zeros", "constant"}:
        raise ValueError(f"expand=True does not support padding_mode={padding_mode}")

    # x: BCHW
    B, _, H, W = x.shape
    pad_h = (int(2 ** 0.5 * H) - H) // 2 + 1
    pad_w = (int(2 ** 0.5 * W) - W) // 2 + 1
    x = _pad(x, (pad_w, pad_w, pad_h, pad_h),
             mode=PAD_MODE_NN.get(padding_mode, padding_mode), value=0)
    B, _, H, W = x.shape

    if cache:
        grid = rotate_grid_cache(B, H, W, angle, device=x.device, dtype=x.dtype)
    else:
        grid = rotate_grid(B, H, W, angle, device=x.device, dtype=x.dtype)
    x = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=align_corners)
    if not expand:
        x = F.pad(x, (-pad_w, -pad_w, -pad_h, -pad_h))

    return x.contiguous()


def diff_random_rotate(x, angle=45, mode="bilinear", padding_mode="zeros", align_corners=False, expand=False):
    B, _, H, W = x.shape
    angle = (torch.rand((B,), device=x.device) * 2 - 1) * angle
    # FIXME: remove loop
    return torch.cat([diff_rotate(x[i:i + 1, :, :, :], angle=angle[i].item(),
                                  mode=mode, padding_mode=padding_mode,
                                  align_corners=align_corners, expand=expand, cache=False)
                      for i in range(B)], dim=0).contiguous()


def diff_random_rotate_pair(x, y, angle=45, mode="bilinear", padding_mode="zeros", align_corners=False, expand=False):
    assert x.shape[0] == y.shape[0]
    B, _, H, W = x.shape
    angle = (torch.rand((B,), device=x.device) * 2 - 1) * angle

    if x.dtype == y.dtype:
        xy = torch.stack((x, y), dim=1)
        xys = []
        for i in range(B):
            xys.append(diff_rotate(
                xy[i], angle=angle[i].item(),
                mode=mode, padding_mode=padding_mode,
                align_corners=align_corners, expand=expand, cache=False))
        x = torch.stack([xyi[0] for xyi in xys], dim=0)
        y = torch.stack([xyi[1] for xyi in xys], dim=0)
    else:
        x = torch.cat([diff_rotate(
            x[i:i + 1, :, :, :], angle=angle[i].item(),
            mode=mode, padding_mode=padding_mode,
            align_corners=align_corners, expand=expand, cache=False) for i in range(B)], dim=0)
        y = torch.cat([diff_rotate(
            y[i:i + 1, :, :, :], angle=angle[i].item(),
            mode=mode, padding_mode=padding_mode,
            align_corners=align_corners, expand=expand, cache=False) for i in range(B)], dim=0)

    return x.contiguous(), y.contiguous()


def diff_translate(x, x_shift, y_shift, padding_mode="zeros", expand_x=0, expand_y=0):
    return _pad(x, (x_shift + expand_x, -x_shift + expand_x,
                    y_shift + expand_y, -y_shift + expand_y),
                mode=PAD_MODE_NN.get(padding_mode, padding_mode), value=0).contiguous()


def diff_random_translate(x, ratio=0.15, size=None, padding_mode="zeros", expand=False):
    B, _, H, W = x.shape
    if size is not None:
        x_shift = torch.randint(low=-size, high=size + 1, size=(B,), device=x.device)
        y_shift = torch.randint(low=-size, high=size + 1, size=(B,), device=x.device)
    else:
        x_shift = torch.randint(low=int(-W * ratio), high=int(W * ratio) + 1, size=(B,), device=x.device)
        y_shift = torch.randint(low=int(-H * ratio), high=int(H * ratio) + 1, size=(B,), device=x.device)

    if expand:
        expand_x, expand_y = (size, size) if size is not None else (int(W * ratio), int(H * ratio))
    else:
        expand_x = expand_y = 0

    # FIXME: remove loop
    return torch.cat([diff_translate(x[i:i + 1, :, :, :], x_shift=x_shift[i], y_shift=y_shift[i],
                                     padding_mode=padding_mode,
                                     expand_x=expand_x, expand_y=expand_y) for i in range(B)], dim=0).contiguous()


def diff_random_translate_pair(x, y, ratio=0.15, size=None, padding_mode="zeros", expand=False):
    assert x.shape[0] == y.shape[0]
    B, _, H, W = x.shape
    if size is not None:
        x_shift = torch.randint(low=-size, high=size + 1, size=(B,), device=x.device)
        y_shift = torch.randint(low=-size, high=size + 1, size=(B,), device=x.device)
    else:
        x_shift = torch.randint(low=int(-W * ratio), high=int(W * ratio) + 1, size=(B,), device=x.device)
        y_shift = torch.randint(low=int(-H * ratio), high=int(H * ratio) + 1, size=(B,), device=x.device)

    if expand:
        expand_x, expand_y = (size, size) if size is not None else (int(W * ratio), int(H * ratio))
    else:
        expand_x = expand_y = 0

    if x.dtype == y.dtype:
        xy = torch.stack((x, y), dim=1)
        xys = []
        for i in range(B):
            xys.append(diff_translate(
                xy[i], x_shift=x_shift[i], y_shift=y_shift[i],
                padding_mode=padding_mode,
                expand_x=expand_x, expand_y=expand_y))
        x = torch.stack([xyi[0] for xyi in xys], dim=0)
        y = torch.stack([xyi[1] for xyi in xys], dim=0)
    else:
        x = torch.cat([diff_translate(
            x[i:i + 1, :, :, :], x_shift=x_shift[i], y_shift=y_shift[i],
            padding_mode=padding_mode,
            expand_x=expand_x, expand_y=expand_y) for i in range(B)], dim=0)
        y = torch.cat([diff_translate(
            y[i:i + 1, :, :, :], x_shift=x_shift[i], y_shift=y_shift[i],
            padding_mode=padding_mode,
            expand_x=expand_x, expand_y=expand_y) for i in range(B)], dim=0)

    return x.contiguous(), y.contiguous()


class DiffPairRandomTranslate(nn.Module):
    def __init__(self, ratio=0.15, size=None, padding_mode="zeros", expand=False, instance_random=False):
        super().__init__()
        self.ratio = ratio
        self.size = size
        self.expand = expand
        self.padding_mode = padding_mode
        self.instance_random = instance_random

    @staticmethod
    def expand_pad(input, target, ratio=0.15, size=None, padding_mode="zeros"):
        size = size if size else int(input.shape[2:] * ratio)
        expand_x = expand_y = size
        padding_mode = PAD_MODE_NN.get(padding_mode, padding_mode)
        pad = (expand_x, expand_x, expand_y, expand_y)
        input = _pad(input, pad, mode=padding_mode, value=0)
        target = _pad(target, pad, mode=padding_mode, value=0)
        return input.contiguous(), target.contiguous()

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0]
        if self.training:
            if self.instance_random:
                return diff_random_translate_pair(input, target, ratio=self.ratio, size=self.size,
                                                  padding_mode=self.padding_mode, expand=self.expand)
            else:
                # batch random
                size = self.size if self.size else int(input.shape[2] * self.ratio)
                shift = torch.randint(low=-size, high=size + 1, size=(2,))
                x_shift, y_shift = shift[0], shift[1]
                if self.expand:
                    expand_x = expand_y = size
                else:
                    expand_x = expand_y = 0
                input = diff_translate(input, x_shift=x_shift, y_shift=y_shift, padding_mode=self.padding_mode,
                                       expand_x=expand_x, expand_y=expand_y)
                target = diff_translate(target, x_shift=x_shift, y_shift=y_shift, padding_mode=self.padding_mode,
                                        expand_x=expand_x, expand_y=expand_y)
                return input, target
        else:
            if self.expand:
                return self.expand_pad(input, target, ratio=self.ratio, size=self.size,
                                       padding_mode=self.padding_mode)
            else:
                return input, target


class DiffPairRandomRotate(nn.Module):
    def __init__(self, angle=45, mode="bilinear", padding_mode="zeros",
                 align_corners=False, expand=False, instance_random=False):
        super().__init__()
        self.angle = angle
        self.mode = mode
        self.expand = expand
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.instance_random = instance_random

    @staticmethod
    def expand_pad(input, target, padding_mode):
        H, W = input.shape[:2]
        pad_h = (int(2 ** 0.5 * H) - H) // 2 + 1
        pad_w = (int(2 ** 0.5 * W) - W) // 2 + 1
        input = _pad(input, (pad_w, pad_w, pad_h, pad_h),
                     mode=PAD_MODE_NN.get(padding_mode, padding_mode), value=0)
        target = _pad(input, (pad_w, pad_w, pad_h, pad_h),
                      mode=PAD_MODE_NN.get(padding_mode, padding_mode), value=0)
        return input.contiguous(), target.contiguous()

    def forward(self, input, target):
        if self.training:
            if self.instance_random:
                return diff_random_rotate_pair(
                    input, target,
                    angle=self.angle, mode=self.mode, padding_mode=self.padding_mode,
                    align_corners=self.align_corners, expand=self.expand)
            else:
                # batch random
                angle = (torch.rand(1).item() * 2 - 1) * self.angle
                input = diff_rotate(input, angle, mode=self.mode, padding_mode=self.padding_mode,
                                    align_corners=self.align_corners, expand=self.expand, cache=False)
                target = diff_rotate(target, angle, mode=self.mode, padding_mode=self.padding_mode,
                                     align_corners=self.align_corners, expand=self.expand, cache=False)
                return input, target
        else:
            if self.expand:
                return self.expand_pad(input, target, padding_mode=self.padding_mode)
            else:
                return input, target


class DiffPairRandomDownsample(nn.Module):
    def __init__(self, scale_factor_min=0.5, scale_factor_max=1.0):
        super().__init__()
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max

    def forward(self, input, target):
        if self.training:
            scale_factor = (self.scale_factor_max - self.scale_factor_min) * torch.rand(1).item() + self.scale_factor_min
        else:
            scale_factor = (self.scale_factor_max - self.scale_factor_min) * 0.5 + self.scale_factor_min

        input = F.interpolate(input, scale_factor=scale_factor, mode="bilinear", align_corners=False, antialias=True)
        target = F.interpolate(target, scale_factor=scale_factor, mode="bilinear", align_corners=False, antialias=True)
        return input, target


def _test_rotate():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import time

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)
    # TF.to_pil_image(x[0]).show()
    print(x.shape)

    z = diff_rotate(x, 45, expand=False)
    TF.to_pil_image(z[0]).show()
    time.sleep(0.5)

    z = diff_rotate(x, 45, expand=False, padding_mode="reflection")
    TF.to_pil_image(z[0]).show()

    x = x.repeat(4, 1, 1, 1)
    z = diff_random_rotate(x, padding_mode="reflection")
    for i in range(4):
        TF.to_pil_image(z[i]).show()
        time.sleep(1)


def _test_translate():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import time

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)

    x = x.repeat(4, 1, 1, 1)
    z = diff_random_translate(x, padding_mode="reflection")
    for i in range(4):
        TF.to_pil_image(z[i]).show()
        time.sleep(1)


def _test_translate_random():
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    import time

    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)
    x = x.repeat(4, 1, 1, 1)

    z1, z2 = diff_random_translate_pair(x, x, padding_mode="zeros", expand=True)
    for i in range(4):
        assert (z1 - z2).abs().sum() == 0
        TF.to_pil_image(z1[i]).show()
        time.sleep(1)


def _bench_random_rotate_pair():
    import time
    N = 100
    B = 16
    x = torch.randn((B, 3, 512, 512)).cuda()
    y = torch.randn((B, 3, 512, 512)).cuda()

    t = time.time()
    for _ in range(N):
        diff_random_rotate_pair(x, y)
    torch.cuda.synchronize()
    print(1 / ((time.time() - t) / (B * N)), "FPS")


def _test_compose():
    import time
    import torchvision.io as IO
    import torchvision.transforms.functional as TF
    from .. transforms import pair as TP

    transform = TP.Compose([
        TP.RandomChoice([DiffPairRandomRotate(padding_mode="reflection"),
                        DiffPairRandomTranslate(padding_mode="reflection")])
    ])
    x = IO.read_image("cc0/dog2.jpg") / 255.0
    x = x[:, :256, :256].unsqueeze(0)
    for _ in range(5):
        x, y = transform(x, x)
        TF.to_pil_image(x[0]).show()
        time.sleep(1)


if __name__ == "__main__":
    # _test_rotate()
    # _test_translate()
    # _test_translate_random()
    # _bench_random_rotate_pair()
    _test_compose()
