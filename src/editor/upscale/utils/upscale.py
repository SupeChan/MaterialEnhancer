import torch
from waifu2x.hub import waifu2x

from PIL import Image

from constants.upscale import ScaleMethod


def upscale(im: Image.Image, scale: str) -> Image.Image:
    if scale == ScaleMethod.X1:
        return im

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = waifu2x(model_type="art_scan", method=scale).to(device)
    pred = model.infer(im)
    if not isinstance(pred, Image.Image):
        raise Exception("upscale failed")

    return pred
