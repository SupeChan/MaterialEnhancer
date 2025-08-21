import torch
from torch import Tensor
from torchvision.transforms import ToTensor

from PIL import Image, ImageChops
import cv2
import numpy as np

from ..model import model_aliasing


def aliasing(im: Image.Image) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im = im.convert("RGBA")
    alpha = im.split()[-1]
    alpha = Image.eval(alpha, lambda a: 0 if a <= 100 else 255).convert("L")
    alpha = ToTensor()(alpha).to(device).unsqueeze(0)

    model_aliasing.eval()
    pred: Tensor = model_aliasing(alpha)
    pred.sigmoid_().mul_(255)
    alpha = pred.squeeze(0).squeeze(0)
    alpha = alpha.to("cpu").detach().numpy().astype("uint8")
    mask = Image.fromarray(alpha).convert("L")

    im_dilated = dilate_outline(im)
    im_dilated.alpha_composite(im)
    im_dilated.putalpha(mask)
    return im_dilated


def dilate_outline(im: Image.Image) -> Image.Image:
    alpha = im.convert("RGBA").split()[-1]
    nd_alpha = np.array(alpha, np.uint8)

    thresh = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, nd_thresh = cv2.threshold(nd_alpha, 0, 255, thresh)
    cnts, _ = cv2.findContours(
        nd_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in cnts:
        cv2.drawContours(nd_thresh, [cnt], 0, (255, 255, 255), 5)

    alpha_outline = Image.fromarray(nd_thresh).convert("L")
    alpha_outline = Image.eval(alpha_outline, lambda e: 0 if e == 0 else 255)
    alpha_clip = Image.eval(alpha, lambda e: 0 if e == 0 else 255)

    alpha_clip = ImageChops.darker(alpha_outline, alpha_clip)
    alpha_clip = ImageChops.invert(alpha_clip)
    alpha_outline = ImageChops.darker(alpha_outline, alpha_clip)
    alpha_outline = ImageChops.invert(alpha_outline)

    canvas = Image.new("RGBA", im.size, (0, 0, 0, 255))
    canvas.alpha_composite(im)
    nd_canvas = np.array(canvas)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    nd_canvas = cv2.dilate(nd_canvas, kernel, iterations=1)
    canvas = Image.fromarray(nd_canvas).convert("RGBA")
    alpha_outline = ImageChops.invert(alpha_outline)
    canvas.putalpha(alpha_outline)
    return canvas
