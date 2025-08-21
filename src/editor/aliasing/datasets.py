import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from natsort import natsorted

import pathlib
from PIL import Image

import constants.aliasing as const

import random


class AlphaDataset(Dataset):
    def __init__(self, folder: pathlib.Path, transform=ToTensor(), target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.folder_raw = folder/"raw"
        self.folder_input = folder/"input"
        self.folder_target = folder/"target"

        self.folder_input.mkdir(exist_ok=True)
        self.folder_target.mkdir(exist_ok=True)
        existsFile = any(p.is_file() for p in self.folder_input.iterdir())
        if not existsFile:
            self.output_files()

        self.items = self.preprocess()

    def output_files(self):
        for file in (self.folder_raw).glob("*.png"):
            im_target = Image.open(file).convert("RGBA").split()[-1]
            bbox = im_target.getbbox()
            im_target = im_target.crop(bbox)
            im_target.thumbnail(
                size=(500, 500), resample=Image.Resampling.BICUBIC)
            canvas = Image.new("L", (500, 500), 0)

            w, h = im_target.size
            w_canvas, h_canvas = canvas.size
            offset = ((w_canvas - w) // 2, (h_canvas-h)//2)

            canvas.paste(im_target, offset)
            im_target = canvas

            im_input = Image.eval(im_target, lambda a: 0 if a <= 125 else 255)

            im_target.save(self.folder_target/f"{file.stem}.png")
            im_input.save(self.folder_input/f"{file.stem}.png")

    def preprocess(self):
        files_input = natsorted(
            [file for file in self.folder_input.glob("*.png")])
        files_target = natsorted(
            [file for file in self.folder_target.glob("*.png")])

        items: list[tuple[Tensor, Tensor]] = []
        for input, target in zip(files_input, files_target):
            im_input = Image.open(input).convert("L")
            im_target = Image.open(target).convert("L")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            im_input = self.transform(im_input).to(device)
            im_target = self.transform(im_target).to(device)
            items.append((im_input, im_target))

        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        im_input, im_target = self.items[ix]
        k = random.randint(0, 4)

        return im_input.rot90(k, dims=(1, 2)), im_target.rot90(k, dims=(1, 2))



