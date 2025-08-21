import pathlib

from PIL import Image


def save_png(dir_output: pathlib.Path, images: list[Image.Image]):
    dir_output.mkdir(exist_ok=True, parents=True)
    w = len(str(len(images)))+1
    for ix, im in enumerate(images):
        path_save = dir_output / f"{str(ix+1).zfill(w)}.png"
        im.save(path_save)
