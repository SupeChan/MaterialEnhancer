import av
from av import VideoStream
from PIL import Image
import numpy as np
from fractions import Fraction
import pathlib


from constants.save import CODEC, PIXEL_FORMAT, VIDEO_FORMAT


def save_mov(path_output: pathlib.Path, images: list[Image.Image], rate: int|Fraction):
    container = av.open(path_output, mode="w")
    stream = container.add_stream(CODEC, rate=rate)
    if not isinstance(stream, VideoStream):
        return

    stream.width = images[0].width
    stream.height = images[0].height
    stream.pix_fmt = PIXEL_FORMAT
    for im in images:
        nd = np.array(im)
        frame = av.VideoFrame.from_ndarray(nd, format=VIDEO_FORMAT)
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


if __name__ == "__main__":
    from natsort import natsorted
    dir_video = pathlib.Path("../sample/output_mov")
    files = natsorted([path for path in dir_video.glob("*.png")])
    images = [Image.open(path).convert("RGBA") for path in files]
    save_mov(dir_video/"output.mov", images,10)
