import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    functional as TF,
    InterpolationMode,
)
from torchvision import transforms as T
from nunif.utils.image_loader import ImageLoader
from nunif.utils import pil_io
from nunif.transforms import pair as TP
from nunif.transforms.cutmix import CutMix
from nunif.transforms.mixup import RandomOverlay
from nunif.training.sampler import HardExampleSampler, MiningMethod
import nunif.transforms as TS
from nunif.transforms import image_magick as IM
from .jpeg_noise import (
    RandomJPEGNoiseX,
    choose_validation_jpeg_quality,
    add_jpeg_noise,
    shift_jpeg_block,
)
from .photo_noise import RandomPhotoNoiseX, add_validation_noise
from PIL.Image import Resampling


NEAREST_PREFIX = "__NEAREST_"
DOT_SCALE2X_PREFIX = "__NEAREST__DOT_2x_"
DOT_SCALE4X_PREFIX = "__NEAREST__DOT_4x_"
SCREENTONE_PREFIX = "__SCREENTONE_"
DOT_PREFIX = "__DOT_"
INTERPOLATION_MODES = (
    "box",
    "sinc",
    "lanczos",
    "triangle",
    "catrom",
)
INTERPOLATION_NEAREST = "box"
INTERPOLATION_BICUBIC = "catrom"
INTERPOLATION_MODE_WEIGHTS = (1/3, 1/3, 1/6, 1/16, 1/3)  # noqa: E226


def _resize(im, size, filter_type, blur):
    if filter_type in {"box", "sinc", "lanczos", "triangle", "catrom"}:
        return IM.resize(im, size, filter_type, blur)
    elif filter_type == "vision.bicubic_no_antialias":
        return TF.resize(im, size, InterpolationMode.BICUBIC, antialias=False)
    elif filter_type == "vision.bilinear_no_antialias":
        return TF.resize(im, size, InterpolationMode.BILINEAR, antialias=False)
    else:
        raise ValueError(filter_type)


def resize(im, size, filter_type, blur,
           enable_step=False, step_p=0.0,
           enable_no_antialias=False, no_antialias_p=0.0):
    if enable_step and filter_type != INTERPOLATION_NEAREST and step_p > 0 and random.uniform(0, 1) < step_p:
        h, w = im.shape[1:]
        scale = h / size[0]
        step1_scale = random.uniform(1, scale)
        step1_h, step1_w = int(step1_scale * h), int(step1_scale * w)
        im = _resize(im, (step1_h, step1_w), filter_type, 1)
        im = _resize(im, size, filter_type, blur)
        return im
    elif enable_no_antialias and random.uniform(0, 1) < no_antialias_p and filter_type != INTERPOLATION_NEAREST:
        filter_type = random.choice(["vision.bilinear_no_antialias", "vision.bicubic_no_antialias"])
        return _resize(im, size, filter_type, blur)
    else:
        return _resize(im, size, filter_type, blur)


def pil_resize(im, size, filter_type):
    if filter_type == "box":
        resample = Resampling.BOX
    elif filter_type == "catrom":
        resample = Resampling.BICUBIC
    elif filter_type in {"sinc", "lanczos"}:
        resample = Resampling.LANCZOS
    elif filter_type == "triangle":
        resample = Resampling.BILINEAR
    else:
        raise NotImplementedError()

    return im.resize(size, resample=resample)


class RandomDownscaleX():
    def __init__(self, scale_factor,
                 blur_shift=0, resize_blur_p=0.1, resize_blur_range=0.05,
                 resize_step_p=0, resize_no_antialias_p=0,
                 interpolation=None, training=True):
        assert scale_factor in {2, 4, 8}
        self.interpolation = interpolation
        self.scale_factor = scale_factor
        self.blur_shift = blur_shift
        self.training = training
        if isinstance(resize_blur_range, (list, tuple)):
            if len(resize_blur_range) == 1:
                self.resize_blur_range = [-resize_blur_range[0], resize_blur_range[0]]
            elif len(resize_blur_range) == 2:
                self.resize_blur_range = [resize_blur_range[0], resize_blur_range[1]]
            else:
                raise ValueError("resize_blur_range")
        else:
            self.resize_blur_range = [-resize_blur_range, resize_blur_range]
        self.resize_blur_p = resize_blur_p
        self.resize_step_p = resize_step_p
        self.resize_no_antialias_p = resize_no_antialias_p

    def __call__(self, x, y):
        w, h = x.size
        if self.scale_factor == 1:
            return x, y
        assert (w % self.scale_factor == 0 and h % self.scale_factor == 0)
        if self.interpolation is None:
            interpolation = random.choices(INTERPOLATION_MODES, weights=INTERPOLATION_MODE_WEIGHTS, k=1)[0]
            fixed_interpolation = False
        else:
            interpolation = self.interpolation
            fixed_interpolation = True
        if self.scale_factor in {2, 4}:
            x = pil_io.to_tensor(x)
            if not self.training:
                blur = 1
            elif random.uniform(0, 1) < self.resize_blur_p:
                blur = 1 + random.uniform(self.resize_blur_range[0] + self.blur_shift,
                                          self.resize_blur_range[1] + self.blur_shift)
            else:
                blur = 1
            x = resize(x, size=(h // self.scale_factor, w // self.scale_factor),
                       filter_type=interpolation, blur=blur,
                       enable_step=self.training and not fixed_interpolation, step_p=self.resize_step_p,
                       enable_no_antialias=self.training and not fixed_interpolation, no_antialias_p=self.resize_no_antialias_p)
            x = pil_io.to_image(x)
        elif self.scale_factor == 8:
            # wand 8x downscale is very slow for some reason
            # and, 8x is not used directly, so use pil instead
            x = pil_resize(x, (h // self.scale_factor, w // self.scale_factor), interpolation)

        return x, y


class AntialiasX():
    def __init__(self):
        pass

    def __call__(self, x, y):
        W, H = x.size
        interpolation = random.choice([InterpolationMode.BICUBIC, InterpolationMode.BILINEAR])
        if random.uniform(0, 1) < 0.5:
            scale = 2
        else:
            scale = random.uniform(1.5, 2)
        x = TF.resize(x, (int(H * scale), int(W * scale)), interpolation=interpolation, antialias=True)
        x = TF.resize(x, (H, W), interpolation=InterpolationMode.BICUBIC, antialias=True)
        return x, y


class Waifu2xDatasetBase(Dataset):
    def __init__(self, input_dir, num_samples,
                 hard_example_history_size=6,
                 exclude_filter=None,
                 additional_data_dir=None, additional_data_dir_p=0.01):
        super().__init__()
        self.files = ImageLoader.listdir(input_dir)
        if exclude_filter is not None:
            self.files = list(filter(exclude_filter, self.files))
        if not self.files:
            raise RuntimeError(f"{input_dir} is empty")
        if additional_data_dir:
            self.additional_files = ImageLoader.listdir(additional_data_dir)
            if exclude_filter is not None:
                self.additional_files = list(filter(exclude_filter, self.additional_files))
            if not self.additional_files:
                raise RuntimeError(f"{additional_data_dir} is empty")
        else:
            self.additional_files = []
        self.additional_data_dir_p = additional_data_dir_p
        self.num_samples = num_samples
        self.hard_example_history_size = hard_example_history_size

    def create_sampler(self):
        if self.additional_files:
            p1 = 1.0 - self.additional_data_dir_p
            p2 = self.additional_data_dir_p
            p1 = p1 * (1.0 / len(self.files))
            p2 = p2 * (1.0 / len(self.additional_files))
            base_weights = torch.full((len(self.files),), fill_value=p1, dtype=torch.double)
            additional_weights = torch.full((len(self.additional_files),), fill_value=p2, dtype=torch.double)
            weights = torch.cat((base_weights, additional_weights), dim=0)
        else:
            weights = torch.ones((len(self),), dtype=torch.double)

        return HardExampleSampler(
            weights,
            num_samples=self.num_samples,
            method=MiningMethod.TOP10,
            history_size=self.hard_example_history_size,
            scale_factor=4.,
        )

    def worker_init(self, worker_id):
        pass

    def __len__(self):
        return len(self.files) + len(self.additional_files)

    def __getitem__(self, index):
        if index < len(self.files):
            return self.files[index]
        else:
            index = index - len(self.files)
            return self.additional_files[index]


class Waifu2xDataset(Waifu2xDatasetBase):
    def __init__(self, input_dir,
                 model_offset,
                 scale_factor,
                 tile_size, num_samples=None,
                 da_jpeg_p=0, da_scale_p=0, da_chshuf_p=0, da_unsharpmask_p=0,
                 da_grayscale_p=0, da_color_p=0, da_antialias_p=0, da_hflip_only=False,
                 da_no_rotate=False,
                 da_cutmix_p=0, da_mixup_p=0,
                 bicubic_only=False,
                 skip_screentone=False,
                 skip_dot=False,
                 crop_samples=4,
                 deblur=0, resize_blur_p=0.1, resize_blur_range=0.05,
                 resize_step_p=0, resize_no_antialias_p=0,
                 noise_level=-1, style=None,
                 return_no_offset_y=False,
                 additional_data_dir=None,
                 additional_data_dir_p=0.01,
                 training=True,
                 ):
        assert scale_factor in {1, 2, 4, 8}
        assert noise_level in {-1, 0, 1, 2, 3}
        assert style in {None, "art", "photo"}
        exclude_prefixes = []
        if scale_factor == 4:
            exclude_prefixes.append(DOT_SCALE2X_PREFIX)
        elif scale_factor == 2:
            exclude_prefixes.append(DOT_SCALE4X_PREFIX)

        if skip_screentone:
            exclude_prefixes.append(SCREENTONE_PREFIX)
        if skip_dot:
            exclude_prefixes.append(DOT_PREFIX)
        if exclude_prefixes:
            exclude_filter = lambda fn: not any([prefix in fn for prefix in exclude_prefixes])
        else:
            exclude_filter = None

        super().__init__(input_dir, num_samples=num_samples, exclude_filter=exclude_filter,
                         additional_data_dir=additional_data_dir,
                         additional_data_dir_p=additional_data_dir_p)
        self.training = training
        self.style = style
        self.noise_level = noise_level
        self.model_offset = model_offset
        self.return_no_offset_y = return_no_offset_y

        if self.training:
            if noise_level >= 0:
                random_crop_p = 0.3 if self.style == "photo" else 0.07
                jpeg_transform = RandomJPEGNoiseX(style=style, noise_level=noise_level, random_crop_p=random_crop_p)
            else:
                jpeg_transform = TP.Identity()

            if style == "photo" and not da_no_rotate:
                rotate_transform = TP.RandomApply([
                    TP.RandomChoice([
                        TP.RandomSafeRotate(y_scale=scale_factor, angle_min=-45, angle_max=45),
                        TP.RandomSafeRotate(y_scale=scale_factor, angle_min=-11, angle_max=11)
                    ], p=[0.2, 0.8]),
                ], p=0.2)
            else:
                rotate_transform = TP.Identity()

            if style == "photo" and noise_level >= 0:
                photo_noise = RandomPhotoNoiseX(noise_level=noise_level)
                if noise_level == 3:
                    jpeg_transform = T.RandomChoice([
                        jpeg_transform,
                        RandomPhotoNoiseX(noise_level=noise_level, force=True)], p=[0.95, 0.05])
            else:
                photo_noise = TP.Identity()

            antialias = TP.RandomApply([AntialiasX()], p=da_antialias_p)

            if scale_factor > 1:
                if bicubic_only:
                    interpolation = INTERPOLATION_BICUBIC
                else:
                    interpolation = None  # random
                random_downscale_x = RandomDownscaleX(scale_factor=scale_factor,
                                                      interpolation=interpolation,
                                                      blur_shift=deblur,
                                                      resize_blur_p=resize_blur_p,
                                                      resize_blur_range=resize_blur_range,
                                                      resize_step_p=resize_step_p,
                                                      resize_no_antialias_p=resize_no_antialias_p)
                random_downscale_x_nearest = RandomDownscaleX(scale_factor=scale_factor,
                                                              resize_blur_p=0,
                                                              interpolation=INTERPOLATION_NEAREST)
            else:
                random_downscale_x = TP.Identity()
                random_downscale_x_nearest = TP.Identity()

            random_flip = TP.RandomHFlip() if da_hflip_only else TP.RandomFlip()

            # 8(max jpeg shift size) * 2(max jpeg shift count) * scale_factor
            y_min_size = tile_size * scale_factor + (8 * 2 * scale_factor)
            self.gt_transforms = T.Compose([
                T.RandomApply([TS.RandomDownscale(min_size=y_min_size)], p=da_scale_p),
                T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1)],
                              p=da_color_p),
                T.RandomApply([TS.RandomChannelShuffle()], p=da_chshuf_p),
                T.RandomApply([TS.RandomUnsharpMask()], p=da_unsharpmask_p),
                T.RandomApply([RandomOverlay()], p=da_mixup_p),
                # TODO: maybe need to prevent color noise for grayscale
                T.RandomApply([TS.RandomGrayscale()], p=da_grayscale_p),
                T.RandomApply([TS.RandomJPEG(min_quality=92, max_quality=99)], p=da_jpeg_p),
                T.RandomApply([CutMix(mask_min=0.2, mask_max=0.5, rotate_p=0.5, blur_p=0.2)], p=da_cutmix_p),
                TS.SizeCondition(y_min_size * 2, TS.ModCrop(mul=scale_factor), T.RandomCrop(y_min_size * 2)),
            ])
            self.gt_transforms_gen = T.Compose([
                T.RandomApply([RandomOverlay()], p=da_mixup_p),
                T.RandomApply([TS.RandomGrayscale()], p=da_grayscale_p),
                T.RandomInvert(p=0.5),
                T.RandomApply([CutMix(mask_min=0.2, mask_max=0.5, rotate_p=0.5)], p=da_cutmix_p),
                TS.SizeCondition(y_min_size * 2, TS.ModCrop(mul=scale_factor), T.RandomCrop(y_min_size * 2)),
            ])
            self.gt_transforms_nearest = T.Compose([
                T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1)],
                              p=da_color_p),
                T.RandomApply([TS.RandomChannelShuffle()], p=da_chshuf_p),
                T.RandomApply([TS.RandomGrayscale()], p=da_grayscale_p),
            ])
            self.transforms = TP.Compose([
                random_downscale_x,
                photo_noise,
                rotate_transform,
                antialias,
                jpeg_transform,
                random_flip,
                TP.RandomCrop(size=tile_size, y_scale=scale_factor),
            ])
            self.transforms_nearest = TP.Compose([
                random_downscale_x_nearest,
                jpeg_transform,
                TP.RandomCrop(size=tile_size, y_scale=scale_factor),
                random_flip,
            ])
        else:
            self.gt_transforms = TS.Identity()
            self.gt_transforms_gen = TS.Identity()
            self.gt_transforms_nearest = TS.Identity()

            interpolation = INTERPOLATION_BICUBIC
            if scale_factor > 1:
                downscale_x = RandomDownscaleX(scale_factor=scale_factor,
                                               blur_shift=deblur,
                                               interpolation=interpolation,
                                               training=False)
                downscale_x_nearest = RandomDownscaleX(scale_factor=scale_factor,
                                                       interpolation=INTERPOLATION_NEAREST,
                                                       resize_blur_p=0,
                                                       training=False)
            else:
                downscale_x = TP.Identity()
                downscale_x_nearest = TP.Identity()
            y_min_size = tile_size * scale_factor + (8 * 2 * scale_factor)
            self.transforms = TP.Compose([
                TP.CenterCrop(size=y_min_size),
                downscale_x,
            ])
            self.transforms_nearest = TP.Compose([
                downscale_x_nearest,
            ])
            self.x_jpeg_shift = [1, 2, 3, 4, 5, 6, 7] + [0] * (100 - 7)
            self.center_crop = TP.CenterCrop(size=tile_size, y_scale=scale_factor)

    def __getitem__(self, index):
        filename = super().__getitem__(index)
        im, _ = pil_io.load_image_simple(filename, color="rgb")
        if im is None:
            raise RuntimeError(f"Unable to load image: {filename}")
        if self.training:
            if NEAREST_PREFIX in filename and random.random() < 0.9:
                im = self.gt_transforms_nearest(im)
                x, y = self.transforms_nearest(im, im)
            elif (SCREENTONE_PREFIX in filename or DOT_PREFIX in filename):
                im = self.gt_transforms_gen(im)
                x, y = self.transforms(im, im)
            else:
                im = self.gt_transforms(im)
                x, y = self.transforms(im, im)
        else:
            if NEAREST_PREFIX in filename:
                im = self.gt_transforms_nearest(im)
                x, y = self.transforms_nearest(im, im)
            elif (SCREENTONE_PREFIX in filename or DOT_PREFIX in filename):
                im = self.gt_transforms_gen(im)
                x, y = self.transforms(im, im)
            else:
                im = self.gt_transforms(im)
                x, y = self.transforms(im, im)

        if not self.training:
            if self.noise_level >= 0:
                if self.style == "photo":
                    x = add_validation_noise(x, noise_level=self.noise_level, index=index)
                qualities, subsampling = choose_validation_jpeg_quality(
                    index=index, style=self.style, noise_level=self.noise_level)
                for i, quality in enumerate(qualities):
                    x = add_jpeg_noise(x, quality=quality, subsampling=subsampling)
                    if len(qualities) > 1 and i != len(qualities) - 1:
                        x, y = shift_jpeg_block(x, y, self.x_jpeg_shift[index % len(self.x_jpeg_shift)])
            y_org = y
            x, y = self.center_crop(x, y)

        y_org = y
        y = TF.pad(y, [-self.model_offset] * 4)
        if not self.return_no_offset_y:
            return TF.to_tensor(x), TF.to_tensor(y), index
        else:
            return TF.to_tensor(x), TF.to_tensor(y), TF.to_tensor(y_org), index


def _test():
    dataset = Waifu2xDataset("./data/waifu2x/eval",
                             model_offset=36, tile_size=256, scale_factor=2,
                             style="art", noise_level=3)
    print(f"len {len(dataset)}")
    x, y, i = dataset[0]
    print("getitem[0]", x.size, y.size)
    TF.to_pil_image(x).show()
    TF.to_pil_image(y).show()


def _test_photo_noise():
    import cv2
    dataset = Waifu2xDataset("./data/photo/eval",
                             model_offset=36, tile_size=256, scale_factor=2,
                             style="photo", noise_level=3)
    print(f"len {len(dataset)}")
    for x, y, *_ in dataset:
        x = pil_io.to_cv2(pil_io.to_image(x))
        y = pil_io.to_cv2(pil_io.to_image(y))
        cv2.imshow("x", x)
        cv2.imshow("y", y)
        c = cv2.waitKey(0)
        if c in {ord("q"), ord("x")}:
            break


if __name__ == "__main__":
    _test_photo_noise()
