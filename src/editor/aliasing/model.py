import segmentation_models_pytorch as smp
import torch
from torch import load
from constants.aliasing import PATH_WEIGHT

model_aliasing = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1,
    activation=None,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_aliasing.to(device)
