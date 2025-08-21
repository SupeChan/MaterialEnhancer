import torch
from torch import load
from ..model import model_aliasing

from constants.aliasing import PATH_WEIGHT

def load_weight():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_aliasing.load_state_dict(
        load(PATH_WEIGHT, weights_only=True, map_location=device)
    )