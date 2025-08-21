import sys
import pathlib

# path
DIR_SRC = pathlib.Path(str(sys.modules["__main__"].__file__)).parent
DIR_DATA = DIR_SRC.parent / "training_data"
DIR_TRAIN = DIR_DATA / "training"
DIR_TEST = DIR_DATA / "test"
DIR_CHECK = DIR_DATA / "check"

DIR_WEIGHT = DIR_SRC / "editor/aliasing/pretrained_models"
PATH_WEIGHT = DIR_WEIGHT / "model.pth"

# training
EPOCH_COUNT = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 8
