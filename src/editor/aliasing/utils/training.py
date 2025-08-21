import torch
from torch import Tensor, load
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from PIL import Image

from ..model import model_aliasing
from ..datasets import AlphaDataset
from .evaluate import aliasing

import constants.aliasing as const


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    digits = len(str(size))
    model.train()
    for ix, (input, target) in enumerate(dataloader):
        pred: Tensor = model(input)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ix % 10 == 0:
            loss, current = loss.item(), ix * const.BATCH_SIZE + len(input)
            print(
                f"loss: {loss:>4f}  [{current:>{digits}d}/{size:>{digits}d}]")


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for ix, (input, target) in enumerate(dataloader):
            pred = model(input)
            if not isinstance(pred, Tensor):
                return

            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) ==
                        target).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Result: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


def check():
    for weight in const.DIR_WEIGHT.glob("*.pth"):
        model_aliasing.load_state_dict(load(weight, weights_only=True))

        dir_raw = const.DIR_CHECK/"raw"
        for file in dir_raw.iterdir():
            folder_result = const.DIR_CHECK/"result"/f"{file.stem}"
            folder_result.mkdir(parents=True, exist_ok=True)
            im_input = Image.open(file)
            im_eval = aliasing(im_input)
            name_weight = "_".join(weight.stem.split("_")[:2])
            im_eval.save(folder_result/f"{name_weight}.png")


def training():
    dataset_train = AlphaDataset(const.DIR_TRAIN)
    dataset_test = AlphaDataset(const.DIR_TEST)

    loader_train = DataLoader(
        dataset_train, batch_size=const.BATCH_SIZE, shuffle=True
    )

    loader_test = DataLoader(
        dataset_test, batch_size=const.BATCH_SIZE, shuffle=True
    )
    
    model_aliasing.train()
    fn_loss = smp.losses.TverskyLoss(
        mode='multilabel',
        log_loss=False,
        alpha=0.5,
        beta=0.5,
    )

    loss_fn = fn_loss
    optimizer = torch.optim.RAdam(
        model_aliasing.parameters(), lr=const.LEARNING_RATE
    )

    for i in range(const.EPOCH_COUNT):
        print(
            "-------------------------------"
            f"Epoch:{i+1}"
            "-------------------------------"
        )
        train(loader_train, model_aliasing, loss_fn, optimizer)
        rate_loss = test(loader_test, model_aliasing, loss_fn)
        if (i+1) % 10 == 0:
            path_save = const.DIR_WEIGHT / \
                f"model_ep{(i+1)}_loss{rate_loss:>3f}.pth"
            torch.save(model_aliasing.state_dict(), path_save)

    check()

    print("Training Complete!")



if __name__=="__main__":
    training()