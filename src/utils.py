import torch
import torch.nn as nn


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal(m.weight.data, 0.0, 0.02)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model checkpoint

    Args:
        state : State of the model
        filename (str, optional): Name of the file. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """Load model checkpoint from the file

    Args:
        checkpoint: model checkpoint
        model: model which checkpoint will be loaded
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
