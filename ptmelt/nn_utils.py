import torch.nn as nn

from ptmelt.losses import MixtureDensityLoss


def get_activation(act_name: str):
    """
    Utility method to get activation based on its name.

    Args:
        act_name (str): Name of the activation function.
    """
    if act_name == "relu":
        return nn.ReLU()
    elif act_name == "leaky_relu":
        return nn.LeakyReLU()
    elif act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "swish":
        return nn.SiLU()
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "linear" or act_name is None:
        return nn.Identity()
    elif act_name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unsupported activation function {act_name}")


def get_initializer(init_name: str):
    """Utility method to get initializer based on its name."""
    if init_name == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif init_name == "glorot_normal":
        return nn.init.xavier_normal_
    elif init_name == "he_uniform":
        return nn.init.kaiming_uniform_
    elif init_name == "he_normal":
        return nn.init.kaiming_normal_
    elif init_name == "normal":
        return nn.init.normal_
    elif init_name == "uniform":
        return nn.init.uniform_
    else:
        raise ValueError(f"Unsupported initializer {init_name}")


def get_loss_fn(loss_name: str):
    """Utility method to get loss function based on its name."""
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "mae":
        return nn.L1Loss()
    elif loss_name == "huber":
        return nn.SmoothL1Loss()
    elif loss_name == "nll":
        return nn.NLLLoss()
    elif loss_name == "poisson":
        return nn.PoissonNLLLoss()
    elif loss_name == "kl_div":
        return nn.KLDivLoss()
    elif loss_name == "mixture_density":
        return MixtureDensityLoss()
    else:
        raise ValueError(f"Unsupported loss function {loss_name}")
