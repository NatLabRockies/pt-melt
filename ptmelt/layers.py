from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class MELTBayesianLinear(nn.Module):
    """
    Custom Bayesian Linear Layer for PT-MELT.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        """
        Initialize the Bayesian Linear Layer.

        """
        super(MELTBayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Initialize learnable parameters for the posterior
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.weight_log_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4)
        )
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        # Define prior distributions
        self.prior = Normal(0, self.prior_std)
        self.posterior_weight = None
        self.posterior_bias = None

    def forward(self, input: torch.Tensor):
        """
        Perform the forward pass of the Bayesian Linear Layer.

        """
        # Sample epsilon for weights and biases
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)

        # Reparameterization trick
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_epsilon
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_epsilon

        # Flipout Variational Inference
        batch_size = input.size(0)
        input_sign = (
            torch.randint(
                0, 2, (batch_size, 1, self.in_features), device=input.device
            ).float()
            * 2
            - 1
        )
        output_sign = (
            torch.randint(
                0, 2, (batch_size, self.out_features, 1), device=input.device
            ).float()
            * 2
            - 1
        )

        # TODO: This needs to be fixed... the einsum is not correct
        # Apply Flipout perturbations
        perturb_weight = torch.einsum(
            "bi,bo->bio", input_sign.squeeze(1), output_sign.squeeze(1)
        ) * torch.exp(self.weight_log_sigma)
        perturb_bias = output_sign.squeeze(1) * torch.exp(self.bias_log_sigma)

        # Compute output with perturbations
        output = F.linear(
            input, weight + perturb_weight.mean(0), bias + perturb_bias.mean(0)
        )

        # Store posterior distributions for KL divergence
        self.posterior_weight = Normal(self.weight_mu, torch.exp(self.weight_log_sigma))
        self.posterior_bias = Normal(self.bias_mu, torch.exp(self.bias_log_sigma))

        return output

    def _kl_divergence(self):
        kl_w = kl_divergence(self.posterior_weight, self.prior).sum()
        kl_b = kl_divergence(self.posterior_bias, self.prior).sum()
        return kl_w + kl_b


class MELTBatchNorm(nn.Module):
    """
    Custom Batch Normalization Layer for PT-MELT.

    Supports implementation of different types of moving averages for the batch norm
    statistics.

    Args:
        num_features (int): Number of features in the input tensor.
        eps (float, optional): Small value to avoid division by zero.
                               Defaults to 1e-5.
        momentum (float, optional): Momentum for moving average. Defaults to 0.1.
        affine (bool, optional): Apply affine transformation. Defaults to True.
        track_running_stats (bool, optional): Track running statistics. Defaults to
                                              True.
        average_type (str, optional): Type of moving average. Defaults to "ema".
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        average_type: Optional[str] = "ema",
    ):
        # TODO: Check that all features of PyTorch BatchNorm are implemented.

        super(MELTBatchNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.average_type = average_type

        # Initialize Parameters
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Initialize Running Statistics
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: torch.Tensor):
        """
        Perform the forward pass of the batch normalization layer.

        Args:
            input (torch.Tensor): Input tensor to be normalized.
        """
        # Calculate Batch Norm Statistics
        if self.training:
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
        else:
            mean = self.running_mean
            var = self.running_var

        # Update Running Statistics
        if self.track_running_stats and self.average_type == "ema":
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            # self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            # self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)

        elif self.track_running_stats and self.average_type == "simple":
            self.running_mean = mean
            self.running_var = var

        # Normalize
        if self.training:
            input = (input - mean) / (var + self.eps).sqrt()
        else:
            input = (input - self.running_mean) / (self.running_var + self.eps).sqrt()

        # Scale and Shift
        if self.affine:
            input = input * self.weight + self.bias

        return input


class MELTBatchRenorm(MELTBatchNorm):
    """
    Custom Batch Renormalization Layer for PT-MELT.

    Supports implementation of different types of moving averages for the batch norm
    statistics.

    Args:
        num_features (int): Number of features in the input tensor.
        eps (float, optional): Small value to avoid division by zero. Defaults to
                               1e-5.
        momentum (float, optional): Momentum for moving average. Defaults to 0.1.
        affine (bool, optional): Apply affine transformation. Defaults to True.
        track_running_stats (bool, optional): Track running statistics. Defaults to
                                              True.
        average_type (str, optional): Type of moving average. Defaults to "ema".
        rmax (float, optional): Maximum value for r. Defaults to 1.0.
        dmax (float, optional): Maximum value for d. Defaults to 0.0.
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        average_type: Optional[str] = "ema",
        rmax: Optional[float] = 1.0,
        dmax: Optional[float] = 0.0,
    ):
        # TODO: Verify accuracy of renorm implementation.

        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, average_type
        )
        self.register_buffer("rmax", torch.tensor(rmax))
        self.register_buffer("dmax", torch.tensor(dmax))
        self.register_buffer("r", torch.ones(1))
        self.register_buffer("d", torch.zeros(1))

    def forward(self, input: torch.Tensor):
        """Perform the forward pass of the batch renormalization layer."""
        # Calculate Batch Norm Statistics
        if self.training:
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
            std = torch.sqrt(var + self.eps)
            r = std / (self.running_var.sqrt() + self.eps)
            r = torch.clamp(r, 1 / self.rmax, self.rmax)
            d = (mean - self.running_mean) / (self.running_var.sqrt() + self.eps)
            d = torch.clamp(d, -self.dmax, self.dmax)
            self.r = r
            self.d = d
        else:
            mean = self.running_mean
            var = self.running_var

        # Update Running Statistics
        if self.track_running_stats and self.average_type == "ema":
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

        elif self.track_running_stats and self.average_type == "simple":
            self.running_mean = mean
            self.running_var = var

        # Apply Batch Renormalization
        if self.training:
            x_hat = (input - mean) * r / std + d
        else:
            x_hat = (input - self.running_mean) / torch.sqrt(
                self.running_var + self.eps
            )

        return self.weight * x_hat + self.bias
