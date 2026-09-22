import warnings
from typing import Any

import torch
import torch.nn as nn

from ptmelt.layers import MELTBatchNorm, MELTBayesianDenseFlipOut, PositionalEncoding


def _get_initializer(initializer: str):
    """Get the initializer function."""

    if not initializer.endswith("_"):
        initializer += "_"

    if initializer == "glorot_uniform_":
        initializer = "xavier_uniform_"

    if hasattr(nn.init, initializer):
        return getattr(nn.init, initializer)
    else:
        raise ValueError(f"Initializer {initializer} not recognized.")


def _get_activation(act_name: str, **kwargs: Any):
    """Get the activation function based on its name."""

    # Do some common activation string mapping
    common_mappings = {
        "relu": "ReLU",
        "leaky_relu": "LeakyReLU",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "softmax": "Softmax",
        "softplus": "Softplus",
        "elu": "ELU",
        "selu": "SELU",
        "gelu": "GELU",
        "swish": "SiLU",
        "linear": "Identity",
    }
    act_name = (
        common_mappings.get(act_name.lower(), act_name) if act_name else "Identity"
    )

    if hasattr(nn, act_name):
        return getattr(nn, act_name)(**kwargs)
    else:
        raise ValueError(f"Activation function {act_name} not recognized.")


class MELTBlock(nn.Module):
    """
    Base class for a MELT block. Provides the building blocks for the MELT architecture.
    Defines the common parameters for the MELT blocks with optional activation, dropout,
    batch normalization, and batch renormalization layers.

    Args:
        input_features (int): Number of input features.
        node_list (List[int]): List of number of nodes in each layer.
        activation (str, optional): Activation function. Defaults to "relu".
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to
                                     False.
        batch_norm_type (str, optional): Type of batch normalization. Defaults to "ema".
        use_batch_renorm (bool, optional): Whether to use batch renormalization.
                                           Defaults to False.
        initializer (str, optional): Weight initializer. Defaults to "glorot_uniform".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_features: int,
        node_list: list[int],
        activation: str | None = "relu",
        dropout: float | None = 0.0,
        batch_norm: bool | None = False,
        batch_norm_type: str | None = "ema",
        use_batch_renorm: bool | None = False,
        initializer: str | None = "glorot_uniform",
        seed: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.node_list = node_list
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.batch_norm_type = batch_norm_type
        self.use_batch_renorm = use_batch_renorm
        self.initializer = initializer
        self.seed = seed

        # Get the initializer function
        self.initializer_fn = _get_initializer(self.initializer)

        # Create layer dictionary
        self.layer_dict = nn.ModuleDict()

        # Number of layers in the block
        self.num_layers = len(self.node_list)

        # Validate dropout value
        if self.dropout is not None:
            assert 0.0 <= self.dropout < 1.0, "Dropout must be in the range [0, 1)."

        # Get the activation layers
        if self.activation:
            self.layer_dict.update(
                {
                    f"activation_{i}": _get_activation(self.activation)
                    for i in range(self.num_layers)
                }
            )

        # Optional dropout layers
        if self.dropout > 0:
            self.layer_dict.update(
                {
                    f"dropout_{i}": nn.Dropout(p=self.dropout)
                    for i in range(self.num_layers)
                }
            )

        # Optional batch normalization layers
        if self.batch_norm:
            if self.batch_norm_type == "pytorch":
                self.layer_dict.update(
                    {
                        f"batch_norm_{i}": nn.BatchNorm1d(
                            num_features=self.node_list[i],
                            affine=True,
                            track_running_stats=True,
                            momentum=1e-2,
                            eps=1e-3,
                        )
                        for i in range(self.num_layers)
                    }
                )
            else:
                self.layer_dict.update(
                    {
                        f"batch_norm_{i}": MELTBatchNorm(
                            num_features=self.node_list[i],
                            affine=True,
                            track_running_stats=True,
                            average_type=self.batch_norm_type,
                            momentum=1e-2,
                            eps=1e-3,
                        )
                        for i in range(self.num_layers)
                    }
                )


class DenseBlock(MELTBlock):
    """
    Dense block for the MELT architecture. The dense block consists of dense layers
    with optional activation, dropout, and batch normalization layers.

    Args:
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Initialize dense layers
        self.layer_dict.update(
            {
                f"dense_{i}": nn.Linear(
                    in_features=(
                        self.input_features if i == 0 else self.node_list[i - 1]
                    ),
                    out_features=self.node_list[i],
                )
                for i in range(self.num_layers)
            }
        )
        # Initialize the weights
        torch.manual_seed(self.seed) if self.seed is not None else None
        [
            self.initializer_fn(self.layer_dict[f"dense_{i}"].weight)
            for i in range(self.num_layers)
        ]

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the dense block."""
        x = inputs

        for i in range(self.num_layers):
            # dense -> batch norm -> activation -> dropout
            x = self.layer_dict[f"dense_{i}"](x)
            x = self.layer_dict[f"batch_norm_{i}"](x) if self.batch_norm else x
            x = self.layer_dict[f"activation_{i}"](x) if self.activation else x
            x = self.layer_dict[f"dropout_{i}"](x) if self.dropout > 0 else x

        return x


class ResidualBlock(MELTBlock):
    """
    Residual block for the MELT architecture. The residual block consists of residual
    connections between dense layers with optional activation, dropout, and batch
    normalization layers. Residual connections are added after every `layers_per_block`
    layers.

    Args:
        layers_per_block (int, optional): Number of layers per residual block. Defaults
                                          to 2.
        pre_activation (bool, optional): Whether to use pre-activation residual blocks.
                                         Defaults to False.
        post_add_activation (bool, optional): Whether to use post-addition activation.
                                              Defaults to False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        layers_per_block: int | None = 2,
        pre_activation: bool | None = False,
        post_add_activation: bool | None = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

        # Warning if the number of layers is not divisible by layers_per_block
        if self.num_layers % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: Number of layers {self.num_layers} is not divisible by "
                f"layers_per_block ({self.layers_per_block}), so the last block will "
                f"have {self.num_layers % self.layers_per_block} layers.",
                stacklevel=2,
            )

        # Initialize dense layers
        self.layer_dict.update(
            {
                f"dense_{i}": nn.Linear(
                    in_features=(
                        self.input_features if i == 0 else self.node_list[i - 1]
                    ),
                    out_features=self.node_list[i],
                )
                for i in range(self.num_layers)
            }
        )
        # Initialize the weights
        torch.manual_seed(self.seed) if self.seed is not None else None
        [
            self.initializer_fn(self.layer_dict[f"dense_{i}"].weight)
            for i in range(self.num_layers)
        ]

        # Optional activation layer after addition
        if self.post_add_activation:
            self.layer_dict.update(
                {
                    f"post_add_act_{i}": _get_activation(self.activation)
                    for i in range(self.num_layers // 2)
                }
            )

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the residual block."""
        x = inputs

        for i in range(self.num_layers):
            y = x

            # dense -> (pre-activation) -> batch norm -> dropout -> (post-activation)
            x = self.layer_dict[f"dense_{i}"](x)
            x = self.layer_dict[f"activation_{i}"](x) if self.pre_activation else x
            x = self.layer_dict[f"batch_norm_{i}"](x) if self.batch_norm else x
            x = self.layer_dict[f"dropout_{i}"](x) if self.dropout > 0 else x
            x = self.layer_dict[f"activation_{i}"](x) if not self.pre_activation else x

            # Add the residual connection when reaching the end of a residual block
            if (i + 1) % self.layers_per_block == 0 or i == self.num_layers - 1:
                x = x + y
                x = (
                    self.layer_dict[f"post_add_act_{i // self.layers_per_block}"](x)
                    if self.post_add_activation
                    else x
                )

        return x


class BayesianBlock(MELTBlock):
    """
    Bayesian block for the MELT architecture using custom Bayesian layers.
    """

    def __init__(
        self,
        num_points,
        perturbation_type="multiplicative",
        seed: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # self.num_points = num_points

        self.perturbation_type = perturbation_type
        self.seed = seed

        # Initialize Bayesian layers
        self.layer_dict.update(
            {
                f"bayesian_{i}": MELTBayesianDenseFlipOut(
                    in_features=(
                        self.input_features if i == 0 else self.node_list[i - 1]
                    ),
                    out_features=self.node_list[i],
                    perturbation_type=self.perturbation_type,
                    seed=self.seed,
                )
                for i in range(self.num_layers)
            }
        )

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the Bayesian block."""
        x = inputs

        for i in range(self.num_layers):
            # bayesian -> batch norm -> activation -> dropout
            x = self.layer_dict[f"bayesian_{i}"](x)
            x = self.layer_dict[f"batch_norm_{i}"](x) if self.batch_norm else x
            x = self.layer_dict[f"activation_{i}"](x) if self.activation else x
            x = self.layer_dict[f"dropout_{i}"](x) if self.dropout > 0 else x

        return x

    def kl_loss(self):
        """Calculate the KL divergence loss for all Bayesian layers."""
        kl_div = 0
        for i in range(self.num_layers):
            kl_div += self.layer_dict[f"bayesian_{i}"]._kl_divergence()
        return kl_div


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block for sequence modeling.

    Args:
        input_features (int): Number of input features per time step.
        model_dim (int): Transformer hidden dimension.
        num_layers (int, optional): Number of encoder layers.
        num_heads (int, optional): Number of attention heads.
        ff_dim (int, optional): Feed-forward hidden dimension.
        activation (str, optional): Transformer activation, supports relu/gelu.
        dropout (float, optional): Dropout used in transformer layers.
        max_seq_len (int, optional): Maximum supported sequence length.
        use_causal_mask (bool, optional): If True, apply a causal mask.
        initializer (str, optional): Weight initializer.
        seed (int, optional): Random seed for initialization.
    """

    def __init__(
        self,
        input_features: int,
        model_dim: int,
        num_layers: int | None = 2,
        num_heads: int | None = 4,
        ff_dim: int | None = None,
        activation: str | None = "relu",
        dropout: float | None = 0.0,
        max_seq_len: int | None = 2048,
        use_causal_mask: bool | None = False,
        initializer: str | None = "glorot_uniform",
        seed: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if num_layers is None or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")

        self.input_features = input_features
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim if ff_dim is not None else 4 * model_dim
        self.activation = activation.lower() if activation else "relu"
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_causal_mask = use_causal_mask
        self.initializer = initializer
        self.seed = seed

        if self.activation not in ["relu", "gelu"]:
            warnings.warn(
                f"Activation '{self.activation}' is not supported by TransformerEncoderLayer; falling back to 'relu'.",
                stacklevel=2,
            )
            self.activation = "relu"

        self.initializer_fn = _get_initializer(self.initializer)

        self.input_projection = (
            nn.Linear(self.input_features, self.model_dim)
            if self.input_features != self.model_dim
            else nn.Identity()
        )
        self.position_encoding = PositionalEncoding(
            d_model=self.model_dim,
            max_len=self.max_seq_len,
            dropout=self.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize linear layer weights with the configured initializer."""
        torch.manual_seed(self.seed) if self.seed is not None else None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                self.initializer_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_padding_mask(
        self, sequence_length: int, lengths: torch.Tensor, device: torch.device
    ):
        """Build key-padding mask where True entries are padding positions."""
        time_index = torch.arange(sequence_length, device=device).unsqueeze(0)
        return time_index >= lengths.unsqueeze(1)

    def _build_causal_mask(self, sequence_length: int, device: torch.device):
        """Build a causal mask where True entries are disallowed positions."""
        return torch.triu(
            torch.ones(
                sequence_length, sequence_length, device=device, dtype=torch.bool
            ),
            diagonal=1,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None):
        """Perform forward pass for a batch-first tensor [B, T, F]."""
        x = self.input_projection(inputs)
        x = self.position_encoding(x)

        sequence_length = x.size(1)
        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._build_padding_mask(
                sequence_length=sequence_length,
                lengths=lengths,
                device=x.device,
            )

        attn_mask = (
            self._build_causal_mask(sequence_length=sequence_length, device=x.device)
            if self.use_causal_mask
            else None
        )

        return self.encoder(
            src=x,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )


class DefaultOutput(nn.Module):
    """
    Default output layer with a single dense layer and optional activation function.

    Args:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        activation (str, optional): Activation function. Defaults to "linear".
        initializer (str, optional): Weight initializer. Defaults to "glorot_uniform".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        activation: str | None = "linear",
        initializer: str | None = "glorot_uniform",
        do_bayesian: bool | None = False,
        seed: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.output_features = output_features
        self.activation = activation
        self.initializer = initializer
        self.seed = seed

        # Get the initializer function
        self.initializer_fn = _get_initializer(self.initializer)

        # Initialize output layer
        if do_bayesian:
            self.output_layer = MELTBayesianDenseFlipOut(
                in_features=self.input_features,
                out_features=self.output_features,
                seed=self.seed,
            )
        else:
            self.output_layer = nn.Linear(
                in_features=self.input_features, out_features=self.output_features
            )
            # Initialize the weights
            torch.manual_seed(self.seed) if self.seed is not None else None
            self.initializer_fn(self.output_layer.weight)

        # Initialize activation layer
        self.activation_layer = _get_activation(self.activation)

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the default output layer."""
        x = self.output_layer(inputs)
        x = self.activation_layer(x)

        return x


class MixtureDensityOutput(nn.Module):
    """
    Output layer for mixture density networks. The output layer consists of three
    dense layers for the mixture coefficients, mean, and log variance of the output
    distribution.

    Args:
        input_features (int): Number of input features.
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output dimensions.
        activation (str, optional): Activation function. Defaults to "linear".
        initializer (str, optional): Weight initializer. Defaults to "glorot_uniform".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_features: int,
        num_mixtures: int,
        num_outputs: int,
        activation: str | None = "linear",
        initializer: str | None = "glorot_uniform",
        seed: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.input_features = input_features
        self.num_mixtures = num_mixtures
        self.num_outputs = num_outputs
        self.activation = activation
        self.initializer = initializer
        self.seed = seed

        # Get the initializer function
        self.initializer_fn = _get_initializer(self.initializer)

        # Initialize output layers
        self.mix_coeffs_layer = nn.Linear(
            in_features=self.input_features, out_features=self.num_mixtures
        )
        self.mean_layer = nn.Linear(
            in_features=self.input_features,
            out_features=self.num_mixtures * self.num_outputs,
        )
        self.log_var_layer = nn.Linear(
            in_features=self.input_features,
            out_features=self.num_mixtures * self.num_outputs,
        )

        # Initialize the weights
        torch.manual_seed(self.seed) if self.seed is not None else None
        self.initializer_fn(self.mix_coeffs_layer.weight)
        self.initializer_fn(self.mean_layer.weight)
        self.initializer_fn(self.log_var_layer.weight)

        # Initialize activation layer
        self.activation_layer = _get_activation(self.activation)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor):
        """Perform the forward pass of the multiple mixture output layer."""
        mix_coeffs = self.mix_coeffs_layer(inputs)
        mix_coeffs = torch.clamp(mix_coeffs, min=-10, max=10)
        mix_coeffs = self.softmax_layer(mix_coeffs)

        mean = self.mean_layer(inputs)
        # TODO: Do we ever want to apply an activation function to the mean?
        # Maybe to the mean, but never to the log variance?
        # mean = self.activation_layer(mean)

        log_var = self.log_var_layer(inputs)

        # return concatenated output
        return torch.cat([mix_coeffs, mean, log_var], dim=-1)
