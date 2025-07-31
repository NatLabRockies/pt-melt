import warnings
from contextlib import nullcontext
from itertools import groupby
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ptmelt.blocks import (
    BayesianBlock,
    DefaultOutput,
    DenseBlock,
    MixtureDensityOutput,
    ResidualBlock,
)
from ptmelt.losses import MixtureDensityLoss


class MELTModel(nn.Module):
    """
    PT-MELT Base model.

    Args:
        num_features (int): The number of input features.
        num_outputs (int): The number of output units.
        width (int, optional): The width of the hidden layers. Defaults to 32.
        depth (int, optional): The number of hidden layers. Defaults to 2.
        act_fun (str, optional): The activation function to use. Defaults to 'relu'.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        input_dropout (float, optional): The input dropout rate. Defaults to 0.0.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to
                                     False.
        batch_norm_type (str, optional): The type of batch normalization to use.
                                         Defaults to 'ema'.
        use_batch_renorm (bool, optional): Whether to use batch renormalization.
                                           Defaults to False.
        output_activation (str, optional): The activation function for the output layer.
                                           Defaults to None.
        initializer (str, optional): The weight initializer to use. Defaults to
                                     'glorot_uniform'.
        l1_reg (float, optional): The L1 regularization strength. Defaults to 0.0.
        l2_reg (float, optional): The L2 regularization strength. Defaults to 0.0.
        num_mixtures (int, optional): The number of mixture components for MDN. Defaults
                                      to 0.
        node_list (list, optional): The list of nodes per layer to alternately define
                                    layers. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        width: Optional[int] = 32,
        depth: Optional[int] = 2,
        act_fun: Optional[str] = "relu",
        dropout: Optional[float] = 0.0,
        input_dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = False,
        batch_norm_type: Optional[str] = "ema",
        use_batch_renorm: Optional[bool] = False,
        output_activation: Optional[str] = None,
        initializer: Optional[str] = "glorot_uniform",
        l1_reg: Optional[float] = 0.0,
        l2_reg: Optional[float] = 0.0,
        num_mixtures: Optional[int] = 0,
        node_list: Optional[list] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super(MELTModel, self).__init__(**kwargs)

        self.num_features = num_features
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.act_fun = act_fun
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.batch_norm = batch_norm
        self.batch_norm_type = batch_norm_type
        self.use_batch_renorm = use_batch_renorm
        self.output_activation = output_activation
        self.initializer = initializer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.num_mixtures = num_mixtures
        self.node_list = node_list
        self.seed = seed

        # Determine if network should be defined based on depth/width or node_list
        if self.node_list:
            self.num_layers = len(self.node_list)
            self.layer_width = self.node_list
        elif self.depth is None:
            self.num_layers = 0
            self.layer_width = []
        else:
            self.num_layers = self.depth
            self.layer_width = [self.width for i in range(self.depth)]

        # Create list for storing names of sub-layers
        self.sub_layer_names = []

        # Create layer dictionary
        self.layer_dict = nn.ModuleDict()

    def build(self):
        """Build the model."""
        self.initialize_layers()

    def initialize_layers(self):
        """Initialize the layers of the model."""
        self.create_dropout_layers()
        self.create_output_layer()

    def create_dropout_layers(self):
        """Create the dropout layers."""
        if self.input_dropout > 0:
            self.layer_dict.update({"input_dropout": nn.Dropout(p=self.input_dropout)})

    def create_output_layer(self):
        """Create the output layer."""
        if self.num_mixtures > 0:
            self.layer_dict.update(
                {
                    "output": MixtureDensityOutput(
                        input_features=(
                            self.layer_width[-1]
                            if self.num_layers > 0
                            else self.num_features
                        ),
                        num_mixtures=self.num_mixtures,
                        num_outputs=self.num_outputs,
                        activation=self.output_activation,
                        initializer=self.initializer,
                        seed=self.seed,
                    )
                }
            )
            self.sub_layer_names.append("output")

        else:
            self.layer_dict.update(
                {
                    "output": DefaultOutput(
                        input_features=(
                            self.layer_width[-1]
                            if self.num_layers > 0
                            else self.num_features
                        ),
                        output_features=self.num_outputs,
                        activation=self.output_activation,
                        initializer=self.initializer,
                        seed=self.seed,
                    )
                }
            )
            self.sub_layer_names.append("output")

    def compute_jacobian(self, x):
        """Compute the Jacobian of the model with respect to the input."""
        pass

    def l1_regularization(self, lambda_l1: float):
        """
        Compute the L1 regularization term for use in the loss function.

        Args:
            lambda_l1 (float): The L1 regularization strength.
        """
        l1_norm = sum(
            p.abs().sum()
            for name, p in self.named_parameters()
            if p.requires_grad and "weight" in name
        )
        return lambda_l1 * l1_norm

    def l2_regularization(self, lambda_l2: float):
        """
        Compute the L2 regularization term for use in the loss function.

        Args:
            lambda_l2 (float): The L2 regularization strength.
        """
        l2_norm = sum(
            p.pow(2.0).sum()
            for name, p in self.named_parameters()
            if p.requires_grad and "weight" in name
        )
        return 0.5 * lambda_l2 * l2_norm

    def get_loss_fn(
        self, loss: Optional[str] = "mse", reduction: Optional[str] = "mean"
    ):
        """
        Get the loss function for the model. Used in the training loop.

        Args:
            loss (str, optional): The loss function to use. Defaults to 'mse'.
            reduction (str, optional): The reduction method for the loss. Defaults to
                                       'mean'.
        """
        if self.num_mixtures > 0:
            warnings.warn(
                "Mixture Density Networks require the use of the MixtureDensityLoss "
                "class. The loss function will be set to automatically."
            )

            return MixtureDensityLoss(
                num_mixtures=self.num_mixtures, num_outputs=self.num_outputs
            )
        elif loss == "mse":
            return nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Loss function {loss} not recognized.")

    def step(self, dataloader, optimizer, criterion, device="cpu", training=True):
        """
        Perform a single step either in training or validation mode.

        """
        self.train() if training else self.eval()

        # Use torch.no_grad() only if not training
        context_manager = torch.no_grad() if not training else nullcontext()

        running_loss = 0.0
        with context_manager:
            for x_in, y_in in dataloader:
                # Move data to device
                x_in, y_in = x_in.to(device), y_in.to(device)

                # Forward pass
                pred = self(x_in)
                loss = criterion(pred, y_in)

                if training:
                    # Add L1 and L2 regularization if present
                    if self.l1_reg > 0:
                        loss += self.l1_regularization(lambda_l1=self.l1_reg)
                    if self.l2_reg > 0:
                        loss += self.l2_regularization(lambda_l2=self.l2_reg)

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Backward pass
                    loss.backward()
                    # Optimize
                    optimizer.step()

                # Accumulate running loss
                running_loss += loss.item()

        # Normalize loss
        running_loss /= len(dataloader)

        return running_loss

    def fit(
        self,
        train_dl,
        val_dl,
        optimizer,
        criterion,
        num_epochs: int,
        device="cpu",
        verbose=False,
    ):
        """
        Perform the model training loop.

        Args:
            train_dl (DataLoader): The training data loader.
            val_dl (DataLoader): The validation data loader.
            optimizer (Optimizer): The optimizer to use.
            criterion (Loss): The loss function to use.
            num_epochs (int): The number of epochs to train the model.
            device (str, optional): The device to use for training. Defaults to 'cpu'.
            verbose (bool, optional): Whether to print training statistics. Defaults to
                                      False.
        """
        # Move model to device
        self.to(device)

        # Create history dictionary
        if not hasattr(self, "history"):
            self.history = {"loss": [], "val_loss": []}

        for epoch in tqdm(range(num_epochs), disable=not verbose):
            # Perform a training and validation step
            train_loss = self.step(
                train_dl, optimizer, criterion, device=device, training=True
            )
            val_loss = self.step(
                val_dl, optimizer, criterion, device=device, training=False
            )

            # Print statistics
            if (epoch + 1) % 10 == 0 and verbose:
                print(
                    f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

            # Save history
            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)


class ArtificialNeuralNetwork(MELTModel):
    """
    Artificial Neural Network (ANN) model.

    Args:
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        **kwargs,
    ):
        super(ArtificialNeuralNetwork, self).__init__(**kwargs)

    def initialize_layers(self):
        """Initialize the layers of the ANN."""
        super(ArtificialNeuralNetwork, self).initialize_layers()

        # Bulk layers
        self.layer_dict.update(
            {
                "dense_block": DenseBlock(
                    input_features=self.num_features,
                    node_list=self.layer_width,
                    activation=self.act_fun,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    batch_norm_type=self.batch_norm_type,
                    use_batch_renorm=self.use_batch_renorm,
                    initializer=self.initializer,
                    seed=self.seed,
                )
            }
        )
        self.sub_layer_names.append("dense_block")

    def forward(self, inputs: torch.Tensor):
        """
        Perform the forward pass of the ANN.

        Args:
            inputs (torch.Tensor): The input data.
        """
        # Apply input dropout
        x = (
            self.layer_dict["input_dropout"](inputs)
            if self.input_dropout > 0
            else inputs
        )

        # Apply dense block
        x = self.layer_dict["dense_block"](x)

        # Apply the output layer(s) and return
        return self.layer_dict["output"](x)


class ResidualNeuralNetwork(MELTModel):
    """
    Residual Neural Network (ResNet) model.

    Args:
        layers_per_block (int, optional): The number of layers per residual block.
                                          Defaults to 2.
        pre_activation (bool, optional): Whether to use pre-activation. Defaults to
                                         True.
        post_add_activation (bool, optional): Whether to use activation after
                                              addition. Defaults to False.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        layers_per_block: Optional[int] = 2,
        pre_activation: Optional[bool] = True,
        post_add_activation: Optional[bool] = False,
        **kwargs,
    ):
        super(ResidualNeuralNetwork, self).__init__(**kwargs)

        self.layers_per_block = layers_per_block
        self.pre_activation = pre_activation
        self.post_add_activation = post_add_activation

    def build(self):
        """Build the model."""
        if self.depth % self.layers_per_block != 0:
            warnings.warn(
                f"Warning: depth {self.num_layers} is not divisible by "
                f"layers_per_block ({self.layers_per_block}), so the last block will "
                f"have {self.depth % self.layers_per_block} layers."
            )

        self.initialize_layers()
        super(ResidualNeuralNetwork, self).build()

    def initialize_layers(self):
        """Initialize the layers of the ResNet."""
        super(ResidualNeuralNetwork, self).initialize_layers()

        # Create the Residual Block
        self.layer_dict.update(
            {
                "residual_block": ResidualBlock(
                    layers_per_block=self.layers_per_block,
                    pre_activation=self.pre_activation,
                    post_add_activation=self.post_add_activation,
                    input_features=self.num_features,
                    node_list=self.layer_width,
                    activation=self.act_fun,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    batch_norm_type=self.batch_norm_type,
                    use_batch_renorm=self.use_batch_renorm,
                    initializer=self.initializer,
                    seed=self.seed,
                )
            }
        )
        self.sub_layer_names.append("residual_block")

    def forward(self, inputs: torch.Tensor):
        """
        Perform the forward pass of the ResNet.

        Args:
            inputs (torch.Tensor): The input data.
        """
        # Apply input dropout
        x = (
            self.layer_dict["input_dropout"](inputs)
            if self.input_dropout > 0
            else inputs
        )

        # Apply residual block
        x = self.layer_dict["residual_block"](x)

        # Apply the output layer(s) and return
        return self.layer_dict["output"](x)


class BayesianNeuralNetwork(MELTModel):
    """
    Bayesian Neural Network (BNN) model.

    Args:
        num_points (int, optional): Number of Monte Carlo samples. Defaults to 1.
        do_aleatoric (bool, optional): Flag to perform aleatoric output. Defaults to False.
        do_bayesian_output (bool, optional): Flag to perform Bayesian output. Defaults to True.
        aleatoric_scale_factor (float, optional): Scale factor for aleatoric uncertainty. Defaults to 5e-2.
        scale_epsilon (float, optional): Epsilon value for the scale of the aleatoric uncertainty. Defaults to 1e-3.
        bayesian_mask (list, optional): List of booleans to determine which layers are Bayesian and which are Dense. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        num_points: Optional[int] = 1,
        do_aleatoric: Optional[bool] = False,
        do_bayesian_output: Optional[bool] = True,
        aleatoric_scale_factor: Optional[float] = 5e-2,
        scale_epsilon: Optional[float] = 1e-3,
        bayesian_mask: Optional[List[bool]] = None,
        **kwargs,
    ):
        super(BayesianNeuralNetwork, self).__init__(**kwargs)

        self.num_points = num_points
        self.do_aleatoric = do_aleatoric
        self.do_bayesian_output = do_bayesian_output
        self.aleatoric_scale_factor = aleatoric_scale_factor
        self.scale_epsilon = scale_epsilon
        self.bayesian_mask = bayesian_mask

    def create_output_layer(self):
        """Create output layer for the Bayesian Neural Network."""

        if self.num_mixtures > 0:
            self.layer_dict.update(
                {
                    "output": MixtureDensityOutput(
                        input_features=(
                            self.layer_width[-1]
                            if self.num_layers > 0
                            else self.num_features
                        ),
                        num_mixtures=self.num_mixtures,
                        num_outputs=self.num_outputs,
                        activation=self.output_activation,
                        initializer=self.initializer,
                        seed=self.seed,
                    )
                }
            )
            self.sub_layer_names.append("output")
        else:
            self.layer_dict.update(
                {
                    "output": DefaultOutput(
                        input_features=(
                            self.layer_width[-1]
                            if self.num_layers > 0
                            else self.num_features
                        ),
                        output_features=self.num_outputs,
                        activation=self.output_activation,
                        initializer=self.initializer,
                        do_bayesian=self.do_bayesian_output,
                        seed=self.seed,
                    )
                }
            )
            self.sub_layer_names.append("output")

    def build(self):
        """Build the BNN."""
        self.initialize_layers()
        super(BayesianNeuralNetwork, self).build()

    def initialize_layers(self):
        """Initialize the layers of the BNN."""
        super(BayesianNeuralNetwork, self).initialize_layers()

        # Create the Bayesian and Dense blocks based on the mask
        if self.bayesian_mask is None:
            self.num_dense_layers = 0
            self.dense_block = None
            self.bayesian_block = BayesianBlock(
                num_points=self.num_points,
                input_features=self.num_features,
                node_list=self.layer_width,
                activation=self.act_fun,
                dropout=self.dropout,
                batch_norm=self.batch_norm,
                batch_norm_type=self.batch_norm_type,
                use_batch_renorm=self.use_batch_renorm,
                initializer=self.initializer,
                seed=self.seed,
            )
            self.layer_dict.update({"full_bayesian_block": self.bayesian_block})
            self.sub_layer_names.append("full_bayesian_block")
        else:
            self.dense_block = []
            self.bayesian_block = []

            bayes_block_idx = 0
            dense_block_idx = 0

            # Loop through the Bayesian mask and create the blocks
            idx = 0
            for is_bayesian, group in groupby(self.bayesian_mask):
                # Get the group and layer width
                group_list = list(group)
                group_len = len(group_list)
                layer_width = self.layer_width[idx : idx + group_len]
                idx += group_len

                # Create a Bayesian block or Dense block
                if is_bayesian:
                    bayesian_block = BayesianBlock(
                        num_points=self.num_points,
                        input_features=(
                            self.num_features
                            if bayes_block_idx == 0
                            else layer_width[0]
                        ),
                        node_list=layer_width,
                        activation=self.act_fun,
                        dropout=self.dropout,
                        batch_norm=self.batch_norm,
                        batch_norm_type=self.batch_norm_type,
                        use_batch_renorm=self.use_batch_renorm,
                        initializer=self.initializer,
                        seed=self.seed,
                    )
                    self.bayesian_block.append(bayesian_block)
                    self.layer_dict.update(
                        {f"bayesian_block_{bayes_block_idx}": bayesian_block}
                    )
                    self.sub_layer_names.append(f"bayesian_block_{bayes_block_idx}")
                    bayes_block_idx += 1
                else:
                    dense_block = DenseBlock(
                        input_features=(
                            self.num_features
                            if dense_block_idx == 0
                            else layer_width[0]
                        ),
                        node_list=layer_width,
                        activation=self.act_fun,
                        dropout=self.dropout,
                        batch_norm=self.batch_norm,
                        batch_norm_type=self.batch_norm_type,
                        use_batch_renorm=self.use_batch_renorm,
                        initializer=self.initializer,
                        seed=self.seed,
                    )
                    self.dense_block.append(dense_block)
                    self.layer_dict.update(
                        {f"dense_block_{dense_block_idx}": dense_block}
                    )
                    self.sub_layer_names.append(f"dense_block_{dense_block_idx}")
                    dense_block_idx += 1

    def forward(self, inputs: torch.Tensor):
        """
        Perform the forward pass of the BNN.

        Args:
            inputs (torch.Tensor): The input data.
        """
        # Apply input dropout
        x = (
            self.layer_dict["input_dropout"](inputs)
            if self.input_dropout > 0
            else inputs
        )

        # Apply the full Bayesian block if it exists
        if "full_bayesian_block" in self.layer_dict:
            x = self.layer_dict["full_bayesian_block"](x)
        else:
            # Apply each Bayesian and Dense block in sequence
            bayesian_index = 0
            dense_index = 0

            for is_bayesian in self.bayesian_mask:
                if is_bayesian:
                    x = self.layer_dict[f"bayesian_block_{bayesian_index}"](x)
                    bayesian_index += 1
                else:
                    x = self.layer_dict[f"dense_block_{dense_index}"](x)
                    dense_index += 1

        # Apply the output layer(s) and return
        return self.layer_dict["output"](x)

    def step(self, dataloader, optimizer, criterion, device="cpu", training=True):
        """
        Perform a single step either in training or validation mode.

        """
        self.train() if training else self.eval()
        dataset_size = len(dataloader.dataset)

        # Use torch.no_grad() only if not training
        context_manager = torch.no_grad() if not training else nullcontext()

        running_loss = 0.0
        with context_manager:
            for x_in, y_in in dataloader:
                # Move data to device
                x_in, y_in = x_in.to(device), y_in.to(device)

                # Forward pass
                pred = self(x_in)
                loss = criterion(pred, y_in)

                # Add in kl divergence for the Bayesian block
                if "full_bayesian_block" in self.layer_dict:
                    loss += (
                        self.layer_dict["full_bayesian_block"].kl_loss() / dataset_size
                    )
                else:
                    bayesian_index = 0
                    # Add in kl divergence for each Bayesian block
                    for is_bayesian in self.bayesian_mask:
                        if is_bayesian:
                            loss += (
                                self.layer_dict[
                                    f"bayesian_block_{bayesian_index}"
                                ].kl_loss()
                                / dataset_size
                            )
                            bayesian_index += 1

                if training:
                    # Add L1 and L2 regularization if present
                    if self.l1_reg > 0:
                        loss += self.l1_regularization(lambda_l1=self.l1_reg)
                    if self.l2_reg > 0:
                        loss += self.l2_regularization(lambda_l2=self.l2_reg)

                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Backward pass
                    loss.backward()
                    # Optimize
                    optimizer.step()

                # Accumulate running loss
                running_loss += loss.item()

        # Normalize loss
        running_loss /= len(dataloader)

        return running_loss
