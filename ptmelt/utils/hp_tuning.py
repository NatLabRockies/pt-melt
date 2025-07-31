from torch import optim

from ptmelt.models import ArtificialNeuralNetwork, ResidualNeuralNetwork
from ptmelt.nn_utils import get_loss_fn


def model_builder(config):
    """
    Build a model with the given configuration dictionary. Useful for hyperparameter
    tuning.

    Args:
        config (dict): Dictionary containing the configuration of the model.
    """

    # Get the model type from the configuration
    arch_type = config["arch_type"]

    # Get the learning rate from the configuration
    learning_rate = config["learning_rate"]

    # Get the loss function from the configuration
    loss_fn = config["loss_fn"]

    # Get the number of input features and output features from the configuration
    num_features = config["num_features"]
    num_outputs = config["num_outputs"]

    if arch_type == "ann":
        act_fun = config.get("act_fun", "relu")
        dropout = config.get("dropout", 0.0)
        input_dropout = config.get("input_dropout", 0.0)
        batch_norm = config.get("batch_norm", False)
        batch_norm_type = config.get("batch_norm_type", "ema")
        use_batch_renorm = config.get("use_batch_renorm", False)
        output_activation = config.get("output_activation", None)
        initializer = config.get("initializer", "glorot_uniform")
        l1_reg = config.get("l1_reg", 0.0)
        l2_reg = config.get("l2_reg", 0.0)
        num_mixtures = config.get("num_mixtures", 0)

        # Configurations for the model size
        node_list = config.get("node_list", None)
        max_depth = config.get("max_depth", None)
        if node_list is None and max_depth is None:
            width = config.get("width", 32)
            depth = config.get("depth", 2)
        elif node_list is None and max_depth is not None:
            node_list = []
            for i in range(max_depth):
                layer_width = config.get(f"layer_{i}_width", 0)
                if layer_width > 0:
                    node_list.append(layer_width)
            # Remove zero width just in case
            node_list = [x for x in node_list if x > 0]
            width = None
            depth = None
        else:
            width = None
            depth = None

        model = ArtificialNeuralNetwork(
            num_features=num_features,
            num_outputs=num_outputs,
            act_fun=act_fun,
            dropout=dropout,
            input_dropout=input_dropout,
            batch_norm=batch_norm,
            batch_norm_type=batch_norm_type,
            use_batch_renorm=use_batch_renorm,
            output_activation=output_activation,
            initializer=initializer,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            num_mixtures=num_mixtures,
            width=width,
            depth=depth,
            node_list=node_list,
        )
        model.build()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = get_loss_fn(loss_fn)

        return model, optimizer, criterion

    # elif arch_type == "resnet":
    #     pass
    # elif arch_type == "custom":
    #     pass
    # TODO: Add explicit support for other models in HP Tuning builder...
    else:
        raise ValueError(f"Unsupported architecture type {arch_type}")
