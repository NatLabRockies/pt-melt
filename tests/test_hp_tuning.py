import pytest
import torch
from ptmelt.models import (
    ArtificialNeuralNetwork,
    BayesianNeuralNetwork,
    RecurrentNeuralNetwork,
    ResidualNeuralNetwork,
    TemporalTransformerNetwork,
    VariationalAutoencoder,
)
from ptmelt.utils.hp_tuning import (
    HPOResult,
    build_model_from_config,
    model_builder,
    run_ray_tune,
)
from torch.utils.data import DataLoader, TensorDataset

BASE_CONFIG = {
    "num_features": 4,
    "num_outputs": 2,
    "width": 4,
    "depth": 2,
    "seed": 1,
}


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        ({"arch_type": "ann"}, ArtificialNeuralNetwork),
        ({"arch_type": "resnet", "layers_per_block": 1}, ResidualNeuralNetwork),
        ({"arch_type": "bnn", "num_points": 1}, BayesianNeuralNetwork),
        ({"arch_type": "rnn"}, RecurrentNeuralNetwork),
        (
            {"arch_type": "temporal_transformer", "num_heads": 2},
            TemporalTransformerNetwork,
        ),
        (
            {
                "arch_type": "vae",
                "latent_dims": 2,
                "encoder_node_list": [4],
                "decoder_node_list": [4],
                "num_mixtures": 1,
            },
            VariationalAutoencoder,
        ),
    ],
)
def test_build_model_from_config_supports_all_architectures(config, expected_type):
    model = build_model_from_config({**BASE_CONFIG, **config})

    assert isinstance(model, expected_type)

    if config["arch_type"] in {"rnn", "temporal_transformer"}:
        output = model(torch.randn(2, 5, BASE_CONFIG["num_features"]))
    elif config["arch_type"] == "vae":
        output = model(torch.randn(2, BASE_CONFIG["num_features"]))
        assert isinstance(output, tuple)
        assert len(output) == 4
        return
    else:
        output = model(torch.randn(2, BASE_CONFIG["num_features"]))

    assert output.shape[0] == 2


def test_model_builder_returns_model_optimizer_and_criterion():
    model, optimizer, criterion = model_builder(
        {
            **BASE_CONFIG,
            "arch_type": "ann",
            "learning_rate": 1e-3,
            "loss_fn": "mse",
        }
    )

    assert isinstance(model, ArtificialNeuralNetwork)
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(criterion, torch.nn.MSELoss)


def test_resnet_builder_rejects_non_shape_preserving_widths():
    with pytest.raises(ValueError, match="width == num_features"):
        build_model_from_config(
            {
                **BASE_CONFIG,
                "arch_type": "resnet",
                "width": BASE_CONFIG["num_features"] + 1,
            }
        )


def test_transformer_builder_rejects_odd_widths():
    with pytest.raises(ValueError, match="even width"):
        build_model_from_config(
            {
                **BASE_CONFIG,
                "arch_type": "temporal_transformer",
                "width": 3,
                "num_heads": 1,
            }
        )


def test_run_ray_tune_smoke(tmp_path):
    ray = pytest.importorskip("ray")
    tune = pytest.importorskip("ray.tune")

    x_data = torch.randn(8, BASE_CONFIG["num_features"])
    y_data = torch.randn(8, BASE_CONFIG["num_outputs"])
    train_dl = DataLoader(TensorDataset(x_data, y_data), batch_size=4)
    val_dl = DataLoader(TensorDataset(x_data, y_data), batch_size=4)

    try:
        result = run_ray_tune(
            train_dl=train_dl,
            val_dl=val_dl,
            base_config={
                **BASE_CONFIG,
                "arch_type": "ann",
                "epochs": 1,
                "learning_rate": 1e-3,
                "loss_fn": "mse",
            },
            search_space={"width": tune.choice([BASE_CONFIG["width"]])},
            metric="val_loss",
            mode="min",
            num_samples=1,
            scheduler=None,
            search_alg="random",
            ray_init_kwargs={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": 1,
            },
            resources={"cpu": 1},
            storage_path=str(tmp_path),
            name="hp_tuning_smoke",
        )
    finally:
        ray.shutdown()

    assert isinstance(result, HPOResult)
    assert result.best_config["arch_type"] == "ann"
    assert result.best_hyperparameters == {"width": BASE_CONFIG["width"]}
    assert result.metric_details["metric"] == "val_loss"
    assert result.trial_history
