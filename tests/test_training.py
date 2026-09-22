import pytest
import torch
from ptmelt.models import ArtificialNeuralNetwork
from ptmelt.utils.hp_tuning import run_ray_tune
from torch.utils.data import DataLoader, TensorDataset


def make_training_setup():
    torch.manual_seed(0)

    x_data = torch.linspace(-1.0, 1.0, 16).reshape(8, 2)
    y_data = x_data.sum(dim=1, keepdim=True)
    dataloader = DataLoader(
        TensorDataset(x_data, y_data),
        batch_size=4,
        shuffle=False,
    )

    model = ArtificialNeuralNetwork(
        num_features=2,
        num_outputs=1,
        width=4,
        depth=1,
        seed=1,
    )
    model.build()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = model.get_loss_fn("mse")

    return model, optimizer, criterion, dataloader


def test_fit_with_validation_records_validation_loss():
    model, optimizer, criterion, dataloader = make_training_setup()

    model.fit(
        dataloader,
        dataloader,
        optimizer,
        criterion,
        num_epochs=2,
        stopping=False,
    )

    assert len(model.history["loss"]) == 2
    assert len(model.history["val_loss"]) == 2
    assert len(model.history["lr"]) == 2
    assert len(model.history["epoch"]) == 2


def test_fit_without_validation_omits_validation_loss():
    model, optimizer, criterion, dataloader = make_training_setup()

    model.fit(
        dataloader,
        None,
        optimizer,
        criterion,
        num_epochs=2,
        stopping=False,
    )

    assert len(model.history["loss"]) == 2
    assert "val_loss" not in model.history
    assert len(model.history["lr"]) == 2
    assert len(model.history["epoch"]) == 2


class RecordingReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.recorded_metrics = []

    def step(self, metrics, epoch=None):
        self.recorded_metrics.append(float(metrics))
        return super().step(metrics, epoch=epoch)


def test_plateau_scheduler_uses_training_loss_without_validation():
    model, optimizer, criterion, dataloader = make_training_setup()
    scheduler = RecordingReduceLROnPlateau(optimizer)

    model.fit(
        dataloader,
        None,
        optimizer,
        criterion,
        num_epochs=2,
        scheduler=scheduler,
        stopping=False,
    )

    assert scheduler.recorded_metrics == pytest.approx(model.history["loss"])


def test_plateau_scheduler_uses_validation_loss_when_available():
    model, optimizer, criterion, dataloader = make_training_setup()
    scheduler = RecordingReduceLROnPlateau(optimizer)

    model.fit(
        dataloader,
        dataloader,
        optimizer,
        criterion,
        num_epochs=2,
        scheduler=scheduler,
        stopping=False,
    )

    assert scheduler.recorded_metrics == pytest.approx(model.history["val_loss"])


def test_ray_tune_requires_validation_data():
    _, _, _, dataloader = make_training_setup()

    with pytest.raises(
        ValueError,
        match="val_dl is required",
    ):
        run_ray_tune(
            train_dl=dataloader,
            val_dl=None,
            search_space={},
        )
