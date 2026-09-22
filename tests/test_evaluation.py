import numpy as np
import pytest
import torch
from ptmelt.utils.evaluation import make_predictions
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler


class StaticMDN(torch.nn.Module):
    """Minimal one-component MDN with fixed normalized mean and std."""

    num_mixtures = 1
    num_outputs = 1

    def __init__(self, mean, std):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, inputs):
        row = torch.tensor(
            [
                1.0,
                self.mean,
                np.log(self.std**2),
            ],
            dtype=inputs.dtype,
            device=inputs.device,
        )
        return row.repeat(inputs.shape[0], 1)


def test_standard_scaler_inverse_transforms_mdn_standard_deviation():
    scaler = StandardScaler().fit(
        np.array(
            [
                [0.0],
                [2.0],
                [4.0],
            ]
        )
    )

    normalized_std = 0.5
    model = StaticMDN(mean=0.0, std=normalized_std)

    predictions, std_pred, components = make_predictions(
        model,
        np.zeros((2, 1), dtype=np.float32),
        y_normalizer=scaler,
        unnormalize=True,
        return_components=True,
    )

    expected_mean = scaler.inverse_transform(np.zeros((2, 1)))
    expected_std = np.full_like(
        predictions,
        normalized_std * scaler.scale_,
        dtype=np.float32,
    )

    np.testing.assert_allclose(predictions, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(std_pred, expected_std, rtol=1e-5)
    np.testing.assert_allclose(
        components["mean_prediction"],
        predictions,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        components["std_prediction"],
        std_pred,
        rtol=1e-5,
    )


def test_minmax_scaler_inverse_transforms_mdn_standard_deviation():
    scaler = MinMaxScaler().fit(
        np.array(
            [
                [10.0],
                [20.0],
                [30.0],
            ]
        )
    )

    normalized_std = 0.1
    model = StaticMDN(mean=0.5, std=normalized_std)

    predictions, std_pred = make_predictions(
        model,
        np.zeros((2, 1), dtype=np.float32),
        y_normalizer=scaler,
        unnormalize=True,
    )

    expected_mean = scaler.inverse_transform(np.full((2, 1), 0.5))
    expected_std = np.broadcast_to(normalized_std / scaler.scale_, predictions.shape)

    np.testing.assert_allclose(predictions, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(std_pred, expected_std, rtol=1e-5)


def test_nonlinear_scaler_does_not_fake_inverse_standard_deviation():
    scaler = PowerTransformer().fit(
        np.array(
            [
                [1.0],
                [2.0],
                [4.0],
                [8.0],
            ]
        )
    )

    normalized_std = 0.2
    model = StaticMDN(mean=0.0, std=normalized_std)

    with pytest.warns(
        UserWarning,
        match="Cannot exactly inverse-transform standard deviations",
    ):
        predictions, std_pred = make_predictions(
            model,
            np.zeros((2, 1), dtype=np.float32),
            y_normalizer=scaler,
            unnormalize=True,
        )

    expected_mean = scaler.inverse_transform(np.zeros((2, 1)))

    np.testing.assert_allclose(predictions, expected_mean, rtol=1e-5)
    np.testing.assert_allclose(
        std_pred,
        np.full((2, 1), normalized_std),
        rtol=1e-5,
    )
