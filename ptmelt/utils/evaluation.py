import warnings
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ptmelt.utils.preprocessing import IdentityScaler


def _inverse_transform_std(std_pred, y_normalizer):
    """Inverse-transform standard deviations for supported affine scalers."""
    if isinstance(y_normalizer, IdentityScaler):
        return std_pred

    if isinstance(y_normalizer, MinMaxScaler):
        scale = getattr(y_normalizer, "scale_", None)
        if scale is None:
            return std_pred
        return std_pred / np.asarray(scale, dtype=np.float32)

    if isinstance(y_normalizer, (StandardScaler, RobustScaler)):
        scale = getattr(y_normalizer, "scale_", None)
        if scale is None:
            return std_pred
        return std_pred * np.asarray(scale, dtype=np.float32)

    warnings.warn(
        "Cannot exactly inverse-transform standard deviations for "
        f"{type(y_normalizer).__name__}; standard deviations will remain "
        "in normalized space.",
        stacklevel=2,
    )
    return std_pred


def parse_mixture_density_predictions(pred_array, num_mixtures, num_outputs):
    """
    Parse MDN outputs into mixture components and aggregate predictive moments.

    Args:
        pred_array (torch.Tensor): Raw MDN output tensor with shape
            [batch_size, num_mixtures + 2 * num_mixtures * num_outputs].
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output dimensions.
    """
    end_mixture = num_mixtures
    end_mean = end_mixture + num_mixtures * num_outputs

    mix_coeffs = pred_array[:, :end_mixture]
    mean_preds = pred_array[:, end_mixture:end_mean]
    log_var_preds = pred_array[:, end_mean:]

    mean_preds = mean_preds.view(-1, num_mixtures, num_outputs)
    log_var_preds = torch.clamp(
        log_var_preds.view(-1, num_mixtures, num_outputs), min=-10.0, max=10.0
    )

    mix_coeffs = torch.clamp(mix_coeffs, min=1e-8)
    mix_coeffs = mix_coeffs / mix_coeffs.sum(dim=-1, keepdim=True)
    mix_coeffs_expanded = mix_coeffs.unsqueeze(-1)

    mean_prediction = torch.sum(mean_preds * mix_coeffs_expanded, dim=1)
    variance_prediction = (
        torch.sum(
            (mean_preds**2 + torch.exp(log_var_preds)) * mix_coeffs_expanded,
            dim=1,
        )
        - mean_prediction**2
    )
    variance_prediction = torch.clamp(variance_prediction, min=1e-10)
    std_prediction = torch.sqrt(variance_prediction)

    return {
        "mix_coeffs": mix_coeffs,
        "mean_preds": mean_preds,
        "log_var_preds": log_var_preds,
        "mean_prediction": mean_prediction,
        "std_prediction": std_prediction,
    }


def _prepare_model_inputs(x_data, lengths=None, device=None):
    """Convert numpy arrays to tensors and move tensors to the inference device."""
    if isinstance(x_data, np.ndarray):
        x_data = torch.from_numpy(x_data).float()

    if lengths is not None and isinstance(lengths, np.ndarray):
        lengths = torch.from_numpy(lengths)

    if device is None:
        device = (
            x_data.device if isinstance(x_data, torch.Tensor) else torch.device("cpu")
        )
    else:
        device = torch.device(device)

    x_data = x_data.to(device)
    if lengths is not None:
        lengths = lengths.to(device)

    return x_data, lengths, device


def _forward_model(model, x_data, lengths=None):
    """Run model forward while supporting both plain and sequence-aware models."""
    try:
        return model(x_data, lengths=lengths)
    except TypeError:
        return model(x_data)


def make_predictions(
    model,
    x_data,
    y_normalizer: Any | None = None,
    unnormalize: bool | None = False,
    training: bool | None = False,
    lengths=None,
    device: str | None = None,
    return_components: bool | None = False,
):
    """
    Make predictions using the provided model and optionally unscaling the results.

    Args:
        model (torch.nn.Module): A PyTorch model.
        x_data (np.ndarray or torch.Tensor): Input data.
        y_normalizer (scaler, optional): Scikit-learn like scaler object. Defaults to
                                         None.
        unnormalize (bool, optional): Whether to unnormalize the predictions. Defaults
                                      to False.
        training (bool, optional): Whether to use training mode for making predictions.
                                   Defaults to False.
        lengths (array-like, optional): Optional sequence lengths for variable-length
                                        forecasting models.
        device (str, optional): Device to run inference on. Defaults to input/model
                                device.
        return_components (bool, optional): Whether to also return parsed MDN
                                            components for probabilistic models.
    """
    # Set model to either training or evaluation mode
    model.train() if training else model.eval()

    x_data, lengths, inference_device = _prepare_model_inputs(
        x_data, lengths=lengths, device=device
    )

    model = model.to(inference_device)

    with torch.no_grad():
        pred_array = _forward_model(model, x_data, lengths=lengths)

    if model.num_mixtures > 0:
        mdn_outputs = parse_mixture_density_predictions(
            pred_array=pred_array,
            num_mixtures=model.num_mixtures,
            num_outputs=model.num_outputs,
        )
        predictions = mdn_outputs["mean_prediction"].detach().cpu().numpy()
        std_pred = mdn_outputs["std_prediction"].detach().cpu().numpy()

    else:
        predictions = pred_array.detach().cpu().numpy()
        std_pred = None

    # Unscale the results if required
    if unnormalize and y_normalizer is not None:
        predictions = y_normalizer.inverse_transform(predictions)
        if std_pred is not None:
            std_pred = _inverse_transform_std(std_pred, y_normalizer)
    elif unnormalize and y_normalizer is None:
        raise ValueError("y_normalizer must be provided to unnormalize predictions.")

    if model.num_mixtures > 0:
        if return_components:
            component_outputs = {
                key: value.detach().cpu().numpy() for key, value in mdn_outputs.items()
            }
            if unnormalize and y_normalizer is not None:
                component_outputs["mean_prediction"] = predictions
                component_outputs["std_prediction"] = std_pred
            return predictions, std_pred, component_outputs
        return predictions, std_pred
    else:
        return predictions


def ensemble_predictions(
    model,
    x_data,
    y_normalizer: Any | None = None,
    unnormalize: bool | None = False,
    n_iter: int | None = 100,
    training: bool | None = False,
    lengths=None,
    device: str | None = None,
):
    """
    Make ensemble predictions using the provided model and optionally unscaling the
    results. The ensemble predictions are computed by making multiple predictions and
    calculating the mean and standard deviation of the predictions.

    Args:
        model (torch.nn.Module): A PyTorch model.
        x_data (np.ndarray or torch.Tensor): Input data.
        y_normalizer (scaler, optional): Scikit-learn like scaler object. Defaults to
                                         None.
        unnormalize (bool, optional): Whether to unnormalize the predictions. Defaults
                                      to False.
        n_iter (int, optional): Number of iterations for making predictions. Defaults
                                to 100.
        training (bool, optional): Whether to use training mode for making predictions.
                                   Defaults to False.
        lengths (array-like, optional): Optional sequence lengths for variable-length
                                        forecasting models.
        device (str, optional): Device to run inference on. Defaults to input/model
                                device.
    """
    # Set model to either training or evaluation mode
    model.train() if training else model.eval()

    x_data, lengths, inference_device = _prepare_model_inputs(
        x_data, lengths=lengths, device=device
    )
    model = model.to(inference_device)

    predictions = []
    for _ in range(n_iter):
        with torch.no_grad():
            pred_tensor = _forward_model(model, x_data, lengths=lengths)
        if model.num_mixtures > 0:
            pred = (
                parse_mixture_density_predictions(
                    pred_tensor, model.num_mixtures, model.num_outputs
                )["mean_prediction"]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            pred = pred_tensor.detach().cpu().numpy()
        if unnormalize and y_normalizer is not None:
            pred = y_normalizer.inverse_transform(pred)
        elif unnormalize and y_normalizer is None:
            raise ValueError(
                "y_normalizer must be provided to unnormalize predictions."
            )
        elif not unnormalize and y_normalizer is not None:
            warnings.warn(
                "y_normalizer provided but unnormalize set to False. "
                "Predictions will be in normalized space.",
                stacklevel=2,
            )

        predictions.append(pred)

    predictions = np.array(predictions)
    pred_mean = np.mean(predictions, axis=0)
    pred_std = np.std(predictions, axis=0)

    return pred_mean, pred_std
