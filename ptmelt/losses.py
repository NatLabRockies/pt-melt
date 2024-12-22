import torch
import torch.nn as nn


def safe_exp(x):
    """Prevents overflow by clipping input range to reasonable values."""
    x = torch.clamp(x, min=-20, max=20)
    return torch.exp(x)


class MixtureDensityLoss(nn.Module):
    """
    Custom loss function for a Gaussian mixture model.

    Args:
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output dimensions.
    """

    def __init__(self, num_mixtures, num_outputs):
        super(MixtureDensityLoss, self).__init__()
        self.num_mixtures = num_mixtures
        self.num_outputs = num_outputs

    def forward(self, y_pred, y_true):
        # NOTE: the order of the parameters is reversed compared to Keras and TensorFlow
        # Extract the mixture coefficients, means, and log-variances
        end_mixture = self.num_mixtures
        end_mean = end_mixture + self.num_mixtures * self.num_outputs
        end_log_var = end_mean + self.num_mixtures * self.num_outputs

        m_coeffs = y_pred[:, :end_mixture]
        mean_preds = y_pred[:, end_mixture:end_mean]
        log_var_preds = y_pred[:, end_mean:end_log_var]

        # Reshape to ensure same shape as y_true replicated across mixtures
        mean_preds = mean_preds.view(-1, self.num_mixtures, self.num_outputs)
        log_var_preds = log_var_preds.view(-1, self.num_mixtures, self.num_outputs)

        # Calculate the Gaussian probability density function for each component
        const_term = -0.5 * self.num_outputs * torch.log(torch.tensor(2 * torch.pi))
        inv_sigma_log = -0.5 * log_var_preds
        exp_term = (
            -0.5
            * torch.square(y_true.unsqueeze(1) - mean_preds)
            / safe_exp(log_var_preds)
        )

        # form the log probabilities
        log_probs = const_term + inv_sigma_log + exp_term

        # Calculate the log likelihood
        weighted_log_probs = log_probs + torch.log(m_coeffs.unsqueeze(-1))
        log_sum_exp = torch.logsumexp(weighted_log_probs, dim=1)

        # Compute the log likelihood loss
        log_likelihood = torch.mean(log_sum_exp)

        # Return the negative log likelihood
        return -log_likelihood


class VAELoss(nn.Module):
    def __init__(self, reconstruction_loss_fn=nn.MSELoss()):
        super(VAELoss, self).__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn

    def forward(self, x, x_reconstructed, mix_coeffs, means, log_vars):
        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(x_reconstructed, x)

        # KL Divergence for Gaussian Mixtures
        kl_div = -0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp(), dim=-1)
        kl_div = torch.mean(torch.sum(mix_coeffs * kl_div, dim=-1))

        return reconstruction_loss + kl_div
