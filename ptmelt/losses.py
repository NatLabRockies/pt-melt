import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_exp(x):
    """Prevents overflow by clipping input range to reasonable values."""
    x = torch.clamp(x, min=-10, max=10)
    return torch.exp(x)


class MixtureDensityLoss(nn.Module):
    """
    Custom loss function for Mixture Density Network (MDN).

    Args:
        num_mixtures (int): Number of mixture components.
        num_outputs (int): Number of output dimensions.
    """

    def __init__(self, num_mixtures, num_outputs, mse_weight=1.0, reduction="mean"):
        super(MixtureDensityLoss, self).__init__()
        self.num_mixtures = num_mixtures
        self.num_outputs = num_outputs
        self.mse_weight = mse_weight

        assert reduction in (
            "mean",
            "sum",
            "none",
        ), "Reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Extract the mixture coefficients, means, and log-variances
        end_mixture = self.num_mixtures
        end_mean = end_mixture + self.num_mixtures * self.num_outputs
        end_log_var = end_mean + self.num_mixtures * self.num_outputs

        # coefficients -> (batch_size, num_mixtures)
        m_coeffs = y_pred[:, :end_mixture]
        # means -> (batch_size, num_mixtures * num_outputs)
        mean_preds = y_pred[:, end_mixture:end_mean]
        # log variances -> (batch_size, num_mixtures * num_outputs)
        log_var_preds = y_pred[:, end_mean:end_log_var]

        # Reshape mean predictions -> (batch_size, num_mixtures, num_outputs)
        mean_preds = mean_preds.view(-1, self.num_mixtures, self.num_outputs)
        # Reshape log variance predictions -> (batch_size, num_mixtures, num_outputs)
        log_var_preds = log_var_preds.view(-1, self.num_mixtures, self.num_outputs)
        log_var_preds = torch.clamp(log_var_preds, min=-10.0, max=10.0)

        # Ensure mixture coefficients sum to 1
        # temperature = 1.0  # lower = sharper, higher = softer
        m_coeffs = F.softmax(m_coeffs, dim=1)
        # Convert log variance to variance
        var_preds = safe_exp(log_var_preds)

        # Difference term -> (batch_size, num_mixtures, num_outputs)
        diff = y_true.unsqueeze(1) - mean_preds
        # # Exponent term -> (batch_size, num_mixtures, num_outputs)
        # exp_term = -0.5 * torch.square(diff) / var_preds

        # Compute log probabilities terms
        const_term = -0.5 * self.num_outputs * torch.log(torch.tensor(2 * torch.pi))
        var_log_term = -0.5 * log_var_preds
        exp_term = -0.5 * torch.square(diff) / torch.clamp(var_preds, min=1e-10)
        log_probs = const_term + var_log_term + exp_term

        # Sum over output dimensions to get log probabilities for each mixture
        # -> (batch_size, num_mixtures)
        log_probs = log_probs.sum(dim=2)

        # Compute mixture weighted log probabilities and add eps to prevent log(0)
        # weighted_log_probs = log_probs + torch.log(m_coeffs + 1e-8)
        weighted_log_probs = log_probs + torch.log(torch.clamp(m_coeffs, min=1e-8))

        # Log-Sum-Exp trick for numerical stability -> (batch_size,)
        log_sum_exp = torch.logsumexp(weighted_log_probs, dim=1)

        # Compute final negative log-likelihood loss -> scalar
        # loss = -torch.mean(log_sum_exp)
        loss = log_sum_exp

        # # add in entropy regularization
        # lambd_reg = 1e-3
        # entropy = -torch.sum(
        #     m_coeffs * torch.log(torch.clamp(m_coeffs, min=1e-8)), dim=1
        # )
        # loss += lambd_reg * entropy

        # add in the mse as well
        if self.mse_weight > 0.0:
            mix_mean = (m_coeffs.unsqueeze(-1) * mean_preds).sum(dim=1)
            mse_loss = F.mse_loss(mix_mean, y_true, reduction="none").mean(dim=-1)
            loss += self.mse_weight * mse_loss

        # Apply reduction to the loss
        if self.reduction == "mean":
            loss = -torch.mean(loss)
        elif self.reduction == "sum":
            loss = -torch.sum(loss)
        # else no reduction, return the full loss tensor

        return loss


class VAELoss(nn.Module):
    def __init__(self, reconstruction_loss_fn=nn.MSELoss()):
        super(VAELoss, self).__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn

    def compute_reconstruction_loss(self, x, x_reconstructed):
        return self.reconstruction_loss_fn(x_reconstructed, x)

    def compute_kl_divergence(self, mix_coeffs, means, log_vars):
        kl_div = -0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp(), dim=-1)
        kl_div = torch.mean(torch.sum(mix_coeffs * kl_div, dim=-1))
        return kl_div

    def forward(self, x, x_reconstructed, mix_coeffs, means, log_vars):
        # Reconstruction loss
        # reconstruction_loss = self.reconstruction_loss_fn(x_reconstructed, x)
        reconstruction_loss = self.compute_reconstruction_loss(x, x_reconstructed)

        # KL Divergence for Gaussian Mixtures
        # kl_div = -0.5 * torch.sum(1 + log_vars - means.pow(2) - log_vars.exp(), dim=-1)
        # kl_div = torch.mean(torch.sum(mix_coeffs * kl_div, dim=-1))
        kl_div = self.compute_kl_divergence(mix_coeffs, means, log_vars)

        return reconstruction_loss + kl_div
