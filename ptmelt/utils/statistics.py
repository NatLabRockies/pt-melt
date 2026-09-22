import numpy as np
from scipy.stats import norm

# TODO: Extract statistics and other utils from ptmelt and tfmelt into a common library


def compute_rsquared(truth, pred):
    """
    Compute the coefficient of determination (:math:`R^2`).

    The :math:`R^2` value is calculated as:

    .. math:: R^2 = 1 - \\frac{\\text{RSS}}{\\text{TSS}}

    where RSS is the residual sum of squares and TSS is the total sum of squares.

    .. math:: \\text{RSS} = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    .. math:: \\text{TSS} = \\sum_{i=1}^{n} (y_i - \\bar{y})^2

    where :math:`y_i` are the true values, :math:`\\hat{y}_i` are the predicted values,
    and :math:`\\bar{y}` is the mean of the true values.

    Args:
        truth (array-like): Truth values from data.
        pred (array-like): Predicted values from model.

    Returns:
        float: :math:`R^2` value.
    """
    rss = np.sum((pred - truth) ** 2)
    tss = np.sum((truth - np.mean(truth)) ** 2)
    r_sq = 1 - rss / tss
    return r_sq


def compute_rmse(truth, pred):
    """
    Compute the root mean squared error (RMSE).

    The RMSE is calculated as:

    .. math:: RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}

    where :math:`y_i` are the true values and :math:`\\hat{y}_i` are the predicted
    values.

    Args:
        truth (array-like): Truth values from data.
        pred (array-like): Predicted values from model.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((truth - pred) ** 2))


def compute_normalized_rmse(truth, pred):
    """
    Compute the normalized root mean squared error (NRMSE).

    The NRMSE is calculated as:

    .. math:: NRMSE = \\frac{\\text{RMSE}}{y_{\\text{max}} - y_{\\text{min}}}

    where RMSE is the root mean squared error, and :math:`y_{\\text{max}}` and
    :math:`y_{\\text{min}}` are the maximum and minimum values of the true values.

    Args:
        truth (array-like): Truth values from data.
        pred (array-like): Predicted values from model.

    Returns:
        float: NRMSE value.
    """
    rmse = compute_rmse(truth, pred)
    return rmse / (truth.max() - truth.min())


def compute_picp(truth, mean, std):
    """
    Compute the prediction interval coverage probability (PICP).

    The PICP is calculated as the proportion of true values that fall within the
    95% prediction interval. The prediction interval (PI) is defined as:

    .. math:: \\text{PI} = [\\mu - 1.96 \\times \\sigma, \\mu + 1.96
              \\times \\sigma]

    The PICP is calculated as:

    .. math:: \\text{PICP} = \\frac{1}{n} \\sum_{i=1}^{n} I(y_i \\in \\text{PI}_i)

    where :math:`y_i` are the true values, :math:`\\mu` is the predicted mean,
    :math:`\\sigma` is the predicted standard deviation, :math:`\\text{PI}_i` is the
    prediction interval for the :math:`i`-th sample, and :math:`I(\\cdot)` is the
    indicator function.

    Args:
        truth (array-like): Truth values from data.
        mean (array-like): Predicted means from model.
        std (array-like): Predicted standard deviations from model.

    Returns:
        float: PICP value.
    """
    lower_bound = mean - 1.96 * std
    upper_bound = mean + 1.96 * std
    # Compute probability as the mean of boolean array
    coverage = np.mean((truth >= lower_bound) & (truth <= upper_bound))
    return coverage


def compute_mpiw(std):
    """
    Compute the mean prediction interval width (MPIW).

    The MPIW is calculated as the average width of the 95% prediction interval. The
    MPIW is defined as:

    .. math:: \\text{MPIW} = \\frac{1}{n} \\sum_{i=1}^{n} (2 \\times 1.96 \\times
              \\sigma_i)

    where :math:`\\sigma_i` is the standard deviation of the :math:`i`-th sample.

    Args:
        std (array-like): Predicted standard deviations from model.

    Returns:
        float: MPIW value.
    """
    return np.mean(2 * 1.96 * std)


def pdf_based_nll(y_true, y_pred, std):
    """
    Compute the PDF-based negative log-likelihood (NLL).

    The PDF-based NLL is calculated as the negative log-likelihood of the true values
    given the predicted means and standard deviations. The PDF-based NLL is defined as:

    .. math:: \\text{NLL} = -\\frac{1}{n} \\sum_{i=1}^{n} \\log(\\mathcal{N}(y_i |
              \\mu_i, \\sigma_i^2))

    where :math:`y_i` are the true values, :math:`\\mathcal{N}` is the normal
    distribution, :math:`\\mu_i` are the predicted means, and :math:`\\sigma_i` are the
    predicted standard deviations.

    Args:
        y_true (array-like): Actual values from data.
        y_pred (array-like): Predicted means from model.
        std (array-like): Predicted standard deviations from model.

    Returns:
        float: PDF-based negative log-likelihood.
    """
    return -np.mean(norm.logpdf(y_true, y_pred, std))


def compute_crps(truth, mean, std):
    """
    Compute the continuous ranked probability score (CRPS) assuming a Gaussian
    distribution.

    The CRPS is calculated as the mean of the continuous ranked probability score for
    each sample. For a Gaussian distribution, the CRPS is defined as:

    .. math:: \\text{CRPS} = \\frac{1}{n} \\sum_{i=1}^{n} \\left\\{\\sigma_i \\left[z_i
              \\left(2\\Phi(z_i) - 1\\right) + 2 \\phi(z_i) -
              \\pi^{-1/2} \\right]\\right\\}

    where :math:`y_i` are the true values, :math:`\\mu_i` are the predicted means,
    :math:`\\sigma_i` are the predicted standard deviations, :math:`z_i` is the z-score,
    :math:`\\Phi` is the cumulative distribution function of the standard normal
    distribution, and :math:`\\phi` is the probability density function of the standard
    normal distribution.

    Args:
        truth (array-like): Actual values from data.
        mean (array-like): Predicted means from model.
        std (array-like): Predicted standard deviations from model.

    Returns:
        float: CRPS value.
    """
    # TODO: Provide some discussion around when to use CRPS versus NLL...

    # Define the CRPS for a Gaussian distribution
    def crps_gaussian(y, mu, sigma):
        z = (y - mu) / sigma
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        return sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))

    # Compute the CRPS for each sample and take the mean
    crps = np.mean(
        [crps_gaussian(t, m, s) for t, m, s in zip(truth, mean, std, strict=True)]
    )

    return crps


def compute_cwc(
    truth,
    mean,
    std,
    alpha: float | None = 0.95,
    gamma: float | None = 1.0,
    penalty_type: str | None = "linear",
):
    """
    Compute the coverage width criterion (CWC).

    The CWC combines the mean prediction interval width (MPIW) and the prediction
    interval coverage probability (PICP) to penalize models that have poor coverage. The
    CWC can use different penalty types. The CWC is defined as:

    .. math:: \\text{CWC} = \\text{MPIW} \\times \\left(1 + \\text{penalty}\\right)

    where the penalty is defined as:

    .. math:: \\text{penalty} = \\max\\left(0, \\gamma \\times (\\alpha - \\text{PICP})
              \\right)

    for a linear penalty, and:

    .. math:: \\text{penalty} = \\gamma \\times \\exp\\left(\\alpha - \\text{PICP}
              \\right) - 1

    for an exponential penalty. Where :math:`\\alpha` is the desired coverage
    probability, :math:`\\gamma` is the penalty weight, :math:`\\text{MPIW}` is the mean
    prediction interval width, and :math:`\\text{PICP}` is the prediction interval
    coverage probability.

    Args:
        truth (array-like): Actual values from data.
        mean (array-like): Predicted means from model.
        std (array-like): Predicted standard deviations from model.
        alpha (float, optional): Desired coverage probability. Defaults to 0.95.
        gamma (float, optional): Penalty weight. Defaults to 1.0.
        penalty_type (str, optional): Type of penalty ('linear', 'exponential').
                                      Defaults to 'linear'.

    Returns:
        float: CWC value.
    """
    # TODO: Maybe have a generalized option to use MPIW or PINAW?
    # Compute the penalty based on the type
    if penalty_type == "linear":
        penalty = gamma * max(0, alpha - compute_picp(truth, mean, std))
    elif penalty_type == "exponential":
        penalty = gamma * np.exp(alpha - compute_picp(truth, mean, std)) - 1

    # Compute the MPIW
    mpiw = compute_mpiw(std)
    # Return the CWC
    return mpiw * (1 + penalty)


def compute_pinaw(truth, std):
    """
    Compute the prediction interval normalized average width (PINAW).

    The PINAW is calculated as the mean prediction interval width normalized by the
    range of the true values. The PINAW is defined as:

    .. math:: \\text{PINAW} = \\frac{\\text{MPIW}}{y_{\\text{max}} - y_{\\text{min}}}

    where :math:`\\text{MPIW}` is the mean prediction interval width, and
    :math:`y_{\\text{max}}` and :math:`y_{\\text{min}}` are the maximum and minimum
    values of the true values.

    Args:
        truth (array-like): Actual values from data.
        std (array-like): Predicted standard deviations from model.

    Returns:
        float: PINAW value.
    """
    return compute_mpiw(std) / (truth.max() - truth.min())


def compute_winkler_score(truth, mean, std, alpha: float | None = 0.05):
    """
    Compute the Winkler score.

    The Winkler score evaluates the quality of prediction intervals by considering both
    the width of the intervals and whether the true value falls within the interval.
    Here, we assume a gaussian distribution with :math:`\\alpha` equal to the
    significance level for the prediction interval. The Winkler score is defined as:

    .. math::
        \\mathcal{W_i} = \\begin{cases}
        \\text{width} + \\frac{2}{\\alpha} (\\mathcal{L} - y_i), & \\text{if } y_i <
        \\mathcal{L} \\\\
        \\text{width} + \\frac{2}{\\alpha} (y_i - \\mathcal{U}), & \\text{if } y_i >
        \\mathcal{U} \\\\
        \\text{width}, & \\text{otherwise}
        \\end{cases}

    where :math:`\\mathcal{L}` and :math:`\\mathcal{U}` are the lower and upper bounds
    of the prediction interval, respectively, :math:`\\text{width}` is the width of the
    prediction interval, and :math:`y_i` is the true value.

    Args:
        truth (array-like): Actual values from data.
        mean (array-like): Predicted means from model.
        std (array-like): Predicted standard deviations from model.
        alpha (float, optional): Significance level for the prediction interval.
                                 Defaults to 0.05.

    Returns:
        float: Winkler score.
    """
    lower_bound = mean - norm.ppf(1 - alpha / 2) * std
    upper_bound = mean + norm.ppf(1 - alpha / 2) * std

    # Initialize the score with the width of the prediction interval
    score = upper_bound - lower_bound

    # Penalize the score if the true value is outside the prediction interval
    below_lower = truth < lower_bound
    above_upper = truth > upper_bound

    score[below_lower] += 2 / alpha * (lower_bound[below_lower] - truth[below_lower])
    score[above_upper] += 2 / alpha * (truth[above_upper] - upper_bound[above_upper])

    # Return the mean score
    return np.mean(score)


def compute_metrics(y_real, y_pred, y_std, metrics_to_compute: list | None = None):
    """
    Compute various metrics between real and predicted values.

    The function computes the following metrics based on an optional input list:

    - :math:`R^2`: Coefficient of determination.
    - RMSE: Root mean squared error.
    - NRMSE: Normalized root mean squared error.
    - PICP: Prediction interval coverage probability.
    - MPIW: Mean prediction interval width.
    - NLL: Negative log-likelihood.
    - CRPS: Continuous ranked probability score.
    - CWC: Coverage width criterion.
    - PINAW: Prediction interval normalized average width.
    - Winkler Score: Winkler score.

    Args:
        y_real (array-like): Actual values.
        y_pred (array-like): Predicted values.
        y_std (array-like): Standard deviation of predictions.
        metrics_to_compute (list, optional): List of metrics to compute.
                                             If None, all metrics are computed.
                                             Defaults to None.

    Returns:
        dict: Dictionary of computed metrics.
    """
    all_metrics = {
        "R^2": (compute_rsquared, (y_real, y_pred)),
        "RMSE": (compute_rmse, (y_real, y_pred)),
        "NRMSE": (compute_normalized_rmse, (y_real, y_pred)),
        "PICP": (compute_picp, (y_real, y_pred, y_std)),
        "MPIW": (compute_mpiw, (y_std,)),
        "NLL": (pdf_based_nll, (y_real, y_pred, y_std)),
        "CRPS": (compute_crps, (y_real, y_pred, y_std)),
        "CWC": (compute_cwc, (y_real, y_pred, y_std)),
        "PINAW": (compute_pinaw, (y_real, y_std)),
        "Winkler Score": (compute_winkler_score, (y_real, y_pred, y_std)),
    }

    if metrics_to_compute is None:
        metrics_to_compute = all_metrics.keys()

    metrics = {}
    for metric in metrics_to_compute:
        if metric in all_metrics:
            func, args = all_metrics[metric]
            metrics[metric] = func(*args)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return metrics
