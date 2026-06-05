from typing import Any, Optional

from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class IdentityScaler:
    """
    A scaler that performs no scaling, behaving like an identity function.

    This class is useful for pipelines where a scaler is optional, but the pipeline
    expects a fit and transform method to be present like in Scikit-learn.
    """

    def __init__(self, **kwargs):
        self.scale_ = 1.0

    def fit(self, X, y: Optional[Any] = None):
        """
        Dummy fit method that does nothing.

        Args:
            X (array-like): Input data.
            y (array-like): Ignored.
        """
        return self

    def transform(self, X):
        """
        Dummy transform method that returns the input data unchanged.

        Args:
            X (array-like): Input data.
        """
        return X

    def fit_transform(self, X, y: Optional[Any] = None):
        """
        Dummy fit_transform method that returns the input data unchanged.

        Args:
            X (array-like): Input data.
            y (array-like): Ignored.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Dummy inverse_transform method that returns the input data unchanged.

        Args:
            X (array-like): Input data.
        """
        return X

    def get_params(self, deep: bool = True):
        """
        Get empty parameters. Needed to be compatible with Scikit-learn.

        Args:
            deep (bool): If True, will return the parameters for this scaler and
            contained sub-objects that are estimators.
        """
        return {}


def get_normalizers(
    norm_type: Optional[str] = "standard", n_normalizers: Optional[int] = 1, **kwargs
):
    """
    Get a list of normalizers based on the specified normalization type and number of
    normalizers.

    Args:
        norm_type (str, optional): Type of normalization ('standard', 'minmax',
        'robust', 'power', 'quantile'). Defaults to 'standard'.
        n_normalizers (int, optional): Number of normalizers to create. Defaults to 1.
        **kwargs: Additional keyword arguments for the specific scaler.

    Returns:
        list: A list of normalizers.
    """
    # Supported normalization types
    normalizers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
        "none": IdentityScaler,
    }

    # Check if the normalization type is supported
    if norm_type not in normalizers:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    scaler_class = normalizers[norm_type]

    # Extract relevant supported kwargs for each scaler
    scaler_params = {
        "minmax": ["feature_range"],
        "quantile": ["output_distribution", "n_quantiles", "random_state"],
        "power": ["method", "standardize"],
    }

    relevant_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in scaler_params.get(norm_type, [])
    }

    # Create the specified number of normalizers
    normalizers_list = [scaler_class(**relevant_kwargs) for _ in range(n_normalizers)]

    # Return the list of normalizers
    return normalizers_list
