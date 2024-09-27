from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .statistics import compute_metrics, compute_rmse, compute_rsquared


def plot_history(
    history,
    metrics: Optional[List[str]] = ["loss"],
    plot_log: Optional[bool] = False,
    savename: Optional[str] = None,
):
    """
    Plot training history for specified metrics and optionally save the plot.

    Args:
        history (dict): Dictionary containing training history.
        metrics (list): List of metrics to plot. Defaults to ["loss"].
        plot_log (bool): Whether to plot the metrics on a log scale. Defaults to False.
        savename (str): Full path to save the plot. Defaults to None.
    """

    if plot_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax2 = None

    # Plot metrics for both training and validation sets
    for metric in metrics:
        ax1.plot(history[metric], label=f"Train {metric}")
        if f"val_{metric}" in history:
            ax1.plot(history[f"val_{metric}"], label=f"Validation {metric}")

        if plot_log:
            ax2.plot(history[metric], label=f"Train {metric}")
            if f"val_{metric}" in history:
                ax2.plot(history[f"val_{metric}"], label=f"Validation {metric}")

    # Set plot labels and legend
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metrics")

    if plot_log:
        ax2.legend()
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metrics")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    fig.tight_layout()

    # Save the plot if a filename is provided, otherwise display it
    if savename:
        plt.savefig(savename)
    else:
        plt.show()


def point_cloud_plot(
    ax,
    y_real,
    y_pred,
    r_squared: Optional[float] = None,
    rmse: Optional[float] = None,
    label: Optional[str] = None,
    marker: Optional[str] = "o",
    color: Optional[str] = "blue",
    text_pos: Optional[tuple] = (0.3, 0.01),
):
    """
    Create a point cloud plot on the given axes.

    Args:
        ax: Matplotlib axes object.
        y_real (array-like): Actual values.
        y_pred (array-like): Predicted values.
        r_squared (float): R-squared value.
        rmse (float): RMSE value.
        label (str, optional): Label for the plot. Defaults to None.
        marker (str, optional): Marker style. Defaults to "o".
        color (str, optional): Marker color. Defaults to "blue".
        text_pos (tuple, optional): Position for the RMSE text annotation (x, y).
                                    Defaults to (0.3, 0.01).
    """
    # Plot the point cloud
    ax.plot(y_real, y_pred, marker=marker, linestyle="None", label=label, color=color)
    ax.plot(y_real, y_real, linestyle="dashed", color="grey")
    # Add text annotation for R-squared and RMSE
    # TODO: Add more metrics to the text annotation similar to the UQ plot
    # TODO: Add ability to change the formatting of the text annotation
    if r_squared is not None and rmse is not None:
        ax.text(
            *text_pos,
            rf"R$^2$ = {r_squared:0.3f}, RMSE = {rmse:0.3f}",
            transform=ax.transAxes,
            color=color,
        )
    ax.legend()
    ax.set_xlabel("truth")
    ax.set_ylabel("prediction")


def plot_predictions(
    pred_train,
    y_train_real,
    pred_val,
    y_val_real,
    pred_test,
    y_test_real,
    output_indices: Optional[List[int]] = None,
    max_targets: Optional[int] = 3,
    savename: Optional[str] = None,
):
    """
    Plot predictions for specified output indices.

    Args:
        pred_train (array-like): Predicted training values.
        y_train_real (array-like): Actual training values.
        pred_val (array-like): Predicted validation values.
        y_val_real (array-like): Actual validation values.
        pred_test (array-like): Predicted test values.
        y_test_real (array-like): Actual test values.
        output_indices (list of int, optional): List of output indices to plot.
                                                Defaults to None.
        max_targets (int, optional): Maximum number of targets to plot. Defaults to 3.
        savename (str, optional): Full path to save the plot image. If None, the plot
                                  will not be saved. Defaults to None.
    """
    if output_indices is None:
        output_indices = list(range(min(max_targets, pred_train.shape[1])))

    # Create a 1x3 subplot for training, validation, and test data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define markers and colors for the point cloud plot
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    colors = plt.cm.tab10.colors

    # Define text positions for the metrics text annotation
    text_positions = [(0.3, i * 0.05 + 0.01) for i in range(len(output_indices))]

    # Plot predictions for each output index
    for i, idx in enumerate(output_indices):
        # Compute R-squared and RMSE for each dataset
        r_sq_train = compute_rsquared(y_train_real[:, idx], pred_train[:, idx])
        rmse_train = compute_rmse(y_train_real[:, idx], pred_train[:, idx])
        r_sq_val = compute_rsquared(y_val_real[:, idx], pred_val[:, idx])
        rmse_val = compute_rmse(y_val_real[:, idx], pred_val[:, idx])
        r_sq_test = compute_rsquared(y_test_real[:, idx], pred_test[:, idx])
        rmse_test = compute_rmse(y_test_real[:, idx], pred_test[:, idx])

        # Create point cloud plot for each dataset
        point_cloud_plot(
            axes[0],
            y_train_real[:, idx],
            pred_train[:, idx],
            r_sq_train,
            rmse_train,
            f"Output {idx}",
            markers[i % len(markers)],
            colors[i % len(colors)],
            text_pos=text_positions[i % len(text_positions)],
        )
        point_cloud_plot(
            axes[1],
            y_val_real[:, idx],
            pred_val[:, idx],
            r_sq_val,
            rmse_val,
            f"Output {idx}",
            markers[i % len(markers)],
            colors[i % len(colors)],
            text_pos=text_positions[i % len(text_positions)],
        )
        point_cloud_plot(
            axes[2],
            y_test_real[:, idx],
            pred_test[:, idx],
            r_sq_test,
            rmse_test,
            f"Output {idx}",
            markers[i % len(markers)],
            colors[i % len(colors)],
            text_pos=text_positions[i % len(text_positions)],
        )

    # Set plot titles
    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot if a filename is provided, otherwise display it
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


def point_cloud_plot_with_uncertainty(
    ax,
    y_real,
    y_pred,
    y_std,
    text_pos: Optional[tuple] = (0.05, 0.95),
    metrics_to_display: Optional[List[str]] = None,
):
    """
    Create a point cloud plot with uncertainty on the given axes.

    Args:
        ax: Matplotlib axes object.
        y_real (array-like): Actual values.
        y_pred (array-like): Predicted values.
        y_std (array-like): Standard deviation of predictions.
        text_pos (tuple, optional): Position for the text annotation (x, y). Defaults to
                                    (0.05, 0.95).
        metrics_to_display (list of str, optional): List of metrics to display in the
                                                    text annotation. If None, all
                                                    metrics in compute_metrics() are
                                                    show. Defaults to None.
    """
    # TODO: Make the metrics_to_display argument more straightforward
    cmap = plt.get_cmap("viridis")
    # TODO: Add in option to normalize the standard deviation predictions
    # pcnorm = plt.Normalize(y_std.min(), y_std.max())
    sc = ax.scatter(
        y_real,
        y_pred,
        c=y_std,
        cmap=cmap,
        # norm=pcnorm,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )

    # Plot perfect prediction line
    min_val = min(np.min(y_real), np.min(y_pred))
    max_val = max(np.max(y_real), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="dashed", color="grey")

    # Compute metrics
    metrics = compute_metrics(
        y_real, y_pred, y_std, metrics_to_compute=metrics_to_display
    )
    textstr = "\n".join([f"{key} = {value:.3f}" for key, value in metrics.items()])

    # Add text annotation for metrics
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    ax.text(
        *text_pos,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Uncertainty (std dev)")


def plot_predictions_with_uncertainty(
    mean_train,
    std_train,
    y_train_real,
    mean_val,
    std_val,
    y_val_real,
    mean_test,
    std_test,
    y_test_real,
    metrics_to_display: Optional[List[str]] = None,
    savename: Optional[str] = None,
):
    """
    Plot predictions with uncertainty for training, validation, and test data.

    Args:
        mean_train, std_train, y_train_real (array-like): Training data.
        mean_val, std_val, y_val_real (array-like): Validation data.
        mean_test, std_test, y_test_real (array-like): Test data.
        metrics_to_display (list of str, optional): List of metrics to display in the
                                                    text annotation. If None, all
                                                    metrics in compute_metrics() are
                                                    show. Defaults to None.
        savename (str, optional): Full path to save the plot image. If None, the plot
                                  will not be saved. Defaults to None.
    """
    # Create a 1x3 subplot for training, validation, and test data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot predictions with uncertainty for each dataset
    datasets = {
        "Train": (mean_train, std_train, y_train_real, axes[0]),
        "Validation": (mean_val, std_val, y_val_real, axes[1]),
        "Test": (mean_test, std_test, y_test_real, axes[2]),
    }

    for dataset_name, (mean, std, y_real, ax) in datasets.items():
        point_cloud_plot_with_uncertainty(
            ax,
            y_real,
            mean,
            std,
            # f"{dataset_name} Data",
            metrics_to_display=metrics_to_display,
        )

    # Set plot titles
    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions with Uncertainty")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot if a filename is provided, otherwise display it
    if savename:
        fig.savefig(savename)
    else:
        plt.show()
