import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ptmelt.models import (
    ArtificialNeuralNetwork,
    BayesianNeuralNetwork,
    RecurrentNeuralNetwork,
    ResidualNeuralNetwork,
    TemporalTransformerNetwork,
    VariationalAutoencoder,
)

SUPPORTED_ARCHITECTURES = {
    "ann",
    "resnet",
    "bnn",
    "rnn",
    "temporal_transformer",
    "vae",
}

_ARCHITECTURE_ALIASES = {
    "ann": "ann",
    "artificial_neural_network": "ann",
    "artificialneuralnetwork": "ann",
    "dense": "ann",
    "resnet": "resnet",
    "residual": "resnet",
    "residual_neural_network": "resnet",
    "residualneuralnetwork": "resnet",
    "bnn": "bnn",
    "bayesian": "bnn",
    "bayesian_neural_network": "bnn",
    "bayesianneuralnetwork": "bnn",
    "rnn": "rnn",
    "recurrent": "rnn",
    "recurrent_neural_network": "rnn",
    "recurrentneuralnetwork": "rnn",
    "temporal_transformer": "temporal_transformer",
    "temporaltransformer": "temporal_transformer",
    "temporal_transformer_network": "temporal_transformer",
    "temporaltransformernetwork": "temporal_transformer",
    "transformer": "temporal_transformer",
    "vae": "vae",
    "variational_autoencoder": "vae",
    "variationalautoencoder": "vae",
}

_COMMON_MODEL_KEYS = (
    "num_features",
    "num_outputs",
    "width",
    "depth",
    "act_fun",
    "dropout",
    "input_dropout",
    "batch_norm",
    "batch_norm_type",
    "use_batch_renorm",
    "output_activation",
    "initializer",
    "l1_reg",
    "l2_reg",
    "num_mixtures",
    "node_list",
    "seed",
)


@dataclass
class HPOResult:
    """Structured return value from :func:`run_ray_tune`."""

    best_config: dict[str, Any]
    best_hyperparameters: dict[str, Any]
    metric_details: dict[str, Any]
    trial_history: dict[str, Any]
    results: Any | None = None


def _require_config(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key '{key}'.")
    return config[key]


def _normalize_architecture(arch_type: Any) -> str:
    arch_key = str(arch_type).strip().lower().replace("-", "_").replace(" ", "_")
    arch_key = _ARCHITECTURE_ALIASES.get(arch_key, arch_key)
    if arch_key not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(
            f"Unsupported architecture type {arch_type}. Supported: {supported}"
        )
    return arch_key


def _layer_widths_from_config(
    config: Mapping[str, Any], prefix: str = "layer", max_depth_key: str = "max_depth"
) -> list | None:
    max_depth = config.get(max_depth_key)
    if max_depth is None:
        return None

    node_list = []
    for layer_index in range(max_depth):
        layer_width = config.get(f"{prefix}_{layer_index}_width", 0)
        if layer_width > 0:
            node_list.append(layer_width)
    return node_list


def _common_model_kwargs(
    config: Mapping[str, Any], use_sampled_node_list: bool = True
) -> dict[str, Any]:
    for key in ("num_features", "num_outputs"):
        _require_config(config, key)

    kwargs = {key: config[key] for key in _COMMON_MODEL_KEYS if key in config}
    kwargs.setdefault("act_fun", "relu")
    kwargs.setdefault("dropout", 0.0)
    kwargs.setdefault("input_dropout", 0.0)
    kwargs.setdefault("batch_norm", False)
    kwargs.setdefault("batch_norm_type", "ema")
    kwargs.setdefault("use_batch_renorm", False)
    kwargs.setdefault("output_activation", None)
    kwargs.setdefault("initializer", "glorot_uniform")
    kwargs.setdefault("l1_reg", 0.0)
    kwargs.setdefault("l2_reg", 0.0)
    kwargs.setdefault("num_mixtures", 0)

    if "node_list" in config:
        kwargs["node_list"] = config["node_list"]
        kwargs["width"] = None
        kwargs["depth"] = None
    elif use_sampled_node_list:
        sampled_node_list = _layer_widths_from_config(config)
        if sampled_node_list is not None:
            kwargs["node_list"] = sampled_node_list
            kwargs["width"] = None
            kwargs["depth"] = None
        else:
            kwargs.setdefault("width", 32)
            kwargs.setdefault("depth", 2)
    else:
        kwargs.pop("node_list", None)
        kwargs.setdefault("width", 32)
        kwargs.setdefault("depth", 2)

    return kwargs


def _vae_node_list(config: Mapping[str, Any], key: str, prefix: str) -> list:
    if key in config:
        return config[key]

    node_list = _layer_widths_from_config(
        config, prefix=prefix, max_depth_key=f"{prefix}_max_depth"
    )
    if node_list is not None:
        return node_list

    raise KeyError(f"Missing required config key '{key}'.")


def _validate_resnet_widths(kwargs: Mapping[str, Any], layers_per_block: int) -> None:
    node_list = kwargs.get("node_list")
    if node_list:
        block_input_width = kwargs["num_features"]
        for block_start in range(0, len(node_list), layers_per_block):
            block_widths = node_list[block_start : block_start + layers_per_block]
            if block_widths[-1] != block_input_width:
                raise ValueError(
                    "ResidualNeuralNetwork requires each residual block output width "
                    "to match that block's input width. Use width == num_features "
                    "for uniform ResNets or choose node_list block endpoints that "
                    "preserve residual-add dimensions."
                )
            block_input_width = block_widths[-1]
        return

    width = kwargs.get("width")
    if width is not None and width != kwargs["num_features"]:
        raise ValueError(
            "ResidualNeuralNetwork requires width == num_features when node_list is "
            "not provided because PT-MELT residual blocks do not use projection layers."
        )


def _validate_transformer_width(kwargs: Mapping[str, Any]) -> None:
    width = kwargs.get("width")
    if width is not None and width % 2 != 0:
        raise ValueError(
            "TemporalTransformerNetwork requires an even width for positional encoding."
        )


def build_model_from_config(config: Mapping[str, Any]):
    """
    Build a PT-MELT model from a flat configuration dictionary.

    Args:
        config (Mapping[str, Any]): Model configuration. Must include ``arch_type``,
            ``num_features``, and ``num_outputs``. Architecture-specific keys are
            passed through to the selected PT-MELT model class.
    """

    arch_type = _normalize_architecture(_require_config(config, "arch_type"))

    if arch_type == "ann":
        model = ArtificialNeuralNetwork(**_common_model_kwargs(config))
        model.build()
        return model

    if arch_type == "resnet":
        kwargs = _common_model_kwargs(config)
        if kwargs.get("node_list") is not None and kwargs.get("depth") is None:
            kwargs["depth"] = len(kwargs["node_list"])
        layers_per_block = config.get("layers_per_block", 2)
        _validate_resnet_widths(kwargs, layers_per_block)
        model = ResidualNeuralNetwork(
            layers_per_block=layers_per_block,
            pre_activation=config.get("pre_activation", True),
            post_add_activation=config.get("post_add_activation", False),
            **kwargs,
        )
        model.build()
        return model

    if arch_type == "bnn":
        model = BayesianNeuralNetwork(
            num_points=config.get("num_points", 1),
            do_aleatoric=config.get("do_aleatoric", False),
            do_bayesian_output=config.get("do_bayesian_output", True),
            aleatoric_scale_factor=config.get("aleatoric_scale_factor", 5e-2),
            scale_epsilon=config.get("scale_epsilon", 1e-3),
            bayesian_mask=config.get("bayesian_mask", None),
            **_common_model_kwargs(config),
        )
        model.build()
        return model

    if arch_type == "rnn":
        model = RecurrentNeuralNetwork(
            rnn_type=config.get("rnn_type", "lstm"),
            return_sequences=config.get("return_sequences", False),
            head_type=config.get("head_type", "last"),
            **_common_model_kwargs(config, use_sampled_node_list=False),
        )
        model.build()
        return model

    if arch_type == "temporal_transformer":
        kwargs = _common_model_kwargs(config, use_sampled_node_list=False)
        _validate_transformer_width(kwargs)
        model = TemporalTransformerNetwork(
            num_heads=config.get("num_heads", 4),
            ff_dim=config.get("ff_dim", None),
            max_seq_len=config.get("max_seq_len", 2048),
            head_type=config.get("head_type", "last"),
            use_causal_mask=config.get("use_causal_mask", False),
            **kwargs,
        )
        model.build()
        return model

    if arch_type == "vae":
        kwargs = _common_model_kwargs(config, use_sampled_node_list=False)
        kwargs.setdefault("num_mixtures", 1)
        return VariationalAutoencoder(
            latent_dims=_require_config(config, "latent_dims"),
            encoder_node_list=_vae_node_list(
                config, "encoder_node_list", "encoder_layer"
            ),
            decoder_node_list=_vae_node_list(
                config, "decoder_node_list", "decoder_layer"
            ),
            **kwargs,
        )

    raise ValueError(f"Unsupported architecture type {arch_type}")


def model_builder(config: Mapping[str, Any]):
    """
    Build a model, optimizer, and criterion from a configuration dictionary.

    This preserves the original hyperparameter tuning API while delegating model
    construction, loss creation, and optimizer creation to PT-MELT model classes.
    """

    model = build_model_from_config(config)
    optimizer_kwargs = dict(config.get("optimizer_kwargs", {}))
    learning_rate = config.get("learning_rate", optimizer_kwargs.get("lr"))
    if learning_rate is None:
        raise KeyError("Missing required config key 'learning_rate'.")
    optimizer_kwargs.setdefault("lr", learning_rate)

    optimizer = model.get_optimizer(
        config.get("optimizer", "adam"),
        **optimizer_kwargs,
    )
    criterion = model.get_loss_fn(
        config.get("loss_fn", "mse"),
        reduction=config.get("loss_reduction", "mean"),
        mse_weight=config.get("mse_weight", None),
    )

    return model, optimizer, criterion


def _require_ray_core() -> dict[str, Any]:
    try:
        import ray
        from ray import tune
    except ImportError as exc:
        raise ImportError(
            "Ray Tune helpers require Ray. Install PT-MELT with Ray Tune "
            "dependencies before calling run_ray_tune()."
        ) from exc

    RunConfig = getattr(tune, "RunConfig", None)
    if RunConfig is None:
        try:
            from ray.air import RunConfig
        except ImportError as exc:
            raise ImportError("Unable to locate Ray Tune RunConfig.") from exc

    return {"ray": ray, "tune": tune, "RunConfig": RunConfig}


def _build_scheduler(
    scheduler: Any,
    metric: str,
    mode: str,
    max_epochs: int | None,
    search_space: Mapping[str, Any],
    scheduler_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if scheduler is None:
        return None
    if not isinstance(scheduler, str):
        return scheduler

    scheduler_name = scheduler.strip().lower()
    if scheduler_name in {"none", "null"}:
        return None

    scheduler_kwargs = dict(scheduler_kwargs or {})
    if scheduler_name == "asha":
        from ray.tune.schedulers import ASHAScheduler

        if max_epochs is not None:
            scheduler_kwargs.setdefault("max_t", max_epochs)
        scheduler_kwargs.setdefault("grace_period", 1)
        scheduler_kwargs.setdefault("reduction_factor", 3)
        return ASHAScheduler(metric=metric, mode=mode, **scheduler_kwargs)

    if scheduler_name == "pbt":
        from ray.tune.schedulers import PopulationBasedTraining

        scheduler_kwargs.setdefault("time_attr", "training_iteration")
        scheduler_kwargs.setdefault("perturbation_interval", 5)
        scheduler_kwargs.setdefault("hyperparam_mutations", dict(search_space))
        return PopulationBasedTraining(metric=metric, mode=mode, **scheduler_kwargs)

    raise ValueError(f"Unsupported Ray Tune scheduler '{scheduler}'.")


def _build_search_alg(
    search_alg: Any,
    metric: str,
    mode: str,
    max_concurrent: int | None = None,
    search_alg_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if search_alg is None:
        return None
    if not isinstance(search_alg, str):
        searcher = search_alg
    else:
        search_alg_name = search_alg.strip().lower()
        if search_alg_name in {"none", "null", "random"}:
            return None
        search_alg_kwargs = dict(search_alg_kwargs or {})
        if search_alg_name == "optuna":
            from ray.tune.search.optuna import OptunaSearch

            searcher = OptunaSearch(metric=metric, mode=mode, **search_alg_kwargs)
        else:
            raise ValueError(f"Unsupported Ray Tune search algorithm '{search_alg}'.")

    if max_concurrent is not None:
        from ray.tune.search import ConcurrencyLimiter

        searcher = ConcurrencyLimiter(searcher, max_concurrent=max_concurrent)
    return searcher


def _checkpoint_from_directory(checkpoint_module: Any, checkpoint_dir: str) -> Any:
    checkpoint_cls = getattr(checkpoint_module, "Checkpoint", None)
    if checkpoint_cls is None:
        from ray.air import Checkpoint

        checkpoint_cls = Checkpoint
    return checkpoint_cls.from_directory(checkpoint_dir)


def _ray_tune_trainable(
    config: Mapping[str, Any],
    train_dl: Any,
    val_dl: Any,
    metric: str,
    mode: str,
    metric_fn: Callable[..., Mapping[str, Any]] | None = None,
    checkpoint_interval: int = 1,
    device: str | None = None,
    step_kwargs: Mapping[str, Any] | None = None,
) -> None:
    ray_core = _require_ray_core()
    tune_module = ray_core["tune"]

    # Use Tune-native APIs inside Tune function trainables.
    # Calling ray.train.get_checkpoint/report in this context now raises
    # a DeprecationWarning exception in newer Ray releases.
    get_checkpoint = getattr(tune_module, "get_checkpoint", None)
    report = getattr(tune_module, "report", None)

    # Some Ray versions expose these APIs under tune.session.
    tune_session = getattr(tune_module, "session", None)
    if tune_session is not None:
        get_checkpoint = get_checkpoint or getattr(tune_session, "get_checkpoint", None)
        report = report or getattr(tune_session, "report", None)

    if get_checkpoint is None or report is None:
        raise RuntimeError(
            "Unable to locate Ray Tune checkpoint/report APIs. "
            "Please use a Ray version that provides tune.get_checkpoint "
            "and tune.report (or equivalents under tune.session)."
        )

    model, optimizer, criterion = model_builder(config)
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(run_device)

    start_epoch = 0
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            start_epoch = int((Path(checkpoint_dir) / "data.ckpt").read_text())

    best_metric = None
    best_epoch = None
    epochs = config.get("epochs", 1)
    step_kwargs = dict(step_kwargs or {})

    for epoch in range(start_epoch, epochs):
        train_loss = model.step(
            train_dl, optimizer, criterion, run_device, training=True, **step_kwargs
        )
        val_loss = model.step(
            val_dl, optimizer, criterion, run_device, training=False, **step_kwargs
        )

        metrics = {
            "iterations": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "checkpoint_epoch": None,
        }
        if metric_fn is not None:
            extra_metrics = metric_fn(
                model=model,
                config=config,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch,
                device=run_device,
            )
            if extra_metrics:
                metrics.update(dict(extra_metrics))

        metric_value = metrics.get(metric)
        if metric_value is not None and (
            best_metric is None
            or (mode == "min" and metric_value < best_metric)
            or (mode == "max" and metric_value > best_metric)
        ):
            best_metric = metric_value
            best_epoch = epoch + 1

        metrics["best_metric"] = best_metric
        metrics["best_epoch"] = best_epoch

        should_checkpoint = checkpoint_interval and (
            (epoch + 1) == epochs or (epoch + 1) % checkpoint_interval == 0
        )
        if should_checkpoint:
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                (Path(checkpoint_dir) / "data.ckpt").write_text(str(epoch + 1))
                metrics["checkpoint_epoch"] = epoch + 1
                report(
                    metrics,
                    checkpoint=_checkpoint_from_directory(tune_module, checkpoint_dir),
                )
        else:
            report(metrics)


def run_ray_tune(
    train_dl: Any,
    val_dl: Any,
    search_space: Mapping[str, Any],
    base_config: Mapping[str, Any] | None = None,
    metric: str = "val_loss",
    mode: str = "min",
    num_samples: int = 10,
    resources: Mapping[str, Any] | None = None,
    scheduler: Any = "asha",
    search_alg: Any = "optuna",
    metric_fn: Callable[..., Mapping[str, Any]] | None = None,
    ray_init_kwargs: Mapping[str, Any] | None = None,
    restore_path: str | None = None,
    storage_path: str | None = None,
    name: str | None = None,
    max_concurrent: int | None = None,
    checkpoint_interval: int = 1,
    scheduler_kwargs: Mapping[str, Any] | None = None,
    search_alg_kwargs: Mapping[str, Any] | None = None,
    tune_config_kwargs: Mapping[str, Any] | None = None,
    run_config_kwargs: Mapping[str, Any] | None = None,
    device: str | None = None,
    step_kwargs: Mapping[str, Any] | None = None,
    return_raw_results: bool = False,
) -> HPOResult:
    """
    Run a Ray Tune hyperparameter search for PT-MELT models.

    Args:
        train_dl: Training dataloader passed to ``model.step``.
        val_dl: Validation dataloader passed to ``model.step``.
        search_space: Ray Tune search-space dictionary.
        base_config: Fixed model/training configuration merged before ``search_space``.
        metric: Metric reported to Ray Tune for best-result selection.
        mode: ``"min"`` or ``"max"``.
        metric_fn: Optional callback that returns extra reported metrics.
    """

    if val_dl is None:
        raise ValueError("val_dl is required for Ray Tune hyperparameter tuning.")

    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'.")

    ray_core = _require_ray_core()
    ray = ray_core["ray"]
    tune = ray_core["tune"]
    RunConfig = ray_core["RunConfig"]

    if ray_init_kwargs is not None and not ray.is_initialized():
        ray.init(**dict(ray_init_kwargs))

    base_config = dict(base_config or {})
    search_space = dict(search_space)
    param_space = {**base_config, **search_space}
    max_epochs = param_space.get("epochs")

    scheduler_obj = _build_scheduler(
        scheduler,
        metric=metric,
        mode=mode,
        max_epochs=max_epochs,
        search_space=search_space,
        scheduler_kwargs=scheduler_kwargs,
    )
    search_alg_obj = _build_search_alg(
        search_alg,
        metric=metric,
        mode=mode,
        max_concurrent=max_concurrent,
        search_alg_kwargs=search_alg_kwargs,
    )

    trainable = tune.with_parameters(
        _ray_tune_trainable,
        train_dl=train_dl,
        val_dl=val_dl,
        metric=metric,
        mode=mode,
        metric_fn=metric_fn,
        checkpoint_interval=checkpoint_interval,
        device=device,
        step_kwargs=step_kwargs,
    )
    if resources is not None:
        trainable = tune.with_resources(trainable, resources=dict(resources))

    tune_config_params = {
        "num_samples": num_samples,
        "scheduler": scheduler_obj,
        **dict(tune_config_kwargs or {}),
    }
    if search_alg_obj is not None:
        tune_config_params["search_alg"] = search_alg_obj

    run_config_params = dict(run_config_kwargs or {})
    if name is not None:
        run_config_params["name"] = name
    if storage_path is not None:
        run_config_params["storage_path"] = storage_path

    if restore_path is not None:
        tuner = tune.Tuner.restore(path=restore_path, trainable=trainable)
    else:
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(**tune_config_params),
            param_space=param_space,
            run_config=RunConfig(**run_config_params),
        )

    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=mode)
    best_config = dict(best_result.config)
    best_hyperparameters = {
        key: best_config[key] for key in search_space if key in best_config
    }
    trial_history = {result.path: result.metrics_dataframe for result in results}
    best_metrics = dict(getattr(best_result, "metrics", {}) or {})
    metric_details = {
        "metric": metric,
        "mode": mode,
        "best_metric": best_metrics.get(metric),
        "best_trial_path": getattr(best_result, "path", None),
        "best_trial_id": best_metrics.get("trial_id", None),
        "best_metrics": best_metrics,
    }

    return HPOResult(
        best_config=best_config,
        best_hyperparameters=best_hyperparameters,
        metric_details=metric_details,
        trial_history=trial_history,
        results=results if return_raw_results else None,
    )


run_hyperparameter_tuning = run_ray_tune


__all__ = [
    "HPOResult",
    "SUPPORTED_ARCHITECTURES",
    "build_model_from_config",
    "model_builder",
    "run_hyperparameter_tuning",
    "run_ray_tune",
]
