from contextlib import contextmanager
import time
from typing import Any

MLFLOW_AVAILABLE = False
MLFLOW_CLIENT = None

MLFLOW_CHILD_RUNS = "prefix"

try:
    # Try to import mlflow and set availability flag
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Metric

    MLFLOW_AVAILABLE = True
    MLFLOW_CLIENT = MlflowClient()
except ImportError:
    pass


def is_mlflow_available() -> bool:
    """
    Checks if mlflow module is available.
    """
    return MLFLOW_AVAILABLE


def start_run(name: str = None, nested: bool = False, log_system_metrics: bool = False):
    """
    Starts an mlflow run. Used in the Azure wrapper and should
    not in general be used elsewhere.

    :param name: Name of run
    :param nested: If the run should be nested.
    :param log_system_metrics: If enabled, log system metrics (CPU/GPU/mem).

    """

    # Return a dummpy context manager so as to not raise an
    # error if mlflow is not available
    @contextmanager
    def dummy_cm():
        yield None

    run = dummy_cm()
    if is_mlflow_available():
        try:
            run = mlflow.start_run(
                run_name=name, nested=nested, log_system_metrics=log_system_metrics
            )
        except Exception as e:
            print(f"Error starting mlflow run: {e}")

    return run


def end_run():
    """
    Ends an mlflow run. Used in the Azure wrapper and should
    not in general be used elsewhere.
    """
    if is_mlflow_available():
        mlflow.end_run()


def get_run_and_prefix():
    """Get the current MLflow run and compute prefix"""
    run = mlflow.active_run()

    # Handle case where no run is active
    if run is None:
        # Either start a new run or return early
        import warnings

        warnings.warn("No active MLflow run found. Skipping logging.")
        return None, ""

    prefix = ""
    # Check if run has valid info before accessing it
    if run and hasattr(run, "info") and run.info:
        temp_run = run
        while (parent := mlflow.get_parent_run(temp_run.info.run_id)) is not None:
            parent_name = parent.info.run_name or ""
            if parent_name:
                prefix = parent_name + "/" + prefix
            temp_run = parent
        # The top-level run is the one we want to log to.
        run = temp_run

    return run, prefix


def setup_metrics_dir(name: str):
    """
    Shorthand for starting a sub-run where metrics will be logged.
    Use as context manager.

    :param name: Name of azure "sub-dir"/metrics pane.
    """
    return start_run(name=name, nested=True)


#
# Simple wrapper functions below. Review full args at:
# https://www.mlflow.org/docs/latest/python_api/mlflow.html
#


def log_metric(key: str, *args, **kwargs):
    """
    Logs a metric to the job (if mlflow is available).
    Parameters are the same as for mlflow.log_metric.

    Important parameters:

    :param key: Name of the metric.
    :param value: Value of the metric.
    :param step: Step for the metric (used for plotting graphs).
    """
    if is_mlflow_available():
        run, prefix = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        MLFLOW_CLIENT.log_metric(run.info.run_id, prefix + key, *args, **kwargs)


def log_metrics(metrics: dict, *args, **kwargs):
    """
    Log multiple metrics

    :param metrics: dict of metrics.
    :param step: Step for the metric (used for plotting graphs).
    """
    if is_mlflow_available():
        run, prefix = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        # Prefix each metric key
        prefixed_metrics = {prefix + k: v for k, v in metrics.items()}
        MLFLOW_CLIENT.log_metrics(run.info.run_id, prefixed_metrics, *args, **kwargs)


def log_param(key: str, value: Any) -> None:
    """Log a parameter to the job (if mlflow is available)."""
    if is_mlflow_available():
        run, prefix = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        MLFLOW_CLIENT.log_param(run.info.run_id, prefix + key, value)


def log_params(params: dict, *args, **kwargs):
    """
    Log multiple parameters.

    :param params: dict of parameters.
    """
    if is_mlflow_available():
        run, prefix = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        prefixed_params = {prefix + k: v for k, v in params.items()}
        MLFLOW_CLIENT.log_params(run.info.run_id, prefixed_params, *args, **kwargs)


def log_image(image, *args, **kwargs):
    """
    Log an image

    :param image: e.g. numpy array or PIL image.
    :param artifact_file: filename to save image under.
    """
    if is_mlflow_available():
        run, prefix = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        # Prefix the artifact_file path
        if "artifact_file" in kwargs:
            kwargs["artifact_file"] = prefix + kwargs["artifact_file"]
        MLFLOW_CLIENT.log_image(run.info.run_id, image, *args, **kwargs)


def log_figure(figure, *args, **kwargs):
    """
    Log a figure (e.g. matplotlib)

    :param figure: e.g. matplotlib figure.
    :param artifact_file: filename to save image under.
    """
    if is_mlflow_available():
        run, prefix = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        # Prefix the artifact_file path
        if "artifact_file" in kwargs:
            kwargs["artifact_file"] = prefix + kwargs["artifact_file"]
        MLFLOW_CLIENT.log_figure(run.info.run_id, figure, *args, **kwargs)


def log_batch(*args, **kwargs):
    """
    Log a batch of metrics

    :param metrics: metrics list
    """
    if is_mlflow_available():
        run, _ = get_run_and_prefix()
        if run is None:
            # Skip logging if no run is active
            return
        MLFLOW_CLIENT.log_batch(*args, run_id=run.info.run_id, **kwargs)


def metric(name, value, step):
    if is_mlflow_available():
        timestamp = int(time.time() * 1000)
        _, prefix = get_run_and_prefix()
        if prefix is None:
            # Skip logging if no run is active
            return (name, value)
        return Metric(prefix + name, value, timestamp, step)
    else:
        return (name, value)
