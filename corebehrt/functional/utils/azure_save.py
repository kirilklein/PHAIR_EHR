from typing import Union
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from corebehrt.azure.util import is_azure_available


def is_running_in_azure_ml() -> bool:
    """Check if actually running in Azure ML environment"""
    import os

    azure_indicators = [
        os.environ.get("AZUREML_RUN_ID"),
        os.environ.get("AZUREML_RUN_OUTPUT_DIR"),
        os.path.exists("/mnt/azureml"),  # Common Azure ML mount point
    ]
    print("azure indicators", azure_indicators)
    return any(azure_indicators)


def save_figure_with_azure_copy(
    fig: Figure,
    save_path: Union[str, Path],
    *,
    close: bool = True,
    **savefig_kwargs,
) -> None:
    """
    Save a matplotlib Figure and, if on Azure, also copy it to outputs/figs/.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    save_path : str | Path
        Destination file path.
    close : bool, default True
        Whether to close the figure after saving.
    **savefig_kwargs :
        Passed through to Figure.savefig (e.g., dpi=300, bbox_inches="tight").
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the original figure
    print(f"Saving figure to: {save_path.absolute()}")
    fig.savefig(save_path, **savefig_kwargs)
    if close:
        plt.close(fig)

    # Mirror to Azure outputs/figs if available
    azure_sdk_available = is_azure_available()
    azure_ml_env = is_running_in_azure_ml()

    print(f"Azure SDK available: {azure_sdk_available}")
    print(f"Running in Azure ML: {azure_ml_env}")

    if azure_sdk_available or azure_ml_env:
        try:
            import os

            cwd = os.getcwd()
            print(f"Current working directory: {cwd}")

            # Print environment variables for debugging
            azure_env_vars = {
                "AZUREML_RUN_ID": os.environ.get("AZUREML_RUN_ID"),
                "AZUREML_RUN_OUTPUT_DIR": os.environ.get("AZUREML_RUN_OUTPUT_DIR"),
                "AZUREML_ROOT_PATH": os.environ.get("AZUREML_ROOT_PATH"),
            }
            print(f"Azure environment variables: {azure_env_vars}")

            # Try to use Azure ML's output path if available, otherwise use relative path
            azure_outputs = os.environ.get("AZUREML_RUN_OUTPUT_DIR", "outputs")
            out_dir = Path(azure_outputs) / "figs"

            print(f"Attempting to create Azure outputs directory: {out_dir.absolute()}")
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Successfully created directory: {out_dir.absolute()}")

            dest_path = out_dir / save_path.name
            print(f"Copying from {save_path.absolute()} to {dest_path.absolute()}")

            # Check if source file exists
            if not save_path.exists():
                print(f"Error: Source file does not exist: {save_path.absolute()}")
                return

            shutil.copy2(save_path, dest_path)

            # Verify the copy was successful
            if dest_path.exists():
                source_size = save_path.stat().st_size
                dest_size = dest_path.stat().st_size
                print(
                    f"Successfully copied figure to Azure outputs: {dest_path.absolute()}"
                )
                print(
                    f"File sizes - Source: {source_size} bytes, Destination: {dest_size} bytes"
                )
            else:
                print(
                    f"Error: Copy failed - destination file does not exist: {dest_path.absolute()}"
                )

        except Exception as e:
            import traceback

            print(f"Error copying to Azure outputs: {e}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
    else:
        print("Skipping Azure copy - not in Azure environment")
