from typing import Union
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
from corebehrt.azure.util import is_azure_available

logger = logging.getLogger(__name__)


def is_running_in_azure_ml() -> bool:
    """Check if actually running in Azure ML environment"""
    import os

    azure_indicators = [
        os.environ.get("AZUREML_RUN_ID"),
        os.environ.get("AZUREML_RUN_OUTPUT_DIR"),
        os.path.exists("/mnt/azureml"),  # Common Azure ML mount point
    ]
    return any(azure_indicators)


def save_figure_with_azure_copy(
    fig: Figure,
    save_path: Union[str, Path],
    *,
    close: bool = True,
    **savefig_kwargs,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save locally
    logger.info(f"Saving figure to: {save_path.absolute()}")
    fig.savefig(save_path, **savefig_kwargs)
    if close:
        plt.close(fig)

    # Mirror to Azure
    azure_sdk_available = is_azure_available()
    azure_ml_env = is_running_in_azure_ml()

    if azure_sdk_available or azure_ml_env:
        try:
            import os

            azure_outputs = os.environ.get("AZUREML_RUN_OUTPUT_DIR", "outputs")
            out_dir = Path(azure_outputs) / "figs"

            # ✅ Extract subpath starting from "figs"/"figures" if present.
            # If absent, place just the filename in outputs/figs/.
            parts_lower = [p.lower() for p in save_path.parts]
            if "figs" in parts_lower:
                idx = parts_lower.index("figs")
                rel_path = Path(*save_path.parts[idx + 1 :])  # after figs/
            elif "figures" in parts_lower:
                idx = parts_lower.index("figures")
                rel_path = Path(*save_path.parts[idx + 1 :])  # after figures/
            else:
                # No figs/figures in path → save at outputs/figs/<filename>
                rel_path = Path(save_path.name)

            dest_path = out_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if not save_path.exists():
                logger.warning(
                    f"Skip copying to Azure: Source file does not exist: {save_path.absolute()}"
                )
                return

            shutil.copy2(save_path, dest_path)

            if dest_path.exists():
                logger.info(f"Successfully copied to Azure: {dest_path.absolute()}")
            else:
                pass

        except Exception as e:
            logger.warning(f"Error copying to Azure outputs: {e}")
    else:
        logger.warning("Skipping Azure copy - not in Azure environment")
