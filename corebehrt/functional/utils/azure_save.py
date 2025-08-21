from typing import Union
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from corebehrt.azure.util import is_azure_available


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

    fig.savefig(save_path, **savefig_kwargs)
    if close:
        plt.close(fig)

    # Mirror to Azure outputs/figs if available
    if is_azure_available():
        try:
            out_dir = Path("outputs/figs")
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(save_path, out_dir / save_path.name)
        except Exception as e:
            print(f"Warning: failed to copy to Azure outputs: {e}")
