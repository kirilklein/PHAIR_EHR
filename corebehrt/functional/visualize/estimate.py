import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Import the os module for path manipulation
from corebehrt.constants.causal.data import EffectColumns


def create_annotated_heatmap_matplotlib(
    df: pd.DataFrame,
    method_names: list,
    effect_name: str = "effect",
    save_path: str = None,
):
    """
    Creates an annotated heatmap using Matplotlib from a list of effect dictionaries.

    Args:
        df: A pandas DataFrame containing effect data with 'method', 'outcome',
            and the specified 'effect_name' columns.
        method_names: A list of method names to be displayed on the y-axis, maintaining their order.
        effect_name: The key in the effect dictionaries to visualize (e.g., 'effect', 'std_err').
        save_path: Optional. A string representing the file path where the plot should be saved
                   (e.g., 'heatmap.png', 'plots/my_heatmap.pdf'). If None, the plot is displayed.
    """  # Ensure 'method' and 'outcome' columns exist
    if (
        EffectColumns.method not in df.columns
        or EffectColumns.outcome not in df.columns
    ):
        raise ValueError("The DataFrame must contain 'method' and 'outcome' columns.")
    if effect_name not in df.columns:
        raise ValueError(
            f"'{effect_name}' column not found in the DataFrame. "
            f"Please ensure the DataFrame contains this column."
        )
    # Pivot the DataFrame
    heatmap_data = df.pivot_table(
        index=EffectColumns.method, columns=EffectColumns.outcome, values=effect_name
    ).reindex(method_names)

    # Determine if annotations should be displayed based on the number of outcomes
    num_outcomes = len(heatmap_data.columns)
    annotate_cells = num_outcomes <= 100

    plt.figure(
        figsize=(num_outcomes * 1.2, len(method_names) * 0.8)
    )  # Adjust figure size dynamically
    ax = sns.heatmap(
        heatmap_data,
        annot=False,  # We will manually control annotations
        fmt=".3f",  # Format for the annotation text
        cmap="plasma",  # Color map (you can choose others like "viridis", "YlGnBu", etc.)
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": effect_name.replace("_", " ").title()},
    )

    # Manually add annotations if annotate_cells is True
    if annotate_cells:
        for text in ax.texts:
            text.set_text(
                ""
            )  # Clear default annotations from seaborn's 'annot=True' if it was used

        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                value = heatmap_data.iloc[i, j]
                if pd.notnull(value):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

    ax.set_title(
        f"Heatmap of {effect_name.replace('_', ' ').title()} by Method and Outcome"
    )
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Method")
    plt.tight_layout()

    # --- New logic for saving or showing the plot ---
    if save_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()  # Close the plot to free up memory
