import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import os


def plot_outcome_distribution(
    df: pd.DataFrame, outcome_dir: str, max_outcomes_per_plot: int = 100
) -> None:
    """
    Plots the distribution of positive outcomes, splitting into multiple files if needed.

    Args:
        df (pd.DataFrame): DataFrame with outcome columns.
        outcome_dir (str): Directory where the plot(s) will be saved.
        max_outcomes_per_plot (int): Maximum number of outcomes to display in a single plot.
    """
    # Ensure the output directory exists
    os.makedirs(outcome_dir, exist_ok=True)

    outcome_proportions = df.mean().sort_values(ascending=True)
    num_outcomes = len(outcome_proportions)

    if num_outcomes <= max_outcomes_per_plot:
        # Plotting a single figure
        plt.figure(
            figsize=(10, max(6, num_outcomes * 0.3))
        )  # Adjust height dynamically
        bars = plt.barh(
            outcome_proportions.index, outcome_proportions.values, color="skyblue"
        )
        plt.xlabel("Proportion of Positive Outcomes")
        plt.ylabel("Outcomes")
        plt.title("Distribution of Positive Outcomes")
        plt.xlim(0, 1)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.2f}",
                va="center",
            )

        plt.tight_layout()
        save_path = join(outcome_dir, "outcome_distribution.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to '{save_path}'")

    else:
        # Splitting into multiple figures
        num_plots = (num_outcomes + max_outcomes_per_plot - 1) // max_outcomes_per_plot
        print(
            f"Number of outcomes ({num_outcomes}) exceeds the limit of {max_outcomes_per_plot}. Generating {num_plots} plots."
        )

        for i in range(num_plots):
            start_index = i * max_outcomes_per_plot
            end_index = start_index + max_outcomes_per_plot
            subset_proportions = outcome_proportions.iloc[start_index:end_index]

            num_outcomes_in_plot = len(subset_proportions)

            plt.figure(
                figsize=(10, max(6, num_outcomes_in_plot * 0.3))
            )  # Adjust height dynamically
            bars = plt.barh(
                subset_proportions.index, subset_proportions.values, color="skyblue"
            )
            plt.xlabel("Proportion of Positive Outcomes")
            plt.ylabel("Outcomes")
            plt.title(f"Distribution of Positive Outcomes (Part {i + 1}/{num_plots})")
            plt.xlim(0, 1)

            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{width:.2f}",
                    va="center",
                )

            plt.tight_layout()
            save_path = join(outcome_dir, f"outcome_distribution_{i + 1}.png")
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory
            print(f"Plot saved to '{save_path}'")
