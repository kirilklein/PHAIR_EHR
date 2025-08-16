import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import os
import seaborn as sns


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
        plt.xlim(0, 1.05)

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


def plot_filtering_stats(stats: dict, output_dir: str, max_items_per_plot: int = 100):
    """
    Plots patient counts before and after filtering, splitting into multiple files if needed.

    Args:
        stats (dict): A dictionary containing the before/after counts.
        output_dir (str): The directory to save the plot(s) in.
        max_items_per_plot (int): Maximum number of items to display in a single plot.
    """
    if not stats:
        print("Statistics dictionary is empty, skipping plot generation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    plot_data = []
    for name, values in stats.items():
        # Append the "Before" count
        plot_data.append(
            {
                "name": name,
                "status": "Before Filtering",
                "count": values.get("before", 0),
            }
        )

        # --- FIX IS HERE ---
        # For the "After" bar, we plot the number of positive events.
        # Use .get(1, 0) to safely get the count of positive cases (key 1), defaulting to 0.
        after_counts = values.get("after", {})
        positive_events_after = after_counts.get(1, 0)

        plot_data.append(
            {
                "name": name,
                "status": "Within Follow-up Window",  # Renamed for clarity
                "count": positive_events_after,
            }
        )

    df_plot = pd.DataFrame(plot_data)

    # Sort by the 'before' count to have a more organized plot
    sorted_names = (
        df_plot[df_plot["status"] == "Before Filtering"]
        .sort_values("count", ascending=False)["name"]
        .tolist()
    )
    df_plot["name"] = pd.Categorical(
        df_plot["name"], categories=sorted_names, ordered=True
    )
    df_plot = df_plot.sort_values("name")

    # (The rest of the plotting logic with splitting into chunks remains the same)
    item_names = df_plot["name"].unique().tolist()
    num_items = len(item_names)
    if num_items == 0:
        print("No data to plot.")
        return

    num_plots = (num_items + max_items_per_plot - 1) // max_items_per_plot
    if num_plots > 1:
        print(
            f"Number of items ({num_items}) exceeds the limit of {max_items_per_plot}. "
            f"Generating {num_plots} plots."
        )

    # 3. Loop through chunks and generate a plot for each
    for i in range(num_plots):
        start_index = i * max_items_per_plot
        end_index = start_index + max_items_per_plot

        # Get the subset of names and data for the current plot
        chunk_names = item_names[start_index:end_index]
        subset_df = df_plot[df_plot["name"].isin(chunk_names)]

        # Dynamically adjust figure size
        fig_height = max(6, len(chunk_names) * 0.4)
        plt.figure(figsize=(12, fig_height))
        sns.set_style("whitegrid")

        bar_plot = sns.barplot(
            x="count", y="name", hue="status", data=subset_df, orient="h"
        )

        # Set titles and labels
        title = "Patient Counts Before and After Filtering"
        if num_plots > 1:
            title += f" (Part {i + 1}/{num_plots})"

        plt.title(title, fontsize=16)
        plt.xlabel("Number of Patients", fontsize=12)
        plt.ylabel("Exposure / Outcome", fontsize=12)
        plt.legend(title="Filtering Status")

        # Add text labels on bars
        for p in bar_plot.patches:
            width = p.get_width()
            if width > 0:  # Only label bars with a value
                bar_plot.annotate(
                    f"{width:.0f}",
                    (width, p.get_y() + p.get_height() / 2.0),
                    ha="left",
                    va="center",
                    xytext=(5, 0),
                    textcoords="offset points",
                )

        # Adjust x-axis limit for labels
        plt.xlim(right=subset_df["count"].max() * 1.15)
        plt.tight_layout()

        # Determine save path
        base_filename = "filtering_counts_comparison"
        if num_plots > 1:
            save_path = join(output_dir, f"{base_filename}_{i + 1}.png")
        else:
            save_path = join(output_dir, f"{base_filename}.png")

        plt.savefig(save_path)
        print(f"Saved filtering statistics plot to {save_path}")
        plt.close()
