from pathlib import Path
import os
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates

from corebehrt.constants.causal.data import END_COL, START_COL
from corebehrt.constants.data import PID_COL, ABSPOS_COL
from corebehrt.functional.utils.time import get_datetime_from_hours_since_epoch


def plot_follow_up_distribution(
    follow_ups: pd.DataFrame, binary_exposure: pd.Series, out_dir: str
):
    """
    Plot the follow-up distribution.
    """
    follow_ups = follow_ups.copy()
    follow_ups["follow_up_days"] = (
        follow_ups[END_COL] - follow_ups[START_COL]
    ) / 24  # hours to days
    follow_ups = pd.merge(
        follow_ups, binary_exposure.to_frame(), left_on=PID_COL, right_index=True
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=follow_ups,
        x="follow_up_days",
        hue="has_outcome",
        kde=False,
        element="step",
        stat="count",
        common_norm=False,
    )

    plt.title("Follow-up Time Distribution by Group")
    plt.xlabel("Follow-up Time (days)")
    plt.ylabel("Density")

    output_path = Path(out_dir) / "follow_up_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()


def plot_followups_timeline(
    exposures: pd.DataFrame,
    outcomes: dict[str, pd.DataFrame] | None,
    follow_ups: pd.DataFrame,
    index_date_matching: pd.DataFrame,
    subject_ids: list[int] | None = None,
    n_random_subjects: int = 8,
    title: str = "Follow-up Timeline (Exposures & Outcomes)",
    outcome_colors: dict[str, str] | None = None,
    save_dir: str | None = None,
):
    """
    Matplotlib timeline showing follow-up windows, exposures, and outcomes.

    - exposures: DataFrame with PID_COL and ABSPOS_COL (hours since epoch)
    - outcomes: Dict[name -> DataFrame with PID_COL and ABSPOS_COL]
    - follow_ups: DataFrame with PID_COL, START_COL, END_COL (hours since epoch)
    """
    if follow_ups is None or follow_ups.empty:
        print("No follow-ups provided. Nothing to plot.")
        return

    # Select subjects
    all_pids = follow_ups[PID_COL].astype(int).unique()
    if not subject_ids:
        n_random_subjects = min(n_random_subjects, len(all_pids))
        subject_ids = (
            pd.Series(all_pids)
            .sample(n_random_subjects, random_state=42)
            .sort_values()
            .tolist()
        )
    else:
        subject_ids = [int(pid) for pid in subject_ids if int(pid) in all_pids]
    if len(subject_ids) == 0:
        print("No matching subject_ids found in follow-ups.")
        return

    fu = follow_ups[follow_ups[PID_COL].isin(subject_ids)].copy()
    fu["start_dt"] = get_datetime_from_hours_since_epoch(fu[START_COL])
    fu["end_dt"] = get_datetime_from_hours_since_epoch(fu[END_COL])

    # --- IMPROVEMENT: Tighter vertical spacing to fit more subjects ---
    # Reduced the multiplier from 0.7 to 0.45 for denser plotting.
    height = max(3, 0.45 * len(subject_ids))
    fig, ax = plt.subplots(figsize=(12, height))

    # y mapping
    y_map = {pid: i for i, pid in enumerate(subject_ids)}

    # --- IMPROVEMENT: Calculate text offset based on plot's date range ---
    # This makes the label placement robust to different time scales.
    total_min_date = fu["start_dt"].min()
    total_max_date = fu["end_dt"].max()
    date_range = total_max_date - total_min_date
    text_offset = date_range * 0.01  # 1% of total range

    # Brackets and labels
    for _, row in fu.iterrows():
        y = y_map[int(row[PID_COL])]
        ax.hlines(
            y=y, xmin=row["start_dt"], xmax=row["end_dt"], color="lightgray", lw=2
        )
        # --- IMPROVEMENT: Smaller font and offset to the right ---
        ax.text(
            row["end_dt"] + text_offset,  # Position text to the right
            y,
            f"{int(row[PID_COL])}",  # No leading space needed
            va="center",
            ha="left",
            fontsize=8,  # Smaller font
            color="gray",
        )
        ax.text(row["start_dt"], y, "[", va="center", ha="right", fontsize=12)
        ax.text(row["end_dt"], y, "]", va="center", ha="left", fontsize=12)

    # Prepare events
    def add_events(df: pd.DataFrame, label: str, color: str, marker: str):
        if df is None or df.empty:
            return
        tmp = df[df[PID_COL].isin(subject_ids)].copy()
        if tmp.empty or ABSPOS_COL not in tmp.columns:
            return
        tmp["time_dt"] = get_datetime_from_hours_since_epoch(tmp[ABSPOS_COL])
        # Use a left merge to ensure we check all events, even those outside any fu window
        merged = tmp.merge(fu[[PID_COL, "start_dt", "end_dt"]], on=PID_COL, how="left")

        for _, r in merged.iterrows():
            # Check if event is within the follow-up period for that subject
            is_within = (
                (r["start_dt"] <= r["time_dt"] <= r["end_dt"])
                if pd.notna(r["start_dt"])
                else False
            )
            alpha = 1.0 if is_within else 0.35
            ax.scatter(
                r["time_dt"],
                y_map[int(r[PID_COL])],
                color=color,
                marker=marker,
                s=40,  # Slightly larger symbols
                alpha=alpha,
                label=label,  # Label every point for auto-legend
                zorder=5,  # Ensure points are drawn on top of lines
            )

    # Colors
    default_outcome_colors = {}
    if outcomes:
        palette = sns.color_palette("Set2", n_colors=len(outcomes))
        for i, name in enumerate(sorted(outcomes.keys())):
            default_outcome_colors[f"Outcome: {name}"] = palette[i]
    if outcome_colors:
        default_outcome_colors.update(outcome_colors)

    # Plot exposures and outcomes
    add_events(exposures, "Exposure", color="black", marker="^")
    if outcomes:
        for name, df in sorted(outcomes.items()):
            label_name = f"Outcome: {name}"
            add_events(
                df,
                label_name,
                color=default_outcome_colors.get(label_name, "C1"),
                marker="o",
            )

    # --- IMPROVEMENT: Automated legend creation to avoid duplicates ---
    # This is more robust and cleaner than manual legend building.
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Events",
    )

    # Formatting
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_yticks([])  # Hide y-axis ticks
    ax.spines[["left", "right", "top"]].set_visible(False)  # Remove frame
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend

    # Save or show plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = join(save_dir, "follow_ups_timeline.png")
        fig.savefig(path, dpi=200)
        print(f"Plot saved to {path}")
    else:
        plt.show()

    plt.close(fig)
