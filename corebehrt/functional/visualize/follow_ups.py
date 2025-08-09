from pathlib import Path
import os
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

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

    # Prepare figure
    height = max(3, 0.7 * len(subject_ids))
    fig, ax = plt.subplots(figsize=(12, height))

    # y mapping
    y_map = {pid: i for i, pid in enumerate(subject_ids)}

    # Brackets and labels
    for _, row in fu.iterrows():
        y = y_map[int(row[PID_COL])]
        ax.hlines(
            y=y, xmin=row["start_dt"], xmax=row["end_dt"], color="lightgray", lw=2
        )
        ax.text(
            row["end_dt"],
            y,
            f" {int(row[PID_COL])}",
            va="center",
            ha="left",
            fontsize=9,
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
        merged = tmp.merge(fu[[PID_COL, START_COL, END_COL]], on=PID_COL, how="left")
        within = (merged[ABSPOS_COL] >= merged[START_COL]) & (
            merged[ABSPOS_COL] <= merged[END_COL]
        )
        for _, r in merged.iterrows():
            y = y_map[int(r[PID_COL])]
            alpha = 1.0 if bool(within.loc[_]) else 0.35
            ax.scatter(
                r["time_dt"],
                y,
                color=color,
                marker=marker,
                s=35,
                alpha=alpha,
                label=label,
            )

    # Colors
    default_outcome_colors = {}
    if outcomes:
        for i, name in enumerate(sorted(outcomes.keys())):
            default_outcome_colors[f"Outcome: {name}"] = sns.color_palette("Set2")[
                i % 8
            ]
    if outcome_colors:
        default_outcome_colors.update(outcome_colors)

    # Plot exposures
    add_events(exposures, "Exposure", color="black", marker="^")
    # Plot outcomes
    if outcomes:
        for name, df in outcomes.items():
            add_events(
                df,
                f"Outcome: {name}",
                color=default_outcome_colors.get(f"Outcome: {name}", "C1"),
                marker="o",
            )

    # Build legend without duplicates
    handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Exposure",
            markerfacecolor="black",
            markersize=7,
        )
    ]
    if outcomes:
        for name in sorted(outcomes.keys()):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Outcome: {name}",
                    markerfacecolor=default_outcome_colors.get(
                        f"Outcome: {name}", "C1"
                    ),
                    markersize=7,
                )
            )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    # Formatting
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            join(save_dir, "follow_ups_timeline.png"), dpi=200, bbox_inches="tight"
        )
    else:
        plt.show()

    plt.close(fig)
