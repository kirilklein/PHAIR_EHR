import os
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from corebehrt.constants.causal.data import DEATH_COL, END_COL, START_COL
from corebehrt.constants.data import ABSPOS_COL, PID_COL, TIMESTAMP_COL
from corebehrt.functional.utils.time import (
    get_datetime_from_hours_since_epoch,
    get_hours_since_epoch,
)


def plot_followup_start_end_distribution(
    follow_ups: pd.DataFrame,
    exposure: pd.Series,
    out_dir: str,
    *,
    pid_col: str = PID_COL,
    start_col: str = START_COL,  # absolute hours since epoch
    end_col: str = END_COL,  # absolute hours since epoch
    index_col: str = TIMESTAMP_COL,  # datetime object or string
    mode: str = "relative",  # "relative" or "absolute"
    colors: dict | None = None,  # optional, e.g. {0: "tab:blue", 1:"tab:orange"}
    dpi: int = 300,
):
    """
    Plots the distribution of follow-up start and end times, grouped by exposure.

    - mode="relative": x = days since index date (time=0 at index_col)
    - mode="absolute": x = time (normal calendar-time view)
    """
    # Minimal prep
    df = follow_ups[[pid_col, start_col, end_col, index_col]].copy()
    df = df.merge(
        exposure.rename("exposure"), left_on=pid_col, right_index=True, how="left"
    )
    df["exposure"] = df["exposure"].fillna(0).astype(int)

    # Convert index_col to hours since epoch if it's a datetime object or string
    if pd.api.types.is_string_dtype(
        df[index_col]
    ) or pd.api.types.is_datetime64_any_dtype(df[index_col]):
        df[index_col] = pd.to_datetime(df[index_col]).apply(get_hours_since_epoch)

    # Ensure all columns are numeric (hours since epoch)
    for col in [start_col, end_col, index_col]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col]).apply(get_hours_since_epoch)

    df = df.dropna(subset=[start_col, end_col, index_col])

    # Ensure start <= end
    swap = df[start_col] > df[end_col]
    if swap.any():
        df.loc[swap, [start_col, end_col]] = df.loc[
            swap, [end_col, start_col]
        ].to_numpy()

    # Choose x-transform
    if mode == "relative":
        df["start_plot"] = (df[start_col] - df[index_col]) / 24
        df["end_plot"] = (df[end_col] - df[index_col]) / 24
        xlabel = "Days relative to index date"
        title_suffix = "(relative)"
    elif mode == "absolute":
        # --- MODIFICATION START ---
        # Convert hours since epoch to datetime objects for a calendar-time axis
        df["start_plot"] = get_datetime_from_hours_since_epoch(df[start_col])
        df["end_plot"] = get_datetime_from_hours_since_epoch(df[end_col])
        xlabel = "Date"
        title_suffix = "(absolute)"
        # --- MODIFICATION END ---
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=dpi)

    # Plot start times distribution
    sns.kdeplot(
        data=df,
        x="start_plot",
        hue="exposure",
        ax=axes[0],
        palette=colors,
        fill=True,
        common_norm=False,
    )
    axes[0].set_title(f"Distribution of Follow-up Start Times {title_suffix}")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)

    # Plot end times distribution
    sns.kdeplot(
        data=df,
        x="end_plot",
        hue="exposure",
        ax=axes[1],
        palette=colors,
        fill=True,
        common_norm=False,
    )
    axes[1].set_title(f"Distribution of Follow-up End Times {title_suffix}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    # --- MODIFICATION START ---
    # Add date formatting for absolute mode
    if mode == "absolute":
        # Format the x-axis to show dates nicely
        date_format = mdates.DateFormatter("%Y-%m")
        axes[1].xaxis.set_major_formatter(date_format)
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate()  # Auto-formats the x-axis labels (rotation, alignment)
    # --- MODIFICATION END ---

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        join(out_dir, f"follow_up_start_end_distribution_{mode}.png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Plot saved to {join(out_dir, f'follow_up_start_end_distribution_{mode}.png')}"
    )


def _prepare_followup_data_for_coverage(
    follow_ups: pd.DataFrame,
    exposure: pd.Series,
    pid_col: str,
    start_col: str,
    end_col: str,
    index_col: str,
) -> pd.DataFrame:
    """Prepares and cleans the follow-up and exposure data for plotting."""
    df = follow_ups[[pid_col, start_col, end_col, index_col]].copy()
    df = df.merge(
        exposure.rename("exposure"), left_on=pid_col, right_index=True, how="left"
    )
    df["exposure"] = df["exposure"].fillna(0).astype(int)

    for col in [start_col, end_col, index_col]:
        if pd.api.types.is_string_dtype(
            df[col]
        ) or pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col]).apply(get_hours_since_epoch)

    df = df.dropna(subset=[start_col, end_col, index_col])

    swap = df[start_col] > df[end_col]
    if swap.any():
        df.loc[swap, [start_col, end_col]] = df.loc[
            swap, [end_col, start_col]
        ].to_numpy()

    return df


def _calculate_coverage_curves(df: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    """Calculates coverage curves using a difference array for efficiency."""
    xmin, xmax = int(df["start_d"].min()), int(df["end_d"].max())
    L = xmax - xmin + 2  # +2 for sentinel slot and inclusive range

    curves = []
    for g in sorted(df["exposure"].unique()):
        sub = df[df["exposure"] == g]
        if sub.empty:
            continue
        total = sub[pid_col].nunique()

        diff = np.zeros(L, dtype=np.int32)
        s = (sub["start_d"].to_numpy() - xmin).clip(0, L - 1)
        e = (sub["end_d"].to_numpy() - xmin).clip(0, L - 1)

        np.add.at(diff, s, 1)
        e1 = e + 1
        np.add.at(diff, e1[e1 < L], -1)

        counts = diff.cumsum()[:-1]
        x = np.arange(xmin, xmax + 1)
        curves.append(
            pd.DataFrame(
                {
                    "x": x,
                    "count": counts,
                    "proportion": counts / total if total > 0 else 0.0,
                    "exposure": g,
                }
            )
        )
    return pd.concat(curves, ignore_index=True)


def plot_followup_coverage(
    follow_ups: pd.DataFrame,
    exposure: pd.Series,
    out_dir: str,
    *,
    pid_col: str = PID_COL,
    start_col: str = START_COL,  # absolute hours since epoch
    end_col: str = END_COL,  # absolute hours since epoch
    index_col: str = TIMESTAMP_COL,  # absolute hours since epoch (index date)
    mode: str = "relative",  # "relative" or "absolute"
    normalize: bool = True,  # True → proportions, False → counts
    colors: dict | None = None,  # optional, e.g. {0: "tab:blue", 1:"tab:orange"}
    dpi: int = 300,
):
    """
    Plots coverage of patients under follow-up over time, grouped by exposure.

    - mode="relative": x = days since index date (time=0 at index_col)
    - mode="absolute": x = time (normal calendar-time view)
    - normalize=True: plot proportions; False: plot counts
    Returns a tidy DataFrame with the calculated coverage data.
    """
    # 1. Prepare Data
    df = _prepare_followup_data_for_coverage(
        follow_ups, exposure, pid_col, start_col, end_col, index_col
    )

    # 2. Calculate integer-day start/end points
    if mode == "relative":
        df["start_d"] = np.floor((df[start_col] - df[index_col]) / 24).astype(int)
        df["end_d"] = np.ceil((df[end_col] - df[index_col]) / 24).astype(int)
        xlabel = "Days relative to index"
    elif mode == "absolute":
        df["start_d"] = np.floor(df[start_col] / 24).astype(int)  # Days since epoch
        df["end_d"] = np.ceil(df[end_col] / 24).astype(int)  # Days since epoch
        xlabel = "Date"
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")

    # 3. Calculate Coverage Curves
    coverage = _calculate_coverage_curves(df, pid_col)

    # 4. Transform x-axis for plotting
    if mode == "absolute":
        # Convert integer days since epoch back to datetime for plotting
        coverage["x_plot"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(
            coverage["x"], unit="D"
        )
    else:
        coverage["x_plot"] = coverage["x"]  # For relative, x is just integer days

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    ycol = "proportion" if normalize else "count"

    for g in sorted(coverage["exposure"].unique()):
        part = coverage[coverage["exposure"] == g]
        color_arg = {"color": colors[g]} if (colors and g in colors) else {}
        ax.plot(
            part["x_plot"],
            part[ycol],
            label=f"Exposure={g}",
            linewidth=2,
            **color_arg,
        )

    if mode == "relative":
        ax.axvline(0, linestyle="--", alpha=0.6, color="k")

    if mode == "absolute":
        # Format the x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "Proportion under follow-up" if normalize else "Patients under follow-up"
    )
    ax.set_title(f"Follow-up coverage ({mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.margins(x=0.01)

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()

    save_path = join(out_dir, f"follow_up_coverage_curve_{mode}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {save_path}")
    return coverage


def plot_follow_up_duration_distribution(
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
        hue="exposure",
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


def _get_subject_groups(
    follow_ups: pd.DataFrame,
    index_date_matching: pd.DataFrame,
    n_random_subjects: int,
    seed: int,
) -> List[List[int]]:
    """Selects random subjects and includes their matched pairs."""
    all_pids = set(follow_ups[PID_COL].astype(int).unique())
    selected_subjects = set()
    subject_groups = []

    matching_df = index_date_matching.copy()
    if "control_subject_id" in matching_df.columns:
        matching_df["control_subject_id"] = matching_df["control_subject_id"].astype(
            int
        )
    if "exposed_subject_id" in matching_df.columns:
        matching_df["exposed_subject_id"] = matching_df["exposed_subject_id"].astype(
            int
        )

    available_pids = list(all_pids)
    np.random.seed(seed)

    while len(subject_groups) < n_random_subjects and available_pids:
        base_pid = np.random.choice(
            [p for p in available_pids if p not in selected_subjects]
        )
        if base_pid in selected_subjects:
            continue

        group = {base_pid}

        # Find all partners matched with base_pid
        control_matches = matching_df[matching_df["control_subject_id"] == base_pid]
        exposed_matches = matching_df[matching_df["exposed_subject_id"] == base_pid]

        for _, row in control_matches.iterrows():
            partner_id = int(row["exposed_subject_id"])
            if partner_id in all_pids:
                group.add(partner_id)

        for _, row in exposed_matches.iterrows():
            partner_id = int(row["control_subject_id"])
            if partner_id in all_pids:
                group.add(partner_id)

        subject_groups.append(sorted(list(group)))
        selected_subjects.update(group)

    return subject_groups


def _plot_censor_dates(ax: plt.axes, censor_dates: pd.Series, y_map: Dict[int, int]):
    """Plots censor dates as vertical dashed lines."""
    if censor_dates is None or censor_dates.empty:
        return

    censor_df = censor_dates.reset_index()
    censor_df.columns = [PID_COL, "censor_abspos"]
    censor_df[PID_COL] = censor_df[PID_COL].astype(int)

    # Filter for subjects in our plot
    subjects_in_plot = list(y_map.keys())
    censor_df = censor_df[censor_df[PID_COL].isin(subjects_in_plot)]

    if censor_df.empty:
        return

    censor_df["censor_dt"] = get_datetime_from_hours_since_epoch(
        censor_df["censor_abspos"]
    )

    for _, row in censor_df.iterrows():
        pid = int(row[PID_COL])
        y = y_map[pid]
        ax.vlines(
            x=row["censor_dt"],
            ymin=y - 0.4,
            ymax=y + 0.4,
            color="dimgray",
            linestyle="--",
            lw=1.5,
            label="Censor Date",
            zorder=4,  # Just below events
        )


def plot_followups_timeline(
    exposures: pd.DataFrame,
    outcomes: Optional[Dict[str, pd.DataFrame]],
    follow_ups: pd.DataFrame,
    index_date_matching: pd.DataFrame,
    censor_dates: pd.Series,
    subject_ids: Optional[List[int]] = None,
    n_random_subjects: int = 8,
    title: str = "Follow-up Timeline (Exposures & Outcomes)",
    outcome_colors: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
    seed: int = 42,
):
    """
    Matplotlib timeline showing follow-up windows, exposures, outcomes, censor dates, and deaths.
    When selecting subjects randomly, their matched pairs (if any) are also included.

    - exposures: DataFrame with PID_COL and ABSPOS_COL (hours since epoch)
    - outcomes: Dict[name -> DataFrame with PID_COL and ABSPOS_COL]
    - follow_ups: DataFrame with PID_COL, START_COL, END_COL, DEATH_COL (hours since epoch)
    - index_date_matching: DataFrame with control_subject_id, exposed_subject_id
    - censor_dates: Series with PID_COL as index and censor_date (hours since epoch) as values
    """
    if follow_ups is None or follow_ups.empty:
        print("No follow-ups provided. Nothing to plot.")
        return

    # --- REFACTORED: Subject Selection Logic ---
    if not subject_ids:
        subject_groups = _get_subject_groups(
            follow_ups, index_date_matching, n_random_subjects, seed
        )
        subject_ids = [pid for group in subject_groups for pid in group]
    else:
        all_pids = set(follow_ups[PID_COL].astype(int).unique())
        subject_ids = [int(pid) for pid in subject_ids if int(pid) in all_pids]
        subject_groups = [[pid] for pid in subject_ids]

    if not subject_ids:
        print("No valid subject_ids found to plot.")
        return

    # --- REFACTORED: Centralized Data Preparation ---
    fu_plot = follow_ups[follow_ups[PID_COL].isin(subject_ids)].copy()
    fu_plot["start_dt"] = get_datetime_from_hours_since_epoch(fu_plot[START_COL])
    fu_plot["end_dt"] = get_datetime_from_hours_since_epoch(fu_plot[END_COL])

    # --- REFACTORED: Prepare death data for unified event plotting ---
    death_events = None
    if DEATH_COL in fu_plot.columns and fu_plot[DEATH_COL].notna().any():
        deaths_df = fu_plot[[PID_COL, DEATH_COL]].dropna().copy()
        deaths_df = deaths_df.rename(columns={DEATH_COL: ABSPOS_COL})
        death_events = deaths_df

    # --- PLOTTING SETUP ---
    height = max(3, 0.45 * len(subject_ids))
    fig, ax = plt.subplots(figsize=(12, height))

    y_pos, y_map = 0, {}
    for group in subject_groups:
        for pid in group:
            y_map[pid] = y_pos
            y_pos += 1

    # --- NEW: Add background shading for matched groups ---
    group_colors = ["#F0F0F0", "white"]  # Alternating light gray and white
    for i, group in enumerate(subject_groups):
        # Only add shading for actual groups (more than 1 person)
        if len(group) > 1:
            y_coords = [y_map[pid] for pid in group]
            # The span should cover the vertical space of the group
            ax.axhspan(
                min(y_coords) - 0.5,
                max(y_coords) + 0.5,
                color=group_colors[i % 2],
                zorder=0,  # zorder=0 sends it to the very back
                alpha=0.7,
            )

    date_range = fu_plot["end_dt"].max() - fu_plot["start_dt"].min()
    text_offset = date_range * 0.01

    # --- PLOT TIMELINE BARS ---
    for _, row in fu_plot.iterrows():
        y = y_map[int(row[PID_COL])]
        ax.hlines(
            y=y, xmin=row["start_dt"], xmax=row["end_dt"], color="lightgray", lw=2
        )

        pid = int(row[PID_COL])
        patient_type = ""
        if not index_date_matching.empty:
            if pid in index_date_matching["exposed_subject_id"].values:
                patient_type = " (E)"
            elif pid in index_date_matching["control_subject_id"].values:
                patient_type = " (C)"

        ax.text(
            row["end_dt"] + text_offset,
            y,
            f"{pid}{patient_type}",
            va="center",
            ha="left",
            fontsize=8,
            color="gray",
        )
        ax.text(row["start_dt"], y, "[", va="center", ha="right", fontsize=12)
        ax.text(row["end_dt"], y, "]", va="center", ha="left", fontsize=12)

    # --- NEW: Plot censor dates ---
    _plot_censor_dates(ax, censor_dates, y_map)

    # --- REFACTORED: Unified Event Plotting Function ---
    def add_events(
        df: pd.DataFrame,
        label: str,
        color: str,
        marker: str,
        size: int,
        alpha: float = 0.9,
        outside_alpha: float = 0.6,
    ):
        if df is None or df.empty or ABSPOS_COL not in df.columns:
            return
        tmp = df[df[PID_COL].isin(subject_ids)].copy()
        if tmp.empty:
            return

        tmp["time_dt"] = get_datetime_from_hours_since_epoch(tmp[ABSPOS_COL])
        merged = tmp.merge(
            fu_plot[[PID_COL, "start_dt", "end_dt"]], on=PID_COL, how="left"
        )

        for _, r in merged.iterrows():
            is_within = (
                (r["start_dt"] <= r["time_dt"] <= r["end_dt"])
                if pd.notna(r["start_dt"])
                else False
            )
            ax.scatter(
                r["time_dt"],
                y_map[int(r[PID_COL])],
                color=color,
                marker=marker,
                s=size,
                alpha=alpha if is_within else outside_alpha,
                label=label,
                zorder=5,
            )

    # --- PLOT ALL EVENTS ---
    # Define colors
    outcome_palette = {}
    if outcomes:
        palette = sns.color_palette("Set2", n_colors=len(outcomes))
        for i, name in enumerate(sorted(outcomes.keys())):
            outcome_palette[f"Outcome: {name}"] = palette[i]
    if outcome_colors:
        outcome_palette.update(outcome_colors)

    # Plot exposures, outcomes, and deaths
    add_events(
        exposures,
        "Exposure",
        color="grey",
        marker=".",
        size=50,
        alpha=0.9,
        outside_alpha=0.5,
    )
    if outcomes:
        for name, df in sorted(outcomes.items()):
            label = f"Outcome: {name}"
            add_events(
                df,
                label,
                color=outcome_palette.get(label, "C1"),
                marker="o",
                size=40,
                alpha=1.0,
                outside_alpha=1.0,
            )

    add_events(
        death_events,
        "Death",
        color="black",
        marker="x",
        size=80,
        alpha=1.0,
        outside_alpha=0.9,
    )

    # --- FINALIZE PLOT ---
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Events",
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = join(save_dir, "follow_ups_timeline.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {path}")
    else:
        plt.show()

    plt.close(fig)
