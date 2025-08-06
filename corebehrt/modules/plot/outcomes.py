from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from os.path import join
import os
from corebehrt.constants.data import PID_COL, TIMESTAMP_COL
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    max_outcomes_to_plot: Optional[int] = None
    num_rows: Optional[int] = None
    max_subplots_per_figure: int = 9
    wide_dist_threshold: int = 30


class OutcomePlotter:
    def __init__(self, outcomes_path: str, figures_path: str, config: PlotConfig):
        self.outcomes_path = outcomes_path
        self.figures_path = figures_path
        os.makedirs(figures_path, exist_ok=True)
        self.config = config

        # --- State variables for iterative plotting ---
        # Each list will hold tuples of (figure, axes)
        self.patient_dist_figures: List[Tuple[Figure, np.ndarray]] = []
        self.time_dist_figures: List[Tuple[Figure, np.ndarray]] = []

        # Counters to track the next available subplot index
        self.patient_subplot_idx = 0
        self.time_subplot_idx = 0

    def run(self):
        """
        Main method to run the memory-efficient, iterative plotting pipeline.
        """
        logger.info(f"Scanning for outcome files in {self.outcomes_path}...")
        try:
            outcome_files = sorted(
                [f for f in os.listdir(self.outcomes_path) if f.endswith(".csv")]
            )
        except FileNotFoundError:
            logger.error(f"Could not find outcomes path: {self.outcomes_path}")
            return

        if not outcome_files:
            logger.warning("No outcome CSV files found.")
            return

        if (
            self.config.max_outcomes_to_plot is not None
            and len(outcome_files) > self.config.max_outcomes_to_plot
        ):
            logger.info(
                f"Found {len(outcome_files)} outcomes. Limiting to first {self.config.max_outcomes_to_plot}."
            )
            outcome_files = outcome_files[: self.config.max_outcomes_to_plot]

        logger.info(f"Begin iterative plotting for {len(outcome_files)} outcomes...")
        # --- Main loop: Process one file at a time ---
        for filename in outcome_files:
            outcome_name = os.path.splitext(filename)[0]
            try:
                # 1. Load ONE dataframe
                df = pd.read_csv(
                    join(self.outcomes_path, filename),
                    nrows=self.config.num_rows,
                    parse_dates=[TIMESTAMP_COL],
                )
                if df.empty:
                    logger.info(f"Skipping empty outcome file: {filename}")
                    continue

                # 2. Add subplots for this dataframe
                if PID_COL in df.columns:
                    counts = df[PID_COL].value_counts()
                    self._add_patient_dist_subplot(outcome_name, counts)

                if TIMESTAMP_COL in df.columns:
                    timestamps = df[TIMESTAMP_COL].dropna()
                    self._add_time_dist_subplot(outcome_name, timestamps)

                # 3. The 'df' is now out of scope and will be garbage collected

            except Exception as e:
                logger.error(f"Failed to process and plot {filename}. Error: {e}")
                continue

        # 4. After processing all files, save the completed figures
        self._finalize_all_figures()

    def _add_patient_dist_subplot(self, outcome_name: str, counts: pd.Series):
        """Adds a single patient distribution subplot to the correct figure."""
        figure_idx = self.patient_subplot_idx // self.config.max_subplots_per_figure

        # Create a new figure if needed
        if figure_idx >= len(self.patient_dist_figures):
            fig, axes = self._create_subplot_grid()
            self.patient_dist_figures.append((fig, axes))

        # Get the correct figure and axis
        fig, axes = self.patient_dist_figures[figure_idx]
        axis_idx = self.patient_subplot_idx % self.config.max_subplots_per_figure
        ax = axes[axis_idx]

        # Plot the data
        max_count = counts.max()
        if max_count > self.config.wide_dist_threshold:
            ax.hist(counts, bins=25, color="teal", alpha=0.7)
            ax.set_title(f"{outcome_name} (Wide Dist.)", fontsize=12)
        else:
            bins = np.arange(1, max_count + 3) - 0.5
            ax.hist(counts, bins=bins, color="teal", alpha=0.7)
            ax.set_title(f"{outcome_name}", fontsize=12)
            if max_count > 0:
                ax.set_xticks(np.arange(1, max_count + 1))
        ax.set_xlabel("Events per Patient")
        ax.set_ylabel("Number of Patients")
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        self.patient_subplot_idx += 1

    def _add_time_dist_subplot(self, outcome_name: str, timestamps: pd.Series):
        """Adds a single time distribution subplot to the correct figure."""
        figure_idx = self.time_subplot_idx // self.config.max_subplots_per_figure

        if figure_idx >= len(self.time_dist_figures):
            fig, axes = self._create_subplot_grid()
            self.time_dist_figures.append((fig, axes))

        fig, axes = self.time_dist_figures[figure_idx]
        axis_idx = self.time_subplot_idx % self.config.max_subplots_per_figure
        ax = axes[axis_idx]

        if not timestamps.empty:
            ax.hist(timestamps, bins="auto", color="purple", alpha=0.7)
        ax.set_title(outcome_name, fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Events")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        self.time_subplot_idx += 1

    def _create_subplot_grid(self) -> Tuple[Figure, np.ndarray]:
        """Helper to create a standard figure and a grid of subplots."""
        n = self.config.max_subplots_per_figure
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 5.5, rows * 4), squeeze=False
        )
        return fig, axes.flatten()

    def _finalize_all_figures(self):
        """Cleans up and saves all generated figures."""
        logger.info("Finalizing and saving figures...")
        # Finalize patient distribution plots
        out_dir = join(self.figures_path, "outcomes_per_patient")
        os.makedirs(out_dir, exist_ok=True)
        for i, (fig, axes) in enumerate(self.patient_dist_figures):
            self._finalize_and_save_figure(
                fig,
                i,
                len(self.patient_dist_figures),
                "Distribution of Events per Patient",
                join(out_dir, f"patient_dist_summary_{i + 1}.png"),
            )

        # Finalize time distribution plots
        out_dir = join(self.figures_path, "outcomes_over_time")
        os.makedirs(out_dir, exist_ok=True)
        for i, (fig, axes) in enumerate(self.time_dist_figures):
            self._finalize_and_save_figure(
                fig,
                i,
                len(self.time_dist_figures),
                "Distribution of Outcomes Over Time",
                join(out_dir, f"time_dist_summary_{i + 1}.png"),
            )

    def _finalize_and_save_figure(
        self, fig: Figure, page_index: int, total_pages: int, title: str, save_path: str
    ):
        """Helper to set titles, clean up layout, and save a single figure."""
        for ax in fig.axes:
            # Hide any axis that wasn't plotted on
            if not ax.has_data():
                ax.set_visible(False)

        fig.suptitle(f"{title} (Part {page_index + 1}/{total_pages})", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved figure: {save_path}")
