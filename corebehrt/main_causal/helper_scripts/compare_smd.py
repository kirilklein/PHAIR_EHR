"""
Script to compare mean absolute standardized mean differences (SMD) across multiple experiments.

Loads weighted and unweighted statistics from multiple paths, computes the mean absolute SMD
for each, and creates a comparison plot.
"""

import argparse
import logging
from os.path import join
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corebehrt.constants.causal.paths import (
    STATS_RAW_FILE_BINARY,
)
from corebehrt.constants.causal.stats import CONTROL, EXPOSED, RAW
from corebehrt.functional.utils.azure_save import save_figure_with_azure_copy
from corebehrt.main_causal.helper_scripts.helper.get_stat import compute_smd

logger = logging.getLogger(__name__)


def load_stats_from_path(stats_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load weighted and unweighted binary stats from a stats directory.
    
    Args:
        stats_path: Path to directory containing stats files
        
    Returns:
        Tuple of (unweighted_stats, weighted_stats) DataFrames
    """
    unweighted_path = join(stats_path, STATS_RAW_FILE_BINARY)
    weighted_path = join(stats_path, "weighted_" + STATS_RAW_FILE_BINARY)
    
    try:
        unweighted_stats = pd.read_csv(unweighted_path)
        logger.info(f"Loaded unweighted stats from {unweighted_path}: {len(unweighted_stats)} criteria")
    except FileNotFoundError:
        logger.warning(f"Unweighted stats not found at {unweighted_path}")
        unweighted_stats = None
    
    try:
        weighted_stats = pd.read_csv(weighted_path)
        logger.info(f"Loaded weighted stats from {weighted_path}: {len(weighted_stats)} criteria")
    except FileNotFoundError:
        logger.warning(f"Weighted stats not found at {weighted_path}")
        weighted_stats = None
    
    return unweighted_stats, weighted_stats


def compute_mean_absolute_smd(stats_df: pd.DataFrame) -> float:
    """
    Compute the mean absolute standardized mean difference from stats DataFrame.
    
    Args:
        stats_df: DataFrame with columns: criterion, group, percentage
        
    Returns:
        Mean absolute SMD value
    """
    if stats_df is None or len(stats_df) == 0:
        return np.nan
    
    # Pivot to get exposed and control percentages
    pivot = stats_df.pivot(index="criterion", columns="group", values="percentage")
    
    # Convert percentages to proportions
    if EXPOSED in pivot.columns and CONTROL in pivot.columns:
        pivot[[EXPOSED, CONTROL]] = pivot[[EXPOSED, CONTROL]] / 100
    else:
        logger.warning(f"Missing required columns. Available: {pivot.columns.tolist()}")
        return np.nan
    
    # Compute SMD
    pivot_smd = compute_smd(pivot)
    
    # Compute mean absolute SMD
    mean_abs_smd = pivot_smd["smd"].abs().mean()
    
    return mean_abs_smd


def compare_smd_across_paths(
    paths: List[str],
    labels: List[str],
    output_path: str,
    plot_filename: str = "smd_comparison.png",
):
    """
    Compare mean absolute SMD across multiple experiment paths.
    
    Args:
        paths: List of paths to stats directories
        labels: List of labels for each path (y-axis labels)
        output_path: Directory to save the plot
        plot_filename: Name of the output plot file
    """
    if len(paths) != len(labels):
        raise ValueError(f"Number of paths ({len(paths)}) must match number of labels ({len(labels)})")
    
    results = []
    
    for path, label in zip(paths, labels):
        logger.info(f"Processing {label} from {path}")
        unweighted_stats, weighted_stats = load_stats_from_path(path)
        
        # Compute mean absolute SMD for unweighted
        unweighted_smd = compute_mean_absolute_smd(unweighted_stats)
        
        # Compute mean absolute SMD for weighted
        weighted_smd = compute_mean_absolute_smd(weighted_stats)
        
        results.append({
            "label": label,
            "unweighted_smd": unweighted_smd,
            "weighted_smd": weighted_smd,
        })
        
        logger.info(
            f"{label}: Unweighted mean |SMD| = {unweighted_smd:.4f}, "
            f"Weighted mean |SMD| = {weighted_smd:.4f}"
        )
    
    # Create DataFrame for easier handling
    results_df = pd.DataFrame(results)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))
    
    y_pos = np.arange(len(labels))
    width = 0.35
    
    # Plot bars
    bars1 = ax.barh(
        y_pos - width/2,
        results_df["unweighted_smd"],
        width,
        label="Unweighted",
        alpha=0.7,
    )
    bars2 = ax.barh(
        y_pos + width/2,
        results_df["weighted_smd"],
        width,
        label="Weighted",
        alpha=0.7,
    )
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if not np.isnan(results_df.iloc[i]["unweighted_smd"]):
            ax.text(
                bar1.get_width(),
                bar1.get_y() + bar1.get_height() / 2,
                f'{results_df.iloc[i]["unweighted_smd"]:.3f}',
                ha="left",
                va="center",
                fontsize=9,
            )
        if not np.isnan(results_df.iloc[i]["weighted_smd"]):
            ax.text(
                bar2.get_width(),
                bar2.get_y() + bar2.get_height() / 2,
                f'{results_df.iloc[i]["weighted_smd"]:.3f}',
                ha="left",
                va="center",
                fontsize=9,
            )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Absolute Standardized Mean Difference (|SMD|)")
    ax.set_title("Comparison of Mean Absolute SMD Across Experiments")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = join(output_path, plot_filename)
    save_figure_with_azure_copy(fig, output_file, dpi=200)
    logger.info(f"Plot saved to {output_file}")
    
    # Save results to CSV
    csv_output = join(output_path, "smd_comparison_results.csv")
    results_df.to_csv(csv_output, index=False)
    logger.info(f"Results saved to {csv_output}")
    
    return results_df


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare mean absolute SMD across multiple experiment paths"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Paths to stats directories (space-separated)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels for each path (space-separated, must match number of paths)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--plot-name",
        default="smd_comparison.png",
        help="Name of the output plot file (default: smd_comparison.png)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run comparison
    results_df = compare_smd_across_paths(
        paths=args.paths,
        labels=args.labels,
        output_path=args.output,
        plot_filename=args.plot_name,
    )
    
    print("\nResults Summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
