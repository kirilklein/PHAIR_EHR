"""
Script to compare mean absolute standardized mean differences (SMD) across multiple experiments.

Loads weighted and unweighted statistics from multiple paths, computes the mean absolute SMD
for each, and creates a comparison plot.
"""

import argparse
import logging
from os.path import join
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants for column names
EXPOSED = "Exposed"
CONTROL = "Control"

logger = logging.getLogger(__name__)


def compute_smd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the standardized mean difference for a dataframe between Exposed and Control.
    
    Args:
        df: DataFrame with EXPOSED and CONTROL columns (proportions, not percentages)
        
    Returns:
        DataFrame with added 'smd' column
    """
    p1 = df[EXPOSED]
    p0 = df[CONTROL]
    pooled_sd = ((p1 * (1 - p1) + p0 * (1 - p0)) / 2) ** 0.5
    # Handle division by zero when pooled_sd is 0
    df = df.copy()
    df['smd'] = np.where(pooled_sd > 0, (p1 - p0) / pooled_sd, 0)
    return df


def load_stats_from_paths(
    unweighted_path: str,
    weighted_path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load weighted and unweighted binary stats from file paths/URIs.
    
    Args:
        unweighted_path: Path/URI to unweighted stats CSV file
        weighted_path: Path/URI to weighted stats CSV file
        
    Returns:
        Tuple of (unweighted_stats, weighted_stats) DataFrames
    """
    try:
        unweighted_stats = pd.read_csv(unweighted_path)
        logger.info(f"Loaded unweighted stats from {unweighted_path}: {len(unweighted_stats)} criteria")
    except Exception as e:
        logger.warning(f"Failed to load unweighted stats from {unweighted_path}: {e}")
        unweighted_stats = None
    
    try:
        weighted_stats = pd.read_csv(weighted_path)
        logger.info(f"Loaded weighted stats from {weighted_path}: {len(weighted_stats)} criteria")
    except Exception as e:
        logger.warning(f"Failed to load weighted stats from {weighted_path}: {e}")
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


def compare_smd_from_dataframes(
    weighted_stats_list: List[pd.DataFrame],
    labels: List[float],
    colors: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    plot_filename: str = "smd_comparison.png",
    save_plot: bool = True,
):
    """
    Compare mean absolute SMD across multiple experiments from DataFrames.
    
    Args:
        weighted_stats_list: List of weighted stats DataFrames
        labels: List of outcome loss weights (x-axis values)
        colors: Optional list of colors for each point (default: all blue)
        output_path: Directory to save the plot (required if save_plot=True)
        plot_filename: Name of the output plot file
        save_plot: Whether to save the plot to file
        
    Returns:
        Tuple of (results_df, fig, ax) - DataFrame with results and matplotlib figure/axes
    """
    if len(weighted_stats_list) != len(labels):
        raise ValueError(
            f"Number of weighted stats ({len(weighted_stats_list)}) must match number of labels ({len(labels)})"
        )
    
    if colors is not None and len(colors) != len(labels):
        raise ValueError(
            f"Number of colors ({len(colors)}) must match number of labels ({len(labels)})"
        )
    
    # Default to blue if no colors provided
    if colors is None:
        colors = ['steelblue'] * len(labels)
    
    results = []
    
    for weighted_stats, label in zip(weighted_stats_list, labels):
        logger.info(f"Processing {label}")
        
        # Compute mean absolute SMD for weighted
        weighted_smd = compute_mean_absolute_smd(weighted_stats)
        
        results.append({
            "label": label,
            "weighted_smd": weighted_smd,
        })
        
        logger.info(f"{label}: Weighted mean |SMD| = {weighted_smd:.4f}")
    
    # Create DataFrame for easier handling
    results_df = pd.DataFrame(results)
    
    # Sort by label value if labels are numeric for better visualization
    if all(isinstance(l, (int, float)) for l in labels):
        # Create a sorted version with corresponding colors
        sorted_data = sorted(zip(labels, results_df["weighted_smd"], colors))
        sorted_labels = [x[0] for x in sorted_data]
        sorted_smd = [x[1] for x in sorted_data]
        sorted_colors = [x[2] for x in sorted_data]
    else:
        sorted_labels = labels
        sorted_smd = results_df["weighted_smd"].tolist()
        sorted_colors = colors
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot scatter
    for label, smd, color in zip(sorted_labels, sorted_smd, sorted_colors):
        if not np.isnan(smd):
            ax.scatter(
                label,
                smd,
                s=100,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=1,
            )
            # Add value labels on points
            ax.text(
                label,
                smd,
                f' {smd:.3f}',
                ha="left",
                va="bottom",
                fontsize=9,
            )
    
    ax.set_xlabel("Outcome Loss Weight")
    ax.set_ylabel("Mean Absolute Standardized Mean Difference (|SMD|)")
    ax.set_title("Mean Absolute SMD vs Outcome Loss Weight")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if output_path is None:
            raise ValueError("output_path is required when save_plot=True")
        output_file = join(output_path, plot_filename)
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
        logger.info(f"Plot saved to {output_file}")
        
        # Save results to CSV
        csv_output = join(output_path, "smd_comparison_results.csv")
        results_df.to_csv(csv_output, index=False)
        logger.info(f"Results saved to {csv_output}")
    
    return results_df, fig, ax


def compare_smd_from_paths(
    weighted_paths: List[str],
    labels: List[float],
    colors: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    plot_filename: str = "smd_comparison.png",
    save_plot: bool = True,
):
    """
    Compare mean absolute SMD across multiple experiments from file paths/URIs.
    
    Args:
        weighted_paths: List of paths/URIs to weighted stats CSV files
        labels: List of outcome loss weights (x-axis values)
        colors: Optional list of colors for each point (default: all blue)
        output_path: Directory to save the plot (required if save_plot=True)
        plot_filename: Name of the output plot file
        save_plot: Whether to save the plot to file
        
    Returns:
        Tuple of (results_df, fig, ax) - DataFrame with results and matplotlib figure/axes
    """
    if len(weighted_paths) != len(labels):
        raise ValueError(
            f"Number of weighted paths ({len(weighted_paths)}) must match number of labels ({len(labels)})"
        )
    
    weighted_stats_list = []
    
    for weighted_path, label in zip(weighted_paths, labels):
        logger.info(f"Loading stats for {label}")
        _, weighted_stats = load_stats_from_paths("", weighted_path)  # Unweighted path not needed
        weighted_stats_list.append(weighted_stats)
    
    return compare_smd_from_dataframes(
        weighted_stats_list,
        labels,
        colors,
        output_path,
        plot_filename,
        save_plot,
    )


def main():
    """Main function to run the comparison from command line."""
    parser = argparse.ArgumentParser(
        description="Compare mean absolute SMD across multiple experiment paths"
    )
    parser.add_argument(
        "--weighted-paths",
        nargs="+",
        required=True,
        help="Paths/URIs to weighted stats CSV files (space-separated)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=float,
        required=True,
        help="Outcome loss weights for each experiment (space-separated, must match number of paths)",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="Colors for each point (space-separated, optional)",
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
    results_df, fig, ax = compare_smd_from_paths(
        weighted_paths=args.weighted_paths,
        labels=args.labels,
        colors=args.colors,
        output_path=args.output,
        plot_filename=args.plot_name,
        save_plot=True,
    )
    
    print("\nResults Summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
