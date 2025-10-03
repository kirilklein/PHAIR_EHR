import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np


def parse_experiment_name(exp_name: str) -> Dict[str, float]:
    """Parse experiment name to extract parameter values, including negative 'm' values."""
    params = {}
    patterns = {
        "ce": r"ce(m?\d+(?:p\d+)?)",
        "cy": r"cy(m?\d+(?:p\d+)?)",
        "y": r"y(\d+(?:p\d+)?)",
        "i": r"i(\d+(?:p\d+)?)",
    }
    for param, pattern in patterns.items():
        match = re.search(pattern, exp_name)
        if match:
            value_str = match.group(1).replace("m", "-").replace("p", ".")
            params[param] = float(value_str)
        else:
            params[param] = 0.0
    return params


def load_and_process_results(
    results_dir: str, experiment_names: Optional[list] = None
) -> pd.DataFrame:
    """Loads all data and calculates bias, coverage, relative bias, and z-score."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = []
    run_dirs = [
        d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        print("No 'run_XX' directories found. Treating results_dir as a single run.")
        run_dirs = [results_path]

    for run_dir in run_dirs:
        exp_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if experiment_names:
            exp_dirs = [d for d in exp_dirs if d.name in experiment_names]

        for exp_dir in exp_dirs:
            possible_paths = [
                exp_dir / "estimate" / "estimate_results.csv",
                exp_dir / "estimate" / "baseline" / "estimate_results.csv",
                exp_dir / "estimate" / "bert" / "estimate_results.csv",
            ]
            results_file = next(
                (path for path in possible_paths if path.exists()), None
            )

            if not results_file:
                continue

            try:
                df = pd.read_csv(results_file)
                df["run_id"] = run_dir.name
                params = parse_experiment_name(exp_dir.name)
                for param, value in params.items():
                    df[param] = value

                df["bias"] = df["effect"] - df["true_effect"]
                df["covered"] = (df["true_effect"] >= df["CI95_lower"]) & (
                    df["true_effect"] <= df["CI95_upper"]
                )
                df["relative_bias"] = (df["bias"] / df["true_effect"]).replace(
                    [np.inf, -np.inf], np.nan
                )
                df["z_score"] = (df["bias"] / df["std_err"]).replace(
                    [np.inf, -np.inf], np.nan
                )
                all_results.append(df)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    if not all_results:
        raise ValueError("No valid results found to process.")

    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(combined_df)} total rows from {len(run_dirs)} runs.")
    return combined_df
