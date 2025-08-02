from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from corebehrt.constants.causal.data import END_COL, START_COL
from corebehrt.constants.data import PID_COL


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
