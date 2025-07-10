import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Labels and order
outcomes = ["death", "MI", "stroke"]
methods = ["IPW", "TMLE", "unadjusted"]
colors = ["tab:blue", "tab:orange", "tab:green"]
x = np.arange(len(outcomes))
offsets = [-0.2, 0, 0.2]  # for methods

# Corresponding effect_1 and effect_0 data
effect_1 = [
    [0.01753, 0.00963, 0.01394],  # IPW
    [0.02773, 0.06245, 0.03584],  # TMLE
    [0.01734, 0.00964, 0.01395],  # unadjusted
]
effect_0 = [
    [0.04635, 0.01097, 0.02312],  # IPW
    [0.05492, 0.06964, 0.04444],  # TMLE
    [0.07662, 0.00990, 0.02255],  # unadjusted
]

fig, ax = plt.subplots(figsize=(9, 5))

bar_width = 0.1
for i, (method, color, dx) in enumerate(zip(methods, colors, offsets)):
    ax.bar(
        x + dx - bar_width / 2,
        [row[i] * 100 for row in zip(*effect_0)],
        width=bar_width,
        color=color,
        alpha=0.5,
        label=f"{method} - control",
    )
    ax.bar(
        x + dx + bar_width / 2,
        [row[i] * 100 for row in zip(*effect_1)],
        width=bar_width,
        color=color,
        alpha=1.0,
        label=f"{method} - treated",
    )

ax.set_xticks(x)
ax.set_xticklabels(outcomes)
ax.set_ylabel("Estimated risk in %")
ax.set_title("Estimated risk by treatment and method")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Add sample size information
ax.text(
    0.98,
    0.98,
    "N = 22,035",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
)
ax.text(
    0.98,
    0.92,
    r"n$_{\mathrm{treated}}$ = 16,309",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
)
ax.text(
    0.98,
    0.86,
    r"n$_{\mathrm{control}}$ = 5,726",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
)

fig.savefig(
    "outputs/figs/semaglutide/effect_estimates_tte_differences.png",
    bbox_inches="tight",
    dpi=200,
)


# Table 2: Outcomes by treatment group with totals
outcomes = ["Death", "MI", "Stroke"]
unadj_treated_pct = [1.734, 0.964, 1.395]  # death, MI, stroke
unadj_control_pct = [7.662, 0.990, 2.255]  # death, MI, stroke

# Calculate actual numbers
treated_outcomes = [int(pct / 100 * 16309) + 1 for pct in unadj_treated_pct]
control_outcomes = [int(pct / 100 * 5726) + 1 for pct in unadj_control_pct]
total_outcomes = [t + c for t, c in zip(treated_outcomes, control_outcomes)]
total_pct = [n / 22035 * 100 for n in total_outcomes]

outcomes_table = pd.DataFrame(
    {
        "Outcome": outcomes,
        "Treated (n)": treated_outcomes,
        "Treated (%)": [f"{pct:.2f}" for pct in unadj_treated_pct],
        "Control (n)": control_outcomes,
        "Control (%)": [f"{pct:.2f}" for pct in unadj_control_pct],
        "Total (n)": total_outcomes,
        "Total (%)": [f"{pct:.2f}" for pct in total_pct],
    }
)

print("Outcomes by Treatment Group:")
print(outcomes_table.to_string(index=False))


# Table 1: Overall cohort composition
cohort_table = pd.DataFrame(
    {"Total N": [22035], "Treated N": [16309], "Control N": [5726]}
)
print("Study Population:")
print(cohort_table.to_string(index=False))
