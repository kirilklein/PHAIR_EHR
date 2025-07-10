import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


outcomes = ["death", "MI", "heart failure", "stroke"]
methods = ["IPW", "TMLE", "unadjusted"]
colors = ["tab:blue", "tab:orange", "tab:green"]
x = np.arange(len(outcomes))
offsets = [-0.15, 0, 0.15]

# Each row: [IPW, TMLE, unadjusted] for each outcome, rounded appropriately!
effects = [
    [-0.026, -0.0017, -0.0004, -0.0002],  # IPW
    [-0.026, -0.0192, -0.0003, 0.0016],  # TMLE
    [-0.059, -0.0003, -0.0003, -0.0033],  # unadjusted
]
lows = [
    [-0.032, -0.0040, -0.0013, -0.0023],
    [-0.033, -0.0251, -0.0011, -0.0042],
    [-0.063, -0.0014, -0.0007, -0.0049],
]
ups = [
    [-0.022, 0.0002, 0.0001, 0.0023],
    [-0.020, -0.0131, 0.0005, 0.0067],
    [-0.056, 0.0009, 0.0001, -0.0017],
]
stds = [
    [0.0025, 0.0013, 0.0004, 0.0011],  # IPW
    [0.0034, 0.0034, 0.0004, 0.0029],  # TMLE
    [0.0020, 0.0006, 0.0002, 0.0008],  # unadjusted
]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
for i, method in enumerate(methods):
    effect = np.array(effects[i])
    low = np.array(lows[i])
    up = np.array(ups[i])

    ax.errorbar(
        x + offsets[i],
        effect,
        yerr=[effect - low, up - effect],
        fmt="o",
        color=colors[i],
        label=method,
        capsize=5,
        markersize=7,
        linestyle="",
    )

ax.axhline(0, color="grey", lw=1, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(outcomes, fontsize=11)
ax.set_ylabel("Effect size", fontsize=12)
ax.set_xlabel("Outcome", fontsize=12)
ax.legend(title="Estimation type", fontsize=11)
ax.set_title("Effect estimates for Semaglutide by outcome and method", fontsize=12)

fig.savefig(
    "outputs/figs/semaglutide/effect_estimates_sgl.png", bbox_inches="tight", dpi=200
)
plt.show()
