import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


outcomes = ["death", "MI", "stroke"]
methods = ["IPW", "TMLE", "unadjusted"]
colors = ["tab:blue", "tab:orange", "tab:green"]
x = np.arange(len(outcomes))
offsets = [-0.15, 0, 0.15]

# Each row: [IPW, TMLE, unadjusted] for each outcome, rounded appropriately!
effects = [
    [-0.0288, -0.0014, -0.0092],  # IPW
    [-0.0279, -0.0071, -0.0084],  # TMLE
    [-0.0593, -0.0003, -0.0086],  # unadjusted
]
lows = [
    [-0.0369, -0.0063, -0.0167],
    [-0.0369, -0.0144, -0.0178],
    [-0.0666, -0.0033, -0.0129],
]
ups = [
    [-0.0211, 0.0027, -0.0028],
    [-0.0185, -0.0002, -0.0017],
    [-0.0520, 0.0028, -0.0043],
]
stds = [
    [0.0042, 0.0024, 0.0037],  # IPW
    [0.0048, 0.0036, 0.0042],  # TMLE
    [0.0037, 0.0015, 0.0022],  # unadjusted
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
    "outputs/figs/semaglutide/effect_estimates_tte.png", bbox_inches="tight", dpi=200
)
plt.show()
