import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

df = pd.read_csv("outputs/openness_timeseries.csv")
df = df[df["Openness_yds"].notna() & (df["Openness_yds"] != "")]
df["Openness_yds"] = pd.to_numeric(df["Openness_yds"], errors="coerce")
df = df.dropna(subset=["Openness_yds"])

colors = {"R1": "#E63946", "R2": "#2196F3", "R3": "#4CAF50", "R4": "#FF9800"}

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

for receiver, group in df.groupby("Receiver"):
    group = group.sort_values("Time_Post_Snap_s")
    color = colors.get(receiver, "white")
    smoothed = gaussian_filter1d(group["Openness_yds"].values, sigma=2)
    ax.plot(
        group["Time_Post_Snap_s"],
        smoothed,
        color=color,
        linewidth=2.2,
        label=receiver,
    )

# Throw highlight band at t=2.14
throw_t = 2.14
ax.axvspan(throw_t - 0.04, throw_t + 0.04, color="white", alpha=0.12, zorder=0)
ax.axvline(x=throw_t, color="white", linestyle="--", linewidth=1.2, alpha=0.7, label="Throw (2.14s)")

ax.set_xlabel("Time Post Snap (s)", color="white", fontsize=12)
ax.set_ylabel("Openness (yds)", color="white", fontsize=12)
ax.set_title("Play 2 — Receiver Openness Over Time", color="white", fontsize=14, pad=14)

ax.tick_params(colors="white")
ax.spines["bottom"].set_color("#444")
ax.spines["left"].set_color("#444")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(color="#2A2A2A", linewidth=0.7)
ax.grid(which="minor", color="#1E1E1E", linewidth=0.4)

legend = ax.legend(
    facecolor="#1E1E1E",
    edgecolor="#444",
    labelcolor="white",
    fontsize=11,
    loc="upper left",
)

plt.tight_layout()
plt.savefig("outputs/openness_chart.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: outputs/openness_chart.png")
