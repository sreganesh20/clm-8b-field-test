"""Renders assets/results.png: command-finder top-1 accuracy, BM25 vs CLM-8B vs raw encoder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = [("BM25 keyword search", "#2a78d6"), ("CLM-8B (zero-shot)", "#eb6834"), ("Raw Qwen3-8B embeddings", "#1baf7a")]
GROUPS = ["Search all 29,852\ncommands", "Right answers mixed with\n50 random commands", "Re-rank BM25's\ntop 50 results"]
VALUES = [[8, 18, 8],   # BM25
          [0, 13, 2],   # CLM-8B
          [0, 5, 0]]    # raw

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

bar_w, gap = 0.17, 0.013            # data units; gap ~= a 2px surface gap between touching bars
fig.subplots_adjust(left=0.06, right=0.98, top=0.74, bottom=0.17)
ax.set_xlim(-0.6, len(GROUPS) - 0.4); ax.set_ylim(0, 21)
bb = ax.get_window_extent()                                # axes size in pixels
x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / bb.width
y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / bb.height
ROUND_PX = 7                                               # rounded data-end radius in output pixels
rx, ry = ROUND_PX * x_per_px, ROUND_PX * y_per_px
for s, (name, color) in enumerate(SERIES):
    for g in range(len(GROUPS)):
        v = VALUES[s][g]
        x = g + (s - 1) * (bar_w + gap) - bar_w / 2
        if v > 0:
            ax.add_patch(FancyBboxPatch((x, 0), bar_w, v, boxstyle=f"round,pad=0,rounding_size={rx}",
                                        mutation_aspect=ry / rx, fc=color, ec="none", zorder=3))
            ax.add_patch(Rectangle((x, 0), bar_w, min(v, ry), fc=color, ec="none", zorder=3))   # square baseline
        ax.text(x + bar_w / 2, v + 0.35, f"{v}", ha="center", va="bottom", fontsize=14,
                fontweight="bold" if s == 1 else "normal", color=INK, zorder=4)

ax.set_xticks(range(len(GROUPS)), GROUPS, color=INK2, fontsize=13)
ax.set_yticks([0, 5, 10, 15, 20], ["0", "5", "10", "15", "20"], color=MUTED, fontsize=11)
ax.tick_params(length=0, pad=10)
ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)

fig.text(0.06, 0.93, "Zero-shot CLM-8B lost to keyword search at finding shell commands",
         fontsize=20, fontweight="bold", color=INK)
fig.text(0.06, 0.875, "Queries (of 20) where the top answer was a correct command, by what the model had to choose from",
         fontsize=13, color=INK2)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for _, c in SERIES]
fig.legend(handles, [n for n, _ in SERIES], loc="upper left", bbox_to_anchor=(0.055, 0.845), ncol=3,
           frameon=False, fontsize=12.5, labelcolor=INK2, handlelength=1.0, handleheight=1.0, columnspacing=2.2)
fig.text(0.06, 0.035, "20 hand-written queries · 29,852 tldr-pages examples · CLM-v0.1-8B as released, no fine-tuning · "
         "small test set, not a benchmark", fontsize=10.5, color=MUTED)
fig.savefig("assets/results.png", facecolor=SURFACE)
print("saved")
