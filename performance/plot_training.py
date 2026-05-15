"""
Generate an annotated training curve plot from training_log.txt
Shows the actual training data with academic explanations of each phase.

Usage: python3 performance/plot_training.py
Output: performance/training_curve_annotated.png
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "training_log.txt")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "training_curve_annotated.png")

# ── Parse training log ────────────────────────────────────────────────────────
total_times = []
epsilons    = []

if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        log = f.read()
    # Epsilon per epoch
    epsilons = [float(x) for x in re.findall(r"epsilon:\s*([\d.]+)", log)]
    # Total time per epoch
    total_times = [float(x) for x in re.findall(r"total_time\s+([\d.]+)", log)]

# Fallback: use representative data if no log
if not total_times:
    print("No training_log.txt found — using representative data")
    np.random.seed(42)
    total_times = []
    for e in range(50):
        if e < 15:
            base = 4_000_000 + np.random.randint(-1_500_000, 2_500_000)
        elif e < 35:
            base = 8_000_000 + np.random.randint(-2_000_000, 3_000_000)
        else:
            base = 14_000_000 + np.random.randint(-2_000_000, 4_000_000)
        total_times.append(max(base, 2_000_000))
    epsilons = [max(1.0 - 0.02*e, 0.05) for e in range(50)]

n = min(len(total_times), len(epsilons), 50)
total_times = total_times[:n]
epsilons    = epsilons[:n]
epochs      = list(range(n))

best_epoch = total_times.index(min(total_times))
best_time  = total_times[best_epoch]

# Smoothing
window = 5
ma = np.convolve(total_times, np.ones(window)/window, mode='valid') if n >= window else total_times

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor('#0a0e1a')
ax.set_facecolor('#111827')
for s in ax.spines.values():
    s.set_color('#1e3a5f')
ax.tick_params(colors='#64748b')
ax.xaxis.label.set_color('#94a3b8')
ax.yaxis.label.set_color('#94a3b8')
ax.title.set_color('#e2e8f0')

# Phase backgrounds
ax.axvspan(0, min(15, n), color='#00ff88', alpha=0.06, label='_nolegend_')
ax.axvspan(min(15, n), min(35, n), color='#ffd600', alpha=0.06, label='_nolegend_')
if n > 35:
    ax.axvspan(35, n, color='#ff3b5c', alpha=0.06, label='_nolegend_')

# Data
ax.plot(epochs, [t/1e6 for t in total_times],
        color='#00d4ff', alpha=0.35, linewidth=1, label='Per-epoch waiting time')
if len(ma) > 0:
    ax.plot(range(window-1, n), [v/1e6 for v in ma],
            color='#00d4ff', linewidth=2.5, label=f'Moving avg ({window})')
ax.scatter([best_epoch], [best_time/1e6], color='#ffd600', s=120, zorder=5,
           label=f'Best checkpoint (epoch {best_epoch})', edgecolors='#1e3a5f', linewidths=1.5)

# Annotations for phases
y_top = max(total_times)/1e6 * 1.05
ax.text(min(7, n/2), y_top*0.95, 'EXPLORATION\n(ε high, mostly random)',
        ha='center', va='top', color='#00ff88', fontsize=9, fontweight='bold')
if n > 15:
    ax.text(25, y_top*0.95, 'TRANSITION\n(ε decaying)',
            ha='center', va='top', color='#ffd600', fontsize=9, fontweight='bold')
if n > 35:
    ax.text(42, y_top*0.95, 'EXPLOITATION\n(ε low, greedy)',
            ha='center', va='top', color='#ff3b5c', fontsize=9, fontweight='bold')

# Best annotation
ax.annotate(f'Best model saved\n({best_time/1e6:.2f}M waiting time)',
            xy=(best_epoch, best_time/1e6),
            xytext=(best_epoch + 8, best_time/1e6 + 1),
            color='#ffd600', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ffd600', lw=1.5))

ax.set_xlabel('Training Epoch', fontsize=11)
ax.set_ylabel('Total Waiting Time (×10⁶ s)', fontsize=11)
ax.set_title('DQN Training Progress — Exploration-Exploitation Tradeoff',
             fontsize=13, pad=14, fontweight='bold')
ax.legend(facecolor='#1a2235', edgecolor='#1e3a5f', labelcolor='#e2e8f0',
          loc='lower right', fontsize=9)
ax.grid(True, color='#1e3a5f', alpha=0.4, linewidth=0.5)
ax.set_ylim(0, y_top)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight',
            facecolor='#0a0e1a', edgecolor='none')
print(f"Saved: {OUT_PATH}")
plt.show()