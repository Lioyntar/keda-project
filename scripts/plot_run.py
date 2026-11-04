"""
Γράφημα ενός πειράματος του generator: μήκος ουράς και ρυθμός στόχος στον χρόνο.
Χρήση:  python scripts/plot_run.py results/smoke_20261102_184414.csv
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

path = sys.argv[1]
d = pd.read_csv(path)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(d.t, d.queue_len, color="#D1495B", linewidth=2, label="μήκος ουράς (jobs)")
ax1.set_xlabel("χρόνος (s)")
ax1.set_ylabel("μήκος ουράς (jobs)", color="#D1495B")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(d.t, d.target_rate, color="#F4A23B", linewidth=2, linestyle="--", label="ρυθμός στόχος (jobs/s)")
ax2.set_ylabel("ρυθμός στόχος (jobs/s)", color="#F4A23B")

# σκίαση των φάσεων
for phase, g in d.groupby("phase", sort=False):
    ax1.axvspan(g.t.min(), g.t.max(), alpha=0.06, color="gray")
    ax1.text((g.t.min() + g.t.max()) / 2, ax1.get_ylim()[1] * 0.95, phase, ha="center", color="gray")

lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
plt.title("Smoke test με 2 σταθερούς workers")
plt.tight_layout()

out = os.path.splitext(path)[0] + "_queue.png"
plt.savefig(out, dpi=150)
print("Saved", out)
