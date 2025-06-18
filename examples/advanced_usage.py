"""
basic_usage.py

Example usage for the sequential change-point detection toolbox
using SIMULATED DATA with MULTIPLE change points.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from package_folder import (
    wrapper_func,
    generate_data_with_change_points
)

# -------------------------------
# 1) Simulate data with multiple change points
# -------------------------------

N = 100               # training period length
total_length = 1000   # total data length
delta = 0.1
num_monte_carlos = 20  # keep small for quick run

# Example: 3 change points at indices 250, 500, 750
cp_locs = [250, 500, 750]
mean_shifts = [0, 2, -1, 2]  # must be num_cp + 1 elements

data = generate_data_with_change_points(
    length=total_length,
    num_cp=3,
    loc=cp_locs,
    mean_shifts=mean_shifts,
    seed=42  # for reproducibility
)

print(f"Simulated data with 3 change points at {cp_locs} and shifts {mean_shifts}.")

# -------------------------------
# 2) Run detection + asymptotics
# -------------------------------

test_stat, asymptotics, theta_set, gamma_signed, r_run, long_run = wrapper_func(
    N=N,
    total_data=total_length,
    delta_fixed=delta,
    num_monte_carlos=num_monte_carlos,
    data=data,
    level=0.95
)

print("Detected change points (theta_set):", theta_set)

# -------------------------------
# 3) Plot: data + true CPs + detected CPs + delays + scanning statistic
# -------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --------------------------------
# TOP PLOT: Data + CPs
# --------------------------------

ax1.plot(data, label='Simulated Data', color='blue')
#ax1.set_title("Simulated Data with True & Detected Change Points")
ax1.set_ylabel("Simulated Data with True & Detected Change Points")

# Mark training end
ax1.axvline(x=N, color='green', linestyle='-', linewidth=2, label='Training End (N)')

# Mark TRUE change points (orange dashed) + label ABOVE line
for cp in cp_locs:
    ax1.axvline(x=cp, color='orange', linestyle='--', linewidth=2)
    ax1.text(
        cp, 
        ax1.get_ylim()[1] * 0.98,  # a bit below the top
        f'True CP: {cp}',
        color='orange',
        va='top', ha='center',
        fontsize=10
    )

# Mark detected change points (red dotted) + label BELOW line
for t in theta_set:
    if 0 <= t < len(data):
        ax1.axvline(x=t+1, color='red', linestyle='dotted', linewidth=2)
        ax1.text(
            t,
            ax1.get_ylim()[0] + 0.05*(ax1.get_ylim()[1] - ax1.get_ylim()[0]),  # near bottom
            f'Detected: {t+1}',
            color='red',
            va='bottom', ha='center',
            fontsize=10
        )

# Custom legend handles
legend_handles = [
    Line2D([0], [0], color='blue', label='Simulated Data'),
    Line2D([0], [0], color='orange', linestyle='--', label='True Change Points'),
    Line2D([0], [0], color='red', linestyle='dotted', label='Detected Change Points'),
    Line2D([0], [0], color='green', linestyle='-', label='Training Period End (N)')
]

ax1.legend(handles=legend_handles)

# --------------------------------
# BOTTOM PLOT: Scanning stat + delays
# --------------------------------

# Detection delays (purple dashed)
for t in theta_set:
    delayed = r_run.get(t)
    if delayed and 0 <= delayed < len(data):
        ax2.axvline(x=delayed, color='purple', linestyle='--', linewidth=2)
        ax2.text(delayed, ax2.get_ylim()[0], f'Delay {delayed}',
                 color='purple', fontsize=10, va='bottom', ha='center')

# Scanning statistic (black)
keys = sorted(test_stat.keys())
values = [-1 * test_stat[k] for k in keys]
ax2.plot(keys, values, color='black', label=r"$\Delta_{\max}(k)$")

# Mark training end
ax2.axvline(x=N, color='green', linestyle='-', linewidth=2)

ax2.set_title("Scanning Statistic and Detection Delays")
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\Delta_{\max}(k)$")

# Custom legend for bottom
ax2.legend(handles=[
    Line2D([0], [0], color='black', label=r'$\Delta_{\max}(k)$'),
    Line2D([0], [0], color='purple', linestyle='--', label='Detection Delays'),
    Line2D([0], [0], color='green', linestyle='-', label='Training End (N)')
])


plt.tight_layout()
plt.show()
