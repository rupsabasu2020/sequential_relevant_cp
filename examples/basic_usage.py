"""
basic_usage.py

Example script showing how to use the sequential_relevant_cp toolbox.
"""

import numpy as np
from package_folder import (
    wrapper_func,
    generate_data_with_change_points
)

# ------------------------------
# 1) Generate synthetic time series with a known change point
# ------------------------------
data = generate_data_with_change_points(
    length=500,
    num_cp=1,
    loc=[250],
    mean_shifts=[0, 2]
)

print(f"Generated data of length {len(data)} with a change point at index 250.")

# ------------------------------
# 2) Run the change-point detection workflow
# ------------------------------
N = 100
delta = 0.1
num_monte_carlos = 50  # keep small for quick example

test_stat, asymptotics, theta_set, gamma_signed, r_run, long_run = wrapper_func(
    N=N,
    total_data=len(data),
    delta_fixed=delta,
    num_monte_carlos=num_monte_carlos,
    data=data
)

# print("Detected change points (theta_set):", theta_set)
# print("Test statistic keys:", list(test_stat.keys()))
# print("Asymptotic quantiles:", asymptotics)

# ------------------------------
# 3) Plot (optional, requires matplotlib)
# ------------------------------
import matplotlib.pyplot as plt

plt.plot(data, label="Data with change point")
for cp in theta_set:
    plt.axvline(cp, color="red", linestyle="--", label=f"Detected CP at {cp}")
plt.title("Example: Change Point Detection")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
