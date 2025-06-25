# sequential_relevant_cp

**sequential_relevant_cp** 
**is a Python toolbox for detecting change points in time series data using CUSUM, iterative algorithms, Brownian motion simulations, etc. corresponding to the methods presented in the paper ....
**---

## 📦 Installation

Clone the repo and install locally:

```bash
git clone https://github.com/rupsabasu2020/sequential_relevant_cp.git
cd my_package
pip install .




## 🚀 Quick Example

```python
import numpy as np
from package_folder import wrapper_func, generate_data_with_change_points

# Example: generate synthetic time series data with a change point
data = generate_data_with_change_points(length=500, num_cp=1, loc=[250], mean_shifts=[0, 2])

# Detect change points and run test
test_stat, asymptotics, theta_set, gamma_signed, r_run, long_run = wrapper_func(
    N=100,
    total_data=500,
    delta_fixed=0.1,
    num_monte_carlos=100,
    data=data
)

print("Detected change points:", theta_set)
