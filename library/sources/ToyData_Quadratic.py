# %% [markdown]
# ---
# id: ToyData_Quadratic
# type: Data
# ---
# # Quadratic Data Generation
# Generating synthetic quadratic data.

# %%
import numpy as np
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 0.5 * X.flatten()**2 + np.random.normal(0, 1, 100)
