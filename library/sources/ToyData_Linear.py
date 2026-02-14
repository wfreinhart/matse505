# %% [markdown]
# ---
# id: ToyData_Linear
# type: Data
# ---
# # Linear Data Generation
# Generating synthetic linear data.

# %%
import numpy as np
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 1, 100)
