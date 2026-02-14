# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # RealScenario_RF

# %% [markdown]
# # Concrete Compressive Strength Data
# Loading the concrete dataset from Kaggle.

# %%
import pandas as pd
import os

filename = 'concrete.csv'
local_path = '../datasets/' + filename

if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    # Fallback to local path relative to repo root if needed
    data = pd.read_csv('datasets/concrete.csv')

# Prepare features and target
x = data.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
y = data['Concrete compressive strength(MPa, megapascals) ']

print(f"Loaded concrete dataset: {x.shape}")

# %% [markdown]
# # Random Forest Regressor
# Random Forest with max_depth=5 as suggested in Lecture 03.

# %%
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=5, random_state=42)

# %% [markdown]
# # Training and Verification
# Training the model and plotting parity.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=42)

# Fit
model.fit(xtrain, ytrain)

# Evaluate
y_pred = model.predict(xtest)
residuals = y_pred - ytest
r2 = 1 - np.var(residuals) / np.var(ytest)
rmse = np.sqrt(np.mean(residuals**2))
print(f'Test Results: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

# Plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(ytrain, model.predict(xtrain), '.', label='Train')
ax.plot(ytest, y_pred, '.', label='Test')
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.set_title(f'Evaluation: {type(model).__name__}')
ax.legend()
plt.show()
