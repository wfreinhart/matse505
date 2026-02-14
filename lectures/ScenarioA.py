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
# # Linear Regression Intro

# %% [markdown]
# # Linear Data Generation
# Generating synthetic linear data.

# %%
import numpy as np
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 1, 100)

# %% [markdown]
# # Linear Regression Model
# Initializing a Scikit-Learn Linear Regression model.

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# %% [markdown]
# # Generic Training
# Training the model on the generated tensors.

# %%
# Uses variables from global state: model, X, y
model.fit(X, y)
score = model.score(X, y)
print(f"R2 Score: {score:.3f}")

# %% [markdown]
# # Regression Visualization
# Plotting the data and model predictions.

# %%
import matplotlib.pyplot as plt
# Uses global state: model, X, y
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, model.predict(X), color='red', label='Prediction')
plt.legend()
plt.show()
