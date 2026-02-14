# %% [markdown]
# ---
# id: RealWorkflow_Verify
# type: Workflow
# ---
# # Training and Verification
# Training the model and plotting parity.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)

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
