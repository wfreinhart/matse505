# %% [markdown]
# ---
# id: Viz_Regression
# type: Visualization
# ---
# # Regression Visualization
# Plotting the data and model predictions.

# %%
import matplotlib.pyplot as plt
# Uses global state: model, X, y
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, model.predict(X), color='red', label='Prediction')
plt.legend()
plt.show()
