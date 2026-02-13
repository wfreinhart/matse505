# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# ---
# id: Lecture03_visualizing_model_performance
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Visualizing model performance
#
# How should we visualize the result?
# $x$ is no longer a 1D vector, but a 2D array with many observations of multiple values.
# We can make charts with up to 4 variables using 3D scatter charts plus a color axis, but here we have 8 features!
#
# The standard way is to plot the Prediction against the Observation, called a **parity plot**.
# This shows you whether there is a systematic bias in your model, and the chart looks the same no matter what sort of model you're using.
#
# We can see a slight bias to over-estimate at the bottom and over-estimate at the top. What might be going on here? One thing we could do is make a scatter plot of the results and color the points according to one of the variables. Then we will know if there is a correlation between the variable value and the residuals. Let's try it with `Age`:
#
# Here we see that indeed several of the points furthest from the Reference line are colored yellow, indicating an Age of 365 days. This shows that the model over-estimates the strength of the concrete when aged for much longer than average. We can see the same effect using a `scatter_3d` chart with `plotly`:
#
# We can take another view of the data by plotting the residual against the Age:
#
# If we look back at the `Concrete compressive strength` vs `Age` chart, we would see a highly nonlinear relationship between the two. This is backed up by the nonlinear shape of the residuals shown above. We will resolve this problem using more sophisticated methods very shortly.

# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))

# plot the results
ax.plot(y, y_pred, '.', label='Data')

# create reference line along y = x to show the desired behavior
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_title('Multiple linear regression model of concrete strength')
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.legend()

fig, ax = plt.subplots(figsize=(5, 5))

# plot the results
im = ax.scatter(y, y_pred, s=16, c=data['Age (day)'], label='Data')
cb = plt.colorbar(im, ax=ax)
cb.set_label('Age (day)')

# create reference line along y = x to show the desired behavior
min_max = np.array([y.min(), y.max()])
ax.plot(min_max, min_max, 'k--', label='Reference')
ax.set_aspect('equal')  # very helpful to show y = x relationship

# add labels and legend
ax.set_title('Linear regression of concrete strength')
ax.set_xlabel('Observation')
ax.set_ylabel('Prediction')
ax.legend()

from plotly import express as px

px.scatter_3d(x=y, y=y_pred, z=data['Age (day)'], color=data['Age (day)'], width=800, height=600)

age = data['Age (day)']
fig, ax = plt.subplots()
ax.plot(age, residuals, '.')
ax.plot([age.min(), age.max()], [0, 0], 'k--')
ax.set_xlabel('Observation')
ax.set_ylabel('Residual')
