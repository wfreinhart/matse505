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
# id: Lecture03_mse_mae_and_regularization
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## MSE, MAE, and regularization
#
# To get around this problem when evaluating residuals, we discussed several alternate metrics.
# These include the Mean Squared Error (MSE) and Root-MSE (RMSE), which are straightforward to compute:
#
# The RMSE has the same units as $y$, so this means the "typical" residual for the model is about 10 MPa.
#
# > Note that the RMSE formula is actually the same as the standard deviation of the residuals. The difference is just the intent behind the calculation (model error versus variation in a population).
#
# We can compare the RMSE to Mean Absolute Error (MAE):
#
# The MAE applies a power of 1 to the residuals while the RMSE applies a power of 2.
# You can observe that the RMSE leads to a larger value than MAE.
# These normalizations are called $L_1$ and $L_2$ **norms**. Let's make a histogram of the residuals to see what this means visually:
#
# Basically, the L1 residuals are more compact than the L2 residuals (less skewed).
# This means that when fitting the model, the residuals out on the tail will carry much more weight than those close to zero when using L2 regularization.

# %%
mse = np.mean(residual**2)
rmse = np.sqrt(mse)

print('MSE: ', mse)
print('RMSE:', rmse)

print(np.std(residual))

mae = np.mean(np.abs(residual))
print(f'MAE  = {mae}')

res_l1 = np.abs(y - y_model)
res_l2 = (y - y_model)**2

fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.hist(res_l1)
ax.set_xlabel('L1 residual')

ax = axes[1]
_ = ax.hist(res_l2)
ax.set_xlabel('L2 residual')
