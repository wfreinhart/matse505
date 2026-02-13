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
# id: Lecture03_challenges_with_measuring_residuals
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Challenges with measuring residuals
#
# Remember that when we fit a linear regression, the residuals will sum to zero.
# This also means necessarily that the mean will be zero since the mean is just the sum normalized by the count:
#
# We can visualize the distribution of these residuals around zero by creating a histogram:
#
# From this historgram, you can see a generally Normal distribution around the zero mean.
# We need to measure the deviations instead of the central tendency.

# %%
residual = y - y_pred

print('sum: ', np.sum(residual))
print('mean:', np.mean(residual))

fig, ax = plt.subplots()
_ = ax.hist(residual)
ax.set_xlabel('Model residual (eV)')
ax.set_ylabel('Frequency')
ax.set_title('Residuals from linear model of concrete strength')
