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
# id: Lecture02_evaluating_goodness_of_fit
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Evaluating goodness of fit
#
# A common metric for goodness of fit is the $R^2$:
#
# $R^2 = 1 - \frac{SS_\mathrm{residual}}{SS_\mathrm{total}}$
#
# where $SS_\mathrm{residual} = \sum_i (y_i - \hat{y}_i)^2$ and $SS_\mathrm{total} = \sum_i (y_i - \mu)^2$ with $\hat{y}$ being the model prediction and $\mu$ being the data mean.
#
# This is also sometimes called "explained variance" because you can use this alternative formulation:
#
# $R^2 = 1 - \frac{ \mathrm{Var}(y_\mathrm{residual}) } { \mathrm{Var}(y_\mathrm{data}) }$
#
# We can check that we get the same answer as provided by `stats.linregress`:

# %%
residual = z - z_model
Rsquare = 1 - np.var(residual) / np.var(z)
print(Rsquare)

print(result.rvalue**2)
