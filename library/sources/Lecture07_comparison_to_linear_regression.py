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
# id: Lecture07_comparison_to_linear_regression
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Comparison to Linear Regression
#
# We can compare these results directly to the linear regression result:
#
# Plotting for clarity:
#
# This chart might be easier to read as the change from baseline since the two models have very different baseline RMSE:
#
# From this chart we can see that Cement and Age are very important for both models.
# However, Water and Superplasticizer are also key features for the Random Forest despite not being influential in the linear model.
# Another difference that is clear when plotting this as a percent difference is that Random Forest in general experiences stronger effects from permuting any given column -- even the Aggregates show a notable effect here, while they were completely inconsequential for linear regression.

# %%
model = linear_model.LinearRegression()
model.fit(xtrain, ytrain)
baseline_lin, permuted_lin = permutation_importance(model, xtest, ytest)

fig, ax = plt.subplots()

ax.bar(x.columns, permuted_rf, width=0.5, align='center', label='Random Forest')
ax.hlines(baseline_rf, 0, len(x.columns)-1, linestyles='dashed', color='tab:blue', label='Baseline (RF)')

ax.bar(np.arange(x.columns.shape[0]), permuted_lin, width=0.5, align='edge', label='Linear')
ax.hlines(baseline_lin, 0, len(x.columns)-1, linestyles='dashed', color='tab:orange', label='Baseline (LR)')

ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Model RMSE')
ax.legend()

fig, ax = plt.subplots()

delta_percent_rf = 100 * (permuted_rf - baseline_rf) / baseline_rf
ax.bar(x.columns, delta_percent_rf, width=0.5, align='center', label='Random Forest')

delta_percent_lin = 100 * (permuted_lin - baseline_lin) / baseline_lin
ax.bar(np.arange(x.columns.shape[0]), delta_percent_lin, width=0.5, align='edge', label='Linear')

ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()
