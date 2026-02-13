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
# id: Lecture07_permutation_importance
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Permutation importance
#
# There are two outstanding questions from the analysis above:
# 1. How to compare variables with different units
# 2. How to evaluate non-linear models
#
# We can address these using other strategies for measuring importance aside from linear coefficients.
# The first method will be called permutation importance.
# We will first train a model, then shuffle (permute) each column one at a time to see how badly the predictions suffer.
# This is basically checking how much the model relies on each feature to make the predictions.
#
# The scheme is illustrated schematically below:
#
# <img src="../lectures/assets/permutation_importance_diagram.jpg" width=600 alt="Visual explanation of the permutation importance algorithm: shuffling a single feature to measure its impact on model error">
#
# We can clean up this result to view it more clearly as a sorted `DataFrame`:
#
# This shows a number of interesting results:
# * Cement is by far the most important despite having an average coefficient
# * Superplasticizer has a small importance despite having the largest coefficient
# * Age is more important than it appeared from the coefficient
#
# The data above might be clearer when visualized as a bar chart:
#
# Remember that higher RMSE indicates a worse result, so `Cement` is the most impactful and `Coarse Aggregate` and `Fine Aggregate` are close to tied for least impactful.
#
# We can add the coefficients to the same chart to compare them head-to-head:
#
# This chart clearly shows that coefficients and permutation importance measure different things.

# %%
def calc_rmse(y, y_pred):
    residuals = y - y_pred
    return np.sqrt(np.mean(residuals**2))

model = linear_model.LinearRegression().fit(x, y)
baseline = calc_rmse(y, model.predict(x))  # first score the baseline model with all columns
print(f'baseline rmse is {baseline}')

permuted = np.zeros_like(x.columns)  # create empty array to store values
for i, col in enumerate(x.columns):
    x_permuted = x.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(x[col])
    permuted[i] = calc_rmse(y, model.predict(x_permuted))  # score on the permuted column
print('rmse on permuted columns is:', permuted)

result = pd.DataFrame({'Feature': x.columns, 'Coefficient': model.coef_,
                       'Permutation Importance': permuted - baseline})
result.sort_values('Permutation Importance')

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend()

axR = ax.twinx()  # set up a second y axis on the same x axis (different scale)
axR.plot(np.arange(model.coef_.shape[0]), model.coef_, label='Coefficient',
         marker='s', linestyle='-', color='tab:orange', zorder=2)
axR.set_ylabel('Coefficient')
axR.set_ylim(-0.25, 0.3)  # make the zero near baseline RMSE
fig
