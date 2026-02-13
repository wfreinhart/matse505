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
# id: Lecture07_drop_column_importance
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Drop-column importance
#
# Permutation feature importance tells us something very specific: how much  **this trained model** depends on the particular feature we are permuting.
# It does not tell us how necessary the given column is for predicting the outcome.
#
# To be more holistic we can actually train a series of models while leaving out each column and see how well the model can do.
# This called drop-column importance because we are dropping each column in the training.
#
# The results here appear much closer than in the permutation importance.
# Let's reuse our plotting code from above:
#
# We can show both Permutation and Drop-Column importance on the same chart to really get a sense for it:
#
# Here we see that not only are the quantitative results different between the two, but the general trends are even different.
# For instance, the model does not suffer nearly as much as we thought when it loses access to Cement *and can retrain without it*.
# This is probably because the same information is available from the rest of the columns (i.e., the total density is similar for all instances).
# Instead, Age of the same becomes the most influential variable because it cannot be inferred from the other data.

# %%
import copy

x = np.random.rand(5)
print(x)
y = copy.deepcopy(x)
y += 1
print(x)

model.fit(x, y)
baseline = calc_rmse(y, model.predict(x))  # first score the baseline model with all columns
print(f'baseline rmse is {baseline}')

dropped = np.zeros_like(x.columns)  # create empty array to store values
for i, col in enumerate(x.columns):
    x_dropped = x.copy().drop(columns=col)  # remember to copy!
    model.fit(x_dropped, y)
    dropped[i] = calc_rmse(y, model.predict(x_dropped))  # score with the dropped column
print('rmse on dropped columns is:', dropped)

fig, ax = plt.subplots()
ax.bar(x.columns, dropped, label='Dropped')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend(loc='lower center')

fig, ax = plt.subplots()
ax.bar(x.columns, permuted, width=0.5, align='edge', label='Permuted')
ax.bar(x.columns, dropped, width=0.5, align='center', label='Dropped')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend(loc='lower center')
