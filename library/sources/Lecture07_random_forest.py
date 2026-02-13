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
# id: Lecture07_random_forest
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Random Forest
#
# Now we can start with the best go-to model, the Random Forest.
#
# It looks like these have a lot more variance compared to our linear result.
#
# We see that Cement and Age continue to play important roles in this new model, although there are appear to be some stronger influences from the other features compared to the linear model.
# Let's check...

# %%
from sklearn import ensemble

model = ensemble.RandomForestRegressor(random_state=0)
model.fit(xtrain, ytrain)

baseline_rf, permuted_rf = permutation_importance(model, xtest, ytest)
print('rmse on baseline data is:', baseline_rf)
print('rmse on permuted columns is:', permuted_rf)

fig, ax = plt.subplots()
ax.bar(x.columns, permuted_rf, label='Random Forest')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline_rf, 0, len(x.columns)-1, linestyles='dashed', label='Baseline (RF)')
ax.set_ylabel('Model RMSE')
ax.legend()
