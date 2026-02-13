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
# id: Lecture09_a_naiive_approach_to_hyperparameter_tuning
# type: Foundational
# parent_lecture: Lecture09
# ---
#
# ## A naiive approach to hyperparameter tuning
#
# Now that we have things well under control, let's just try all the possible values of `n_neighbors`!
#
# We can see from this that there is a non-monotonic relationship with `n_neighbors` and an optimal value lies in the middle of the range.
# So is that all there is to it?

# %%
for k in range(1, 20):
    print(f'k = {k:2d}: ', end='')
    train_and_report_performance(neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance'),
                             xtrain, ytrain, xtest, ytest)
