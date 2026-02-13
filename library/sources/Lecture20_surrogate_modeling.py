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
# id: Lecture20_surrogate_modeling
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Surrogate modeling
#
# Let's generate a surrogate model for this simplified system:

# %%
# split data
idx_train, idx_test = model_selection.train_test_split(np.arange(x.shape[0]))

# fit model
model = ensemble.RandomForestRegressor(random_state=0)
_ = model.fit(x[idx_train], y[idx_train].flatten())

# report performance
print(f'Train {model.score(x[idx_train], y[idx_train]):.3f}')
print(f'Test  {model.score(x[idx_test],  y[idx_test]):.3f}')
