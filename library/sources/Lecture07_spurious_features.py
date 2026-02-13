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
# id: Lecture07_spurious_features
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Spurious features
#
# Let's add a fake feature to the data and compare results between PCA and feature importance:
#
# Just to make this totally unambiguous, let's take a look at the resulting PCA decomposition:
#
# As you can see the Random feature totally dominates the space due to its large magnitude.
# Now let's pass this augmented data through permutation importance:
#
# You'll see here that the Random feature is assigned practically no importance in terms of changing the model score.

# %%
x_aug = x.copy()
x_aug['Random'] = np.random.rand(x_aug.shape[0]) * 1e6
x_aug

# do pca
pca = decomposition.PCA()
z = pca.fit_transform(x_aug)

# plot the result
fig, ax = plt.subplots()
_ = ax.scatter(*z[:, :2].T)

# plot the components
fig, ax = plt.subplots()
_ = ax.bar(x_aug.columns, pca.components_[0])
_ = ax.set_xticklabels([it.split('(')[0].strip() for it in x_aug.columns], rotation=90)

# compute baseline
model = linear_model.LinearRegression().fit(x_aug, y)
baseline = calc_rmse(y, model.predict(x_aug))  # first score the baseline model with all columns

# compute permutation importance
permuted_aug = np.zeros_like(x_aug.columns)  # create empty array to store values
for i, col in enumerate(x_aug.columns):
    x_permuted = x_aug.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(x_aug[col])
    permuted_aug[i] = calc_rmse(y, model.predict(x_permuted))  # score on the permuted column

# plot result
fig, ax = plt.subplots()
ax.bar(x_aug.columns, permuted_aug, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x_aug.columns], rotation=90)
ax.hlines(baseline, 0, len(x_aug.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend()
