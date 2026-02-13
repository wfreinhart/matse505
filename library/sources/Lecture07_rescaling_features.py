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
# id: Lecture07_rescaling_features
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Rescaling features
#
# Let's rescale the features and see how it compares to PCA (or any other dimensionality reduction approach).
# First, the original (unscaled) data:
#
# Here we see that Cement is the dominant feature in the input data.
# Why?
#
# Very simply, that column had the largest absolute variance.
# Thus it was the dominant column in the first component of the PCA.
#
# Now we can move on to the rescaled data for comparison:
#
# Once the columns are normalized, the variance in every column is 1 and the PCA becomes more balanced.
# Finally, we can see what effect this has on the permutation importance:
#
# We can see here that aside from some minor fluctuations (possibly due to uncontrolled random seed in `np.random.permutation`), there is no difference in the permutation feature importance before and after scaling.
# This illustrates an important difference between unsupervised learning and supervised learning!

# %%
from sklearn import decomposition

# do pca
pca = decomposition.PCA()
z = pca.fit_transform(x)

# plot the result
fig, ax = plt.subplots()
_ = ax.scatter(*z[:, :2].T)

# plot the components
fig, ax = plt.subplots()
_ = ax.bar(x.columns, pca.components_[0])
_ = ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)

print( x.std(axis=0) )

from sklearn import preprocessing

# use a scaler to normalize the feature magnitude
x_sc = preprocessing.StandardScaler().fit_transform( x.values )
x_sc = pd.DataFrame( x_sc, columns=x.columns )  # make it back into a DataFrame
pca = decomposition.PCA()
z = pca.fit_transform(x_sc)

# plot the result
fig, ax = plt.subplots()
_ = ax.scatter(*z[:, :2].T)

# plot the components
fig, ax = plt.subplots()
_ = ax.bar(x_sc.columns, pca.components_[0])
_ = ax.set_xticklabels([it.split('(')[0].strip() for it in x_sc.columns], rotation=90)

# compute baseline
model = linear_model.LinearRegression().fit(x_sc, y)
baseline = calc_rmse(y, model.predict(x_sc))  # first score the baseline model with all columns

# compute permutation importance
permuted_sc = np.zeros_like(x_sc.columns)  # create empty array to store values
for i, col in enumerate(x_sc.columns):
    x_permuted = x_sc.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(x_sc[col])
    permuted_sc[i] = calc_rmse(y, model.predict(x_permuted))  # score on the permuted column

# plot result
fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.bar(x_sc.columns, permuted_sc, label='Permuted (S)', edgecolor='tab:orange', facecolor='none')
ax.set_xticklabels([it.split('(')[0].strip() for it in x_sc.columns], rotation=90)
ax.hlines(baseline, 0, len(x_sc.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend()
