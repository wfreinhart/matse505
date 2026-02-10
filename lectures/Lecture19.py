# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * Imputation
# * Data augmentation
# * Multi-Task Learning

# %% [markdown]
# Let's use this Alloys dataset from before:

# %%
import os

# Set the path to the data file
filename = 'steels.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    data = pd.read_csv(github_url)
data                            # show a view of the data file

# %% [markdown]
# There is an outlier in this dataset that needs to be corrected:

# %%
import numpy as np

bad_idx = np.argmax( data.loc[:, ' Tensile Strength (MPa)'] )
data.loc[bad_idx, ' Tensile Strength (MPa)'] /= 10.0  # missed a decimal point

# %% [markdown]
# # Imputation
#
# Data imputation is the process of filling in missing values in a dataset. This is a common problem in data analysis, as many real-world datasets have missing values.
# Fortunately, there are several readily available methods available for imputing missing data.
#
# <img src="../lectures/assets/lecture19_imputation.jpg" alt="Illustration of data imputation techniques" width=600>

# %% [markdown]
# ## The limiting case of supervised learning
#
# In the limit of missing entries from only one column, the imputation problem becomes supervised learning.

# %%
from sklearn import model_selection

x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

print('feature names:\n', x.columns)
print()
print('label name:\n', y.name)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

print('dataset sizes:')
print(x_train.shape, x_test.shape)

# %% [markdown]
# This is the same as having 229 missing entries from the `Reduction in Area (%)` column.

# %%
from sklearn import ensemble

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

# %% [markdown]
# Keep this performance in mind as we move ahead to talk about multi-column imputation.

# %% [markdown]
# ## Missing at random from multiple columns
#
# Let's set up a `DataFrame` with more missing values.

# %%
import numpy as np

missing = data.copy(deep=True)

n_missing = 100
rng = np.random.default_rng(0)

miss_idx = []
for i, col in enumerate([' Si', ' Ni', ' Al', ' Temperature (°C)', ' Elongation (%)']):
    rand_idx = rng.choice(np.arange(missing.shape[0]), n_missing, replace=False)
    missing.loc[rand_idx, col] = np.nan
    miss_idx.append(rand_idx)

missing

# %% [markdown]
# Just to prove to you that we have dropped a bunch of columns:

# %%
print( missing.dropna().shape )

# %% [markdown]
# Here is the problem: if we want to use all the columns for training a model, we now lost half our training data despite missing less than 3% of the values (because if one column is missing the entire row is invalidated).
#
# Also note that we are not missing `5 x 100 = 500` rows, instead only 398.
# This shows that a bunch of columns have multiple missing entries.

# %%
x = missing.dropna().iloc[:, 1:-1]
y = missing.dropna().iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

# %% [markdown]
# The performance is degraded compared to the larger dataset without missing values.
# We can also check its performance on the held-out data from our artifical introduction of missing data:

# %%
full_idx = np.sort(np.hstack(miss_idx))
x_full = data.iloc[full_idx, 1:-1]
y_full = data.iloc[full_idx, -1]

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# Here we can see the performance is even lower than implied by the test set from our reduced `missing` DataFrame.

# %% [markdown]
# ## `pandas` imputation schemes
#
# OK, so missing data hurts the model performance.
# What can we do about it?
#
# One simple method for data imputation is mean imputation.
# This involves filling in missing values with the mean of the non-missing values in that column.
# The basic logic behind mean imputation is that the mean value is often a reasonable estimate of the missing value, especially when the missing data is randomly distributed and not related to other variables.
#
# We can use the `fillna` method with the `mean` argument:
#

# %%
imputed = missing.copy(deep=True)

imputed[' Temperature (°C)'].fillna(imputed[' Temperature (°C)'].mean(), inplace=True)

print(imputed.dropna().shape)

# %% [markdown]
# Here the number of rows is increased from 517 to 573 by filling in missing `Temperature (°C)` values.

# %%
x = imputed.dropna().iloc[:, 1:-1]
y = imputed.dropna().iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# Interestingly, despite a seemingly reduced performance on the test set, the performance is improved on the full (held-out) dataset when training the model on the mean-imputed temperature values!
# This shows how imputation can help learn from all the other available data even if the value in the imputed column is not perfect.

# %% [markdown]
# Another common method for data imputation is forward or backward filling.
# This involves filling in missing values with the value that appears in the previous or next row, respectively.
# The basic logic behind forward fill imputation is that the missing values are sometimes similar to the previous observed values (i.e., there is an ordering to the dataset).
#
# We can use the `fillna` method with the method argument set to either `ffill` (for forward filling) or `bfill` (for backward filling). For example:

# %%
imputed = missing.copy(deep=True)

imputed[' Temperature (°C)'].fillna(method='ffill', inplace=True)

print(imputed.dropna().shape)

# %%
x = imputed.dropna().iloc[:, 1:-1]
y = imputed.dropna().iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# Here we see a similar result compared to the mean imputation above.
# It is likely that there is some kind of ordering to the temperatures.

# %% [markdown]
# ## `sklearn` imputers
#
# The `scikit-learn` module has a submodule called `sklearn.impute` that implements the following:
# * `SimpleImputer`
# * `IterativeImputer`
# * `KNNImputer`
#
# `SimpleImputer` is similar to the schemes we just went over in `pandas`, so we'll skip it.
# The `IterativeImputer` is also "experimental" for now and in heavy development, so we'll skip that one as well.

# %% [markdown]
# From the [documentation](https://scikit-learn.org/stable/modules/impute.html):
#
#
# The `KNNImputer` class provides imputation for filling in missing values using the k-Nearest Neighbors approach.
# By default, a Euclidean distance metric that supports missing values, `nan_euclidean_distances`, is used to find the nearest neighbors.
# Each missing feature is imputed using values from `n_neighbors` nearest neighbors that have a value for the feature.
# The feature of the neighbors are averaged uniformly or weighted by distance to each neighbor.
#
# If a sample has more than one feature missing, then the neighbors for that sample can be different depending on the particular feature being imputed.
# When the number of available neighbors is less than `n_neighbors` and there are no defined distances to the training set, the training set average for that feature is used during imputation.
# If there is at least one neighbor with a defined distance, the weighted or unweighted average of the remaining neighbors will be used during imputation.
# If a feature is always missing in training, it is removed during transform.

# %%
from sklearn import impute

imputer = impute.KNNImputer(n_neighbors=5)
result = imputer.fit_transform(missing.iloc[:, 1:])

x = result[:, :-1]
y = result[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# The imputer returns an array with the missing values imputed.
# Note that we did not include the first column,
# We can use it to train the model as above:

# %%
x = result[:, :-1]
y = result[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# Here you can see that the model performance is substantially higher on the full dataset when the values have been imputed.
# > The warning about feature names is because we trained this Random Forest on an `ndarray` (`result`) but we are testing it on a `DataFrame` (`x_full`)

# %% [markdown]
# ## MissForest
#
# MissForest is an imputation algorithm for handling missing data that is based on the random forest algorithm. The algorithm works as follows:
#
# 1. Identify the features that contain missing values.
# 2. Split the dataset into two parts: one part that contains the missing values and one part that contains the observed values.
# 3. For each feature that contains missing values, build a random forest regression model using the observed values as the training data and the feature with missing values as the target variable.
# 4. Predict the missing values for each feature using the corresponding random forest regression model.
# 5. Repeat steps 3-4 until the missing values converge.
#
# During each iteration, the random forest model is trained using the observed values for all features except the feature with missing values. The missing values for the feature are then predicted using the trained model. This process is repeated for all features with missing values until the missing values converge, which means that the imputed values no longer change significantly.
#
#
#
# MissForest uses the random forest algorithm because it can handle a mixture of categorical and numerical data, and can handle non-linear relationships between features. The algorithm is also able to capture feature interactions, which is important when imputing missing values.

# %% [markdown]
# We will have to start by installing the package:

# %%
# !pip install MissForest

# %% [markdown]
# Once installed, using it is straightforward:

# %%
from missforest.miss_forest import MissForest

mf = MissForest()
imputed = mf.fit_transform(missing)

imputed

# %% [markdown]
# First, let's compare the results from `MissForest` with the real thing.
# > Note that we would never have this opportunity in real life when we have actual missing data. This academic example is only to evaluate the quality of the imputation.

# %%
from matplotlib import pyplot as plt

missing_cols = [' Si', ' Ni', ' Al', ' Temperature (°C)', ' Elongation (%)']

fig, ax = plt.subplots()
for col in missing_cols:
    scale = 1 / data[col].max()  # scale all the data to [0, 1]
    ax.plot(imputed[col] * scale, data[col] * scale, '.', label=col)
ax.set_xlabel('Imputed')
ax.set_ylabel('Observation')
ax.legend()

# %% [markdown]
# Unfortunately it seems the `MissForest` imputer is not doing a great job at guessing the missing values.
#
# Let's see how our regression model performs with this imputation.

# %%
x = imputed.iloc[:, 1:-1]
y = imputed.iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# Despite the imperfect imputation above, this is the best performance we found so far.
# Why are we able to get good performance when using values that don't match the real observations?
# Basically it's because only 1-2 of the values in each row are wrong, while all the rest are real data!
# The slight hit we take to reliability of those 1-2 features is worth it to retain all the other columns and double the volume of training data.

# %% [markdown]
# # Synthetic data generation

# %% [markdown]
# ## Concept
#
# We can take a related approach using synthetic data.
# This is an extension of imputation to *the entire row*.

# %% [markdown]
# ## Implementation
#
# Let's use [Synthetic Data Vault](https://sdv.dev/) to tackle this problem.

# %%
# !pip install sdv

# %% [markdown]
# From the [documentation](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html):
#
# In mathematical terms, a copula is a distribution over the unit cube $[0,1]^d$ which is constructed from a multivariate normal distribution over $\mathbb{R}^d$ by using the probability integral transform.
# Intuitively, a copula is a mathematical function that allows us to describe the joint distribution of multiple random variables by analyzing the dependencies between their marginal distributions.
#
# <img src="../lectures/assets/lecture19_gaussian_copula.jpg" alt="Illustration of Gaussian Copula for joint distributions" width=600>
#
# Let's try using `GaussianCopula` to generate synthetic values for training:

# %%
from sdv.tabular import GaussianCopula

# create a synthetic data generator using GaussianCopula model
generator = GaussianCopula()
generator.fit(missing)

# generate synthetic data with no missing values
synthetic_data = generator.sample(missing.shape[0])
synthetic_data

# %% [markdown]
# Now that a bunch of synthetic data were generated, let's use them to train a model:

# %%
x = synthetic_data.iloc[:, 1:-1]
y = synthetic_data.iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# Here we see two problems:
#
# 1. The performance is poor on the test (synthetic) data
# 2. The performance is poor on the real data

# %% [markdown]
# ## Augmentation
#
# Above, we have taken the most extreme possible stance here by fitting the synthetic data model to the `NaN`-containing `missing` DataFrame and then training on ONLY the synthetic data generated by the copula model.
# In reality, it makes more sense to train on an *augmented* dataset that contains both real and synthetic observations, like so:

# %%
# concatenate the real data (without missing values) and the synthetic data
augmented_data = pd.concat([missing.dropna(), synthetic_data])

x = augmented_data.iloc[:, 1:-1]
y = augmented_data.iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')

# %% [markdown]
# This is the best performance on the full dataset of any imputation method we have seen so far!
# It is somewhat surprising because we didn't actually do imputation in the end, but instead we generated new synthetic data that follows a similar joint probability distribution.

# %% [markdown]
# # Multi-Task Learning

# %% [markdown]
# ## Concepts
#
# <img src="../lectures/assets/lecture19_mtl_architecture.jpg" alt="Multi-task learning architecture diagram" width=600>

# %% [markdown]
# <img src="../lectures/assets/lecture19_mtl_types.jpg" alt="Diagram showing different types of multi-task learning (MISO, SIMO)" width=600>
#
# Multi Input Single Output (MISO) is for something like computer vision, where there could be multiple views of the same object with one common feature.
#
# Single Input Multi Output (SIMO) is for something like material characterization, where a single object (a material composition) produces multiple signals (measured properties).

# %% [markdown]
# ## Practical example
#
# Let's randomly remove labels from some of the columns that correspond to mechanical properties:

# %%
import numpy as np

missing = data.copy(deep=True)

n_missing = 515
rng = np.random.default_rng(0)

miss_idx = []
for i, col in enumerate([' 0.2% Proof Stress (MPa)', ' Tensile Strength (MPa)',
                         ' Elongation (%)', ' Reduction in Area (%)']):
    rand_idx = rng.choice(np.arange(missing.shape[0]), n_missing, replace=False)
    missing.loc[rand_idx, col] = np.nan
    miss_idx.append(rand_idx)

missing

# %% [markdown]
# How many data are left if we remove rows without any target labels?

# %%
missing.iloc[:, -4:].dropna(how='all').shape

# %% [markdown]
# Now let's check how much data is left in each combination of properties:

# %%
import itertools

for i in range(1, 5):
    for cols in itertools.combinations(range(4), i):
        sub_df = missing.iloc[:, -4:].iloc[:, list(cols)].dropna()
        print(f'{str(cols):12s} -> {sub_df.shape[0]} entries')

# %% [markdown]
# You see here that with 400 entries from each property, we have only 45 entries with all 4 properties available.
# This will make it difficult to leverage all the data to train an accurate model.

# %% [markdown]
# ## Regression in `scikit-learn`
#
# Let's evaluate the performance of 4 single-task models compared to a model that simultaneously predicts all 4 labels using `sklearn.neural_network`

# %% [markdown]
# ### 1 model, 1 target
#
# We'll start by training 4 individual models, one for each target label:

# %%
from sklearn import neural_network
from sklearn import preprocessing, model_selection

x_full = data.iloc[:, 1:-4].values.astype(float)
y_full = data.iloc[:, -4:].values.astype(float)

for i in range(4):
    print(f'Task {i}')
    print('------------------')

    # set up the feature and labels
    x_arr = missing.iloc[:, 1:-4].values.astype(float)
    y_arr = missing.iloc[:, i-4].values.astype(float).reshape(-1, 1)

    # filter out missing labels
    not_nan = np.isfinite(y_arr).flatten()

    # create scalers for both input and output features
    x_scaler = preprocessing.StandardScaler()
    x_sc = x_scaler.fit_transform(x_arr[not_nan])

    y_scaler = preprocessing.StandardScaler()
    y_sc = y_scaler.fit_transform(y_arr[not_nan]).flatten()

    # train / test split
    idx_train, idx_test = model_selection.train_test_split(np.arange(x_sc.shape[0]), random_state=0)

    # train a model
    model_skl = neural_network.MLPRegressor(hidden_layer_sizes=(128, 64, ),
                                            max_iter=1000, random_state=0)
    model_skl.fit(x_sc[idx_train], y_sc[idx_train])

    # calculate error on these train data
    residuals = model_skl.predict(x_sc[idx_train]) - y_sc[idx_train]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Train RMSE {i}: {rmse:.3f}')

    # calculate error on these test data
    residuals = model_skl.predict(x_sc[idx_test]) - y_sc[idx_test]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Test  RMSE {i}: {rmse:.3f}')

    # calculate error on the full dataset
    xf_sc = x_scaler.transform(x_full)
    yf_sc = y_scaler.transform(y_full[:, i-4].reshape(-1, 1))

    y_hat = model_skl.predict(xf_sc)
    residuals = y_hat - yf_sc
    print(f'Full  RMSE {i}: {np.sqrt(np.mean(residuals**2)):.3f}')
    print()

# %% [markdown]
# You see here that the model is highly overfitted to the training set in each task.

# %% [markdown]
# ### 1 model, 4 targets
#
# What if we train a single model to predict all 4 targets at once?

# %%
from sklearn import neural_network
from sklearn import preprocessing

x_arr = missing.iloc[:, 1:-4].values.astype(float)
y_arr = missing.iloc[:, -4:].values.astype(float)

not_nan = np.isfinite(y_arr.sum(axis=1)).flatten()

x_scaler = preprocessing.StandardScaler()
x_sc = x_scaler.fit_transform(x_arr[not_nan])

y_scaler = preprocessing.StandardScaler()
y_sc = y_scaler.fit_transform(y_arr[not_nan])

idx_train, idx_test = model_selection.train_test_split(np.arange(x_sc.shape[0]), random_state=0)

print(f'{len(idx_train)} train data, {len(idx_test)} test data')

model_skl = neural_network.MLPRegressor(hidden_layer_sizes=(128, 64, ), max_iter=2000, random_state=0)
_ = model_skl.fit(x_sc[idx_train], y_sc[idx_train])

# %% [markdown]
# This is a tiny dataset!
# We can't expect good performance.
# Nevertheless, let's evaluate it:

# %%
for i in range(4):
    print(f'Task {i}')
    print('------------------')

    # calculate error on these train data
    residuals = model_skl.predict(x_sc[idx_train])[:, i] - y_sc[idx_train, i]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Train RMSE {i}: {rmse:.3f}')

    # calculate error on these test data
    residuals = model_skl.predict(x_sc[idx_test])[:, i] - y_sc[idx_test, i]
    rmse = np.sqrt(np.mean(residuals**2))

    print(f'Test  RMSE {i}: {rmse:.3f}')

    # calculate error on the full dataset
    xf_sc = x_scaler.transform(x_full)
    yf_sc = y_scaler.transform(y_full)[:, i-4].reshape(-1, 1)

    y_hat = model_skl.predict(xf_sc)
    residuals = y_hat[:, i] - yf_sc
    print(f'Full  RMSE {i}: {np.sqrt(np.mean(residuals**2)):.3f}')
    print()

# %% [markdown]
# We can visualize this using a parity plot:

# %%
from matplotlib import pyplot as plt

# calculate error on the full dataset
xf_sc = x_scaler.transform(x_full)
yf_sc = y_scaler.transform(y_full)[:, -4:]

y_hat = model_skl.predict(xf_sc)

# make parity plot
fig, ax = plt.subplots()
ax.plot([-3, 3], [-3, 3], 'k--')
for i in range(y_hat.shape[1]):
    ax.scatter(y_hat[:, i], yf_sc[:, i], label=f'{missing.columns[i-4]}')

ax.set_xlabel('Predicted')
ax.set_ylabel('Observed')
ax.legend()

# %% [markdown]
# You can see here that the result is quite bad.
# It is interesting to note that the performance on the "full" dataset (before making missing values) is much worse than the "test" dataset.
# This reveals that the generalization error is quite high and the model is highly overfitted (as expected from a NN with so few training data).

# %% [markdown]
# ## Multi-task regression in PyTorch
#
# Now we can implement a Single Input Multi Output (SIMO) scheme in PyTorch.
#
# As always, we start with PyTorch Lightning:

# %%
try:
    import pytorch_lightning as pl
except:
    # !pip install pytorch_lightning
    import pytorch_lightning as pl

# %% [markdown]
# We set up the datasets like so:

# %%
import torch
from torch.utils.data import DataLoader

idx_train, idx_test = model_selection.train_test_split(np.arange(missing.shape[0]), random_state=0)

x_arr = missing.iloc[:, 1:-4].values.astype(float)  # 15 features (skip categorical codes)
y_arr = missing.iloc[:, -4:].values.astype(float)    # 4 labels (the mechanical properties)

x_scaler = preprocessing.StandardScaler().fit(x_arr[idx_train])
y_scaler = preprocessing.StandardScaler().fit(y_arr[idx_train])

x = torch.tensor(x_scaler.transform(x_arr)).float()
y = torch.tensor(y_scaler.transform(y_arr)).float()

ds_train = [(x[i], y[i]) for i in idx_train]
ds_test = [(x[i], y[i]) for i in idx_test]

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

# %% [markdown]
# ### Custom loss function
#
# We need a custom loss function to handle missing (`NaN`) values:

# %%
from torch import nn
import torch.nn.functional as F

class MultiTaskMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Mask out the NaN values in the targets
        mask = ~torch.isnan(target)

        # Compute the mean squared error, ignoring NaN values
        mse = F.mse_loss(input[mask], target[mask], reduction='mean')

        return mse


# %% [markdown]
# ### Model architecture
#
# The model will have some extra tricks compared to our usual MLP:

# %%
import pytorch_lightning as pl

class MLPRegressor(nn.Module):
    def __init__(self, hidden_size=(100, )):
        super(MLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(nn.LeakyReLU())

        self.fc_layers = nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class MultiTaskMLP(pl.LightningModule):
    def __init__(self, hidden_size, n_tasks):
        super(MultiTaskMLP, self).__init__()

        self.n_tasks = n_tasks

        # make a shared backbone for all tasks
        self.backbone = MLPRegressor(hidden_size)

        # now make a head for each task
        self.heads = [nn.LazyLinear(1) for _ in range(n_tasks)]

        # self.criterion = nn.MSELoss(reduction='none')  # handle nan specially
        self.criterion = MultiTaskMSELoss()

    def forward(self, x):
        x = self.backbone(x)
        out = [h(x) for h in self.heads]  # task-specific layers
        return out

    def training_step(self, batch, batch_idx):

        x, y = batch
        out = self(x)

        # compute loss for each task separately
        loss = 0
        for i in range(self.n_tasks):
            loss += self.criterion(out[i], y[:, i].unsqueeze(1))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        with torch.no_grad():
            out = self(x)

            loss = 0
            for i in range(self.n_tasks):
                task_loss = self.criterion(out[i], y[:, i].unsqueeze(1))
                loss += task_loss
                self.log(f"task_{i}_loss", task_loss)

        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


# %% [markdown]
# ### Implementation and evaluation

# %%
from pytorch_lightning.loggers import CSVLogger

logger = CSVLogger('logs', name='mtl-mlp')

torch.manual_seed(0)  # control random effects

model_ft = MultiTaskMLP([128, 64, ], 4)

# Initialize the weights before sending to pl to count trainable weights
model_ft(dl_train.dataset[0][0].unsqueeze(0))

# Use pl to train
trainer = pl.Trainer(max_epochs=40, logger=logger, log_every_n_steps=1)
trainer.fit(model=model_ft, train_dataloaders=dl_train, val_dataloaders=dl_test)

# %% [markdown]
# Let's plot the performance:

# %%
from matplotlib import pyplot as plt

log_path = 'logs/mtl-mlp/version_0/metrics.csv'
if os.path.exists(log_path):
    metrics = pd.read_csv(log_path)
else:
    colab_path = '/content/logs/mtl-mlp/version_0/metrics.csv'
    if os.path.exists(colab_path):
        metrics = pd.read_csv(colab_path)
    else:
        print(f"Warning: Log file not found at {log_path} or {colab_path}")
        metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'validation_loss'])

fig, ax = plt.subplots()
ax.plot( metrics['epoch'] * len(dl_train) + metrics['step'], np.sqrt(metrics['train_loss']), '.', label='Train')
ax.plot( metrics['epoch'] * len(dl_train) + metrics['step'], np.sqrt(metrics['validation_loss']), '.', label='Test')
ax.set_yscale('log')
ax.set_xlabel('Train Step')
ax.set_ylabel('RMSE')
ax.legend()

# %% [markdown]
# And if we evaluate the performance on each task:

# %%
for name, dl in [('Train', dl_train), ('Test', dl_test)]:
    print(name)
    print('-------------')
    losses = [0] * model_ft.n_tasks

    for batch in dl:
        x, y = batch
        with torch.no_grad():
            out = model_ft(x)

            for i in range(model_ft.n_tasks):
                this_loss = model_ft.criterion(out[i], y[:, i].unsqueeze(1)).item()
                losses[i] += this_loss

    for i in range(model_ft.n_tasks):
        print(f'Task {i}: {np.sqrt(losses[i]):.3f}')
    # print(f'> Total: {np.sqrt(np.sum(losses)):.3f}')
    print()

# %% [markdown]
# So far this looks worse than our train performance but similar to our test performance with the single-task models from `scikit-learn`.
#
# What about on the full dataset?

# %%
x_full = data.iloc[:, 1:-4].values.astype(float)
y_full = data.iloc[:, -4:].values.astype(float)

xf_sc = torch.tensor(x_scaler.transform(x_full)).float()
yf_sc = torch.tensor(y_scaler.transform(y_full)).float()

with torch.no_grad():
    y_hat = model_ft(xf_sc)

print('Full')
print('-------------')

for i in range(y.shape[1]):
    residuals = y_hat[i] - yf_sc[:, i].unsqueeze(1)
    print(f'Task {i}: {np.sqrt(np.mean(residuals.detach().numpy()**2)):.3f}')

# %% [markdown]
# If you recall, our performance on the Full dataset with our 4 individual models was:
#
# ```
# Task 0: 1.398
# Task 1: 1.405
# Task 2: 1.356
# Task 3: 1.412
# ```
#
# and our performance with the full 4-output model was:
#
# ```
# Task 0: 1.389
# Task 1: 1.362
# Task 2: 1.622
# Task 3: 1.492
# ```
#
# These terrible scores were clearly the result of overfitting a large model (10k parameters) to small datasets.
# With the Multi-Task Learning scheme, we get several advantages:
#
# 1. **More rows with $(x, y)$ pairs.** Because there are a small number of rows missing all 4 labels compared to the number missing 0-3 labels, more $x$ samples are included in training. Each of these helps train the backbone to learn something useful.
# 2. **More $(x, y)$ pairs per row.** Because there are 4 targets being predicted from one input $x$, the total number of target labels is much higher.
# 3. **Better generalization.** Because there is a common backbone for 4 different tasks, the model is explicitly forced to learn a representation that is most general. This predictably helps the model avoid overfitting.

# %% [markdown]
# We can visualize this improved performance with a parity plot again:

# %%
for batch in dl_test:
    x, y = batch
    with torch.no_grad():
        out = model_ft(x)

fig, ax = plt.subplots()
ax.plot([-3, 3], [-3, 3], 'k--')
for i in range(model_ft.n_tasks):
    ax.scatter(out[i].detach().numpy(), y[:, i].detach().numpy(), label=f'{missing.columns[i-4]}')

ax.set_xlabel('Predicted')
ax.set_ylabel('Observed')
ax.legend()

# %% [markdown]
# Much better! Recall that these are the scaled outputs.
# We can unscale the outputs with the `inverse_transform` method of the scaler:

# %%
y_orig = y_scaler.inverse_transform( y.detach().numpy() )
y_hat = y_scaler.inverse_transform( np.hstack([it.detach().numpy() for it in out]) )

fig, ax = plt.subplots()
ax.plot([0, 800], [0, 800], 'k--')
for i in range(model_ft.n_tasks):
    ax.scatter(y_hat[:, i], y_orig[:, i], label=f'{missing.columns[i-4]}')

ax.set_xlabel('Predicted')
ax.set_ylabel('Observed')
ax.legend()

# %% [markdown]
# Note that if we were to regress on these raw labels we would heavily bias the MTL model to `Tensile Strength` since it is measured in 100's compared to `Elongation` which is measured in 10's.

# %%
