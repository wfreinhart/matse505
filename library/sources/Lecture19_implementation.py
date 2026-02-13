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
# id: Lecture19_implementation
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## Implementation
#
# Let's use [Synthetic Data Vault](https://sdv.dev/) to tackle this problem.
#
# From the [documentation](https://sdv.dev/SDV/user_guides/single_table/gaussian_copula.html):
#
# In mathematical terms, a copula is a distribution over the unit cube $[0,1]^d$ which is constructed from a multivariate normal distribution over $\mathbb{R}^d$ by using the probability integral transform.
# Intuitively, a copula is a mathematical function that allows us to describe the joint distribution of multiple random variables by analyzing the dependencies between their marginal distributions.
#
# <img src="../lectures/assets/lecture19_gaussian_copula.jpg" alt="Illustration of Gaussian Copula for joint distributions" width=600>
#
# Let's try using `GaussianCopula` to generate synthetic values for training:
#
# Now that a bunch of synthetic data were generated, let's use them to train a model:
#
# Here we see two problems:
#
# 1. The performance is poor on the test (synthetic) data
# 2. The performance is poor on the real data

# %%
# !pip install sdv

from sdv.tabular import GaussianCopula

# create a synthetic data generator using GaussianCopula model
generator = GaussianCopula()
generator.fit(missing)

# generate synthetic data with no missing values
synthetic_data = generator.sample(missing.shape[0])
synthetic_data

x = synthetic_data.iloc[:, 1:-1]
y = synthetic_data.iloc[:, -1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=0)

rf = ensemble.RandomForestRegressor(random_state=0).fit(x_train, y_train)
print(f'R2 on train: {rf.score(x_train, y_train):.3f}')
print(f'R2 on test:  {rf.score(x_test, y_test):.3f}')

print(f'R2 on full dataset:  {rf.score(x_full, y_full):.3f}')
