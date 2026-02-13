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
# id: Lecture02_statistical_moments
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Statistical moments
#
# We might be interested in some higher moments of data.
# Here are the skew and kurtosis for our `Series`:
#
# These can be computed manually using the definition for [standardized moments](https://en.wikipedia.org/wiki/Standardized_moment):
#
# $\bar{\mu}_3 = E[(X-\mu)^3] / \sigma^3$
#
# $\bar{\mu}_4 = E[(X-\mu)^4] / \sigma^4$
#
# where $\mu$ is the mean and $\sigma$ is the standard deviation.
#
# You will notice that the skew is pretty close, but the kurtosis is way off (even a different sign).
# We can look at the documentation for `kurt` to find out why:
#
# The Fisher's definition of kurtosis uses a normal distribution as the baseline, so we need to subtract 3:

# %%
print(x.skew())
print(x.kurt())

mu = np.mean(x)
sigma = np.std(x)
skew = np.mean((x-mu)**3) / sigma**3
print(skew)

kurt = np.mean((x-mu)**4) / sigma**4
print(kurt)

help(x.kurt)

kurt_fisher = np.mean((x-mu)**4) / sigma**4 - 3
print(kurt_fisher)
