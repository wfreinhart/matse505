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
# id: Lecture02_correlation
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Correlation
#
# Let's quantify the relationship between several variables using the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
# We can compute this in Python with `scipy.stats.pearsonr`.
# If we look at the docstring for `pearsonr`, we see the following:
#
# This means we should expect two outputs: `r` and `p-value`.
# We can assign these like we do `figure` and `axis` from `plt.subplots`, using a `,` to separate the two variable assignments:
#
# If we don't do this, we'll get a `tuple` of outputs (like when we defined our own function with multiple variables after the `return` statement):
#
# Now let's try computing the Pearson R between all pairs of these variables:

# %%
from scipy import stats

help(stats.pearsonr)

r, p = stats.pearsonr(x, y)
print(r)

out = stats.pearsonr(x, y)
print(out)
print(type(out))

r, p = stats.pearsonr(x, y)
print(r)

r, p = stats.pearsonr(x, z)
print(r)

r, p = stats.pearsonr(y, z)
print(r)
