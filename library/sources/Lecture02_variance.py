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
# id: Lecture02_variance
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Variance
#
# Exactly the same except with either `np.var` (variance) or `np.std` (standard deviation):
#
# We can also use the `axis` keyword as always:

# %%
print(np.std(x))
print(np.var(x))
print(np.sqrt(np.var(x)))  # demonstrate the definition of std dev

print(data.std(axis=0))
