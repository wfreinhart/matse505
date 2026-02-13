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
# id: Lecture03_investigating_features
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Investigating features
#
# One additional thing we can do is check which features dominate the response by looking at the coefficients, stored in the `coef_` attribute:
#
# Based on this, we see that `Superplasticizer` has the strongest effect per kg, then `Water` and `Cement`.
# `Age` is also near the top of the list, but it's measured in days, not kg, so it's hard to compare it to the others.

# %%
model.coef_

for i, col in enumerate(xtrain.columns):
    print(f'{col:55s} {model.coef_[i]:10.4f}')
