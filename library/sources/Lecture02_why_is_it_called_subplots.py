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
# id: Lecture02_why_is_it_called_subplots
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Why is it called subplots?
#
# `subplots` is meant for making multiple plots in a single `figure`.
# We specify `nrows` and `ncols`:
#
# For now we'll just use it with the default `nrows=1` and `ncols=1` to make simple charts.

# %%
fig, ax = plt.subplots(nrows=2, ncols=3)

# plt.subplots?

fig, ax = plt.subplots()
