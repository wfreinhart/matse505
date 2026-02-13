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
# id: Lecture02_histograms
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Histograms
#
# Let's quickly add some additional plot styles to our repertoire.
# **Histograms** can be created using `plt.hist`.
# Remember that histograms visualize distributions of a single variable, so it takes only one variable as input:

# %%
fig, ax = plt.subplots()
ax.hist(x)
ax.set_xlabel(x.name)
ax.set_ylabel('Frequency')
