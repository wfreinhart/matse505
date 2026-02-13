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
# id: Lecture02_a_shortcut
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## A shortcut
#
# We can start by defining some data of interest.
# I highly recommend a more declarative style of programming with variables defined whenever something will be reused.
# This makes it easier to read and debug.
# You can always optimize for speed later (we'll discuss...).
#
# Let's take a second to investigate these new variables:
#
# A `Series` is kind of like a `DataFrame` but only for one column.
# It retains some useful features:

# %%
x = data.loc[:, 'Atomic Mass']             # one option for indexing
y = data['Vacancy Formation Energy (eV)']  # another option for indexing
z = data.loc[:, 'Bulk Static Energy (eV)'] # I prefer this one -- more explicit

print(type(x))

print(x.name)    # an attribute
print(x.shape)   # another attribute
print(x.mean())  # a method (albeit with a static value)
