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
# id: Lecture02_indexing_practice
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Indexing practice
#
# As we said earlier, in addition to accessing single values, `loc` can access ranges of values.
# This is done with `:` as we saw with `lists`.
# When we use `:`, we get the elements starting with the number to the left and ending with one less than the number on the right.
# So `0:3` accesses elements at indices `[0, 1, 2]`.
# Let's try it out:
#
# We could get all the elements in either a row or column by not specifying any numbers to the side of the `:`:
#
# We can also use `:` with the index and column names, which is more intuitive -- in this case pandas actually includes the ending entry too:
#
# **IMPORTANT:**
# Note that in this case pandas actually returns a `DataFrame` object which is a subset of the larger `ele_data` `DataFrame`.
#
# Pandas formats this nicely if we drop the `print` statement:

# %%
print(ele_data.iloc[0, 0:3])

print(ele_data.iloc[0, :])  # access a whole column by specifying a row only
print('')  # give a line break between print statements
print(ele_data.iloc[:, 0])  # access a whole row by specifying a column only

print(ele_data.loc['Al':'Cu', 'Atomic Number':'STP Phase'])

sub_data = ele_data.loc['Al':'Cu', 'Atomic Number':'STP Phase']
print(type(sub_data))
sub_data
