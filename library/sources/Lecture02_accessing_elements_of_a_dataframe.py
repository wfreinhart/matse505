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
# id: Lecture02_accessing_elements_of_a_dataframe
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Accessing elements of a `DataFrame`
#
# What if we want to access only one of these columns instead of the whole row?
# We can stack indices together using `,` to index on each dimension: first row, then column.
#
# We can also use the `iloc` command to reference the rows and columns by number instead of by name.
# Note that the columns are (0: `Symbol`, 1: `Bulk Static Energy (eV)`, 2: `Reference Energy (eV)`, ...) and so on.
#
# The difference between `loc` and `iloc` is a little confusing because the `index` in our `DataFrame` is numerical.
# Let's try loading it again with the `index_col` keyword of `read_csv`.
# From the documentation, we can see the following information about `index_col`:
#
# > `index_col`: Column(s) to use as the row labels of the DataFrame, either given as string name or column index ...
#
# In this case it makes the most sense to use the `Element` symbols as the index.
# Let's try it out and see what happens:
#
# Now if we want to access information about **Ag** we can use its symbol instead of looking at the table to find its `index`.
# This is often much more convenient.
# Let's try it out:

# %%
# using `loc[]` with [index, column]
print(data.loc[0, 'Symbol'])         # <- the elemental Symbol in row 0
print(data.loc[2, 'Atomic Number'])  # <- the Atomic Number in row 2 (for Al)

# using `iloc[]` with [index_integer, column_integer]
print(data.iloc[0, 0])
print(data.iloc[0, 1])
print(data.iloc[0, 2])

# Load again with index_col using the same smart path logic
if os.path.exists(local_path):
    ele_data = pd.read_csv(local_path, index_col='Symbol')
else:
    ele_data = pd.read_csv(github_url, index_col='Symbol')

ele_data                                                # show a view of the data file

# using `loc[]` with [index][column]
print(ele_data.loc['Ag', 'Atomic Number'])

# using `iloc[]` with [index_integer, column_integer]
print(ele_data.iloc[1, 5])
