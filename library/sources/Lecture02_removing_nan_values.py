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
# id: Lecture02_removing_nan_values
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Removing NaN values
#
# What happened? The error message says:
# > `ValueError: array must not contain infs or NaNs`
#
# If you look back at the table above, you will see some entries with `NaN`.
# This stands for "**N**ot **a** **N**umber".
# Here it means there was no entry for this cell in the original data file.
# If there is no entry, the `pearsonr` function cannot include it in the calculation and it returns a `ValueError` for the user to deal with.
#
# We can use the `np.isfinite` function to find these problematic entries (returns `False` for `nan` and `inf` values, as requested by `pearsonr`).
#
# You can see here we dropped 10 rows from the `DataFrame`.
# Now we can resume our calculations of correlation:
#
# We see that the numerical values agree with our qualitative assessment from the top:
#
# It looks like `Bulk Static Energy` and `Vacancy Formation Energy` have a strong negative correlation, but `Atomic Mass` does not correlate strongly with either of these.
#
# Actually now we can also see that the level of correlation is similar for `Atomic Mass` with both the other properties, though the sign is opposite.
#
# We can do something similar using the `dropna` method of the `DataFrame`.
# This is designed to "drop" (remove) the "na" (NaN) values from the `DataFrame`.
# Let's try it:
#
# Wait, we only have 9 rows. What happened?
# Well, `dropna` drops rows with *any NaN value*.
# There must have been NaN in other columns besides the `'Bulk Static Energy (eV)'` that got dropped.
#
# Let's try a different strategy: first create a subset of our data, then use `dropna` on that.
#
# We can see that we have downselected from 52 rows to 42 rows, and only kept 3 columns.
# This number of rows corresponds to the result from earlier using `np.isfinite`.
# Now we can repeat our previous calculation:
#
# Of course we don't always have to make a new `DataFrame` with only a few columns.
# The other option is to use the `subset` keyword, indicating which columns should be considered when looking for `NaN`.
# All the columns will be kept, and only rows with `NaN` in the `subset` will be dropped.
#
# We can check that we get the same value using `pearsonr`:

# %%
indices = np.isfinite( data.loc[:, 'Bulk Static Energy (eV)'] )
data_filtered = data.loc[indices, :]
print(data.shape, '->', data_filtered.shape)

x = data_filtered[x.name]
y = data_filtered[y.name]
z = data_filtered[z.name]

r, p = stats.pearsonr(x, y)
print(r)

r, p = stats.pearsonr(x, z)
print(r)

r, p = stats.pearsonr(y, z)
print(r)

data_drop = data.dropna()
print(data_drop.shape)

# create a DataFrame with 3 columns:
xyz_data = data.loc[:, [x.name, y.name, z.name]]

# drop NaN from this smaller DataFrame:
xyz_data = xyz_data.dropna()

# print the shape of the resulting DataFrame:
print(data.shape, '->', xyz_data.shape)

r, p = stats.pearsonr(xyz_data.loc[:, x.name], xyz_data.loc[:, y.name])
print(r)

data_drop = data.dropna(subset=[x.name, y.name, z.name])
print(data_drop.shape)

r, p = stats.pearsonr(data_drop.loc[:, x.name], data_drop.loc[:, y.name])
print(r)
