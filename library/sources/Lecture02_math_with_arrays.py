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
# id: Lecture02_math_with_arrays
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Math with arrays
#
# In addition to math functions, NumPy can compute all sorts of statistics as well.
# Let's take a look:
#
# However, one common use case for these functions will be computing statistics for each column in a `DataFrame`.
# Rather than compute them all individually or even using a `for` loop, we can use the `axis` keyword argument.
# Let's start by making `my_data` into a matrix using `reshape`:
#
# We can verify that calling `np.mean` on the whole matrix returns the same value, `0.1815`:
#
# We can also take these statistics over rows or columns with `axis`.
# This keyword argument indicates which *direction* the mean should be taken over.
# So `axis=0` indicates taking the mean over a column (i.e., collect values from the up-down direction):
#
# Note that `axis=0` returns 5 values while `axis=1` returns 4 values, consistent with the 5 columns and 4 rows of the matrix:
#
# You can think of this operation as eliminating the shape in that spot of `shape`.
# For instance, with `axis=0`, we get a `(,5)`-shaped object (eliminating `shape[0]` which is 4):
#
# Whereas for `axis=1` we get a `(4,)`-shaped object (eliminating `shape[1]` which is 5):

# %%
# I generated this list of random numbers ahead of time
my_data = np.array([-0.44,  0.77,  0.6 ,  0.94,  2.1 , -0.74,  0.52, -0.67,  0.18, -0.24, -1.45, -1.55,  1.52, -0.21,  0.3 ,  0.38,  0.82,  0.57, -0.49,  0.72])
print(my_data)
print(f'Mean:   {np.mean(my_data)}')  # the f is called f-string
print(f'Median: {np.median(my_data):.4f}')  # :.4f gives 4 digits after decimal
print(f'StDev:  {np.std(my_data):.4f}')

my_matrix = np.reshape(my_data, [4, 5])
print(my_matrix)

print(f'The matrix mean is: {np.mean(my_matrix)}')
print(f'mean(my_matrix) == mean(my_data): {np.mean(my_matrix) == np.mean(my_data)}')

print(f'Column mean: {np.mean(my_matrix, axis=0)}')
print(f'Row mean: {np.mean(my_matrix, axis=1)}')

print(my_matrix.shape)

print(np.mean(my_matrix, axis=0).shape)

print(np.mean(my_matrix, axis=1).shape)
