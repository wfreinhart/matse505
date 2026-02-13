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
# id: Lecture18_multi_valued_functions
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Multi-valued functions
#
# To fix this, we need to use logical indexing and create a localized interpolation. Let's say we only want the value up to the first maximum.
#
# Now we can reliably invert that first part of the function:

# %%
top_strain = x[np.argmax(y)]  # find peak location
print(f'peak occurs at strain = {top_strain}')

before_top = x <= top_strain

inv_spline = interpolate.interp1d(y[before_top], x[before_top], kind='cubic')

y_hat = np.linspace(y[before_top].min(), y[before_top].max(), 1000)
x_hat = inv_spline(y_hat)

fig, ax = plt.subplots()
ax.plot(y, x, '.', c=light_blue)
ax.plot(y_hat, x_hat, '-', c=light_blue)
ax.set_xlabel(ys)
ax.set_ylabel(xs)
ax.set_title('Compression of architected material')

y_target = 30
print(f'Strain for {y_target} MPa Stress = {inv_spline(30)}')
