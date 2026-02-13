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
# id: Lecture18_spline_interpolation
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Spline interpolation
#
# It turns out there are more sophisticated ways to guess the intermediate values than assuming a linear trend. One such method is called "cubic spline." This is actually one of the most common interpolation schemes, and so it can be accessed using the `kind` keyword argument of `interp1d`:
#
# We can now use it exactly the same way we did with the linear fit:
#
# There is additional "curviness" to the line now, which is calculated by looking at points beyond just the nearest two; the name "cubic spline" means that second and third derivatives are also taken into account.
#
# Just for your information, there are also other options for the `kind` keyword argument, including a "zero-order spline" which is the same as the very first interpolation strategy we tried, just assuming the same as the nearest value:

# %%
spline = interpolate.interp1d(x, y, kind='cubic')
print(spline)

x_hat = np.linspace(data[xs].min(), data[xs].max(), 1000)
y_hat = spline(x_hat)

ax2.plot(x_hat, y_hat, '.', c='tab:orange', label='Cubic')
ax2.legend()
fig2

zero = interpolate.interp1d(x, y, kind='zero')
y_hat = zero(x_hat)

ax2.plot(x_hat, y_hat, '-', c='tab:green', label='Zero')
ax2.legend()
fig2
