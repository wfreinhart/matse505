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
# id: Lecture18_problems_with_real_data
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Problems with real data
#
# This function will only be well defined when there is a single $y \to x$ mapping. Let's consider what this means graphically:
#
# If we ask our function $f^{-1}(y)$ to return $x$ for $y = 30$, which $x$ should it choose? Let's see how `interp1d` behaves here:
#
# Ok, that's not what we want at all. If the function is multi-valued, the behavior can be erratic.

# %%
x = data[xs]
y = data[ys]

spline = interpolate.interp1d(x, y, kind='cubic')

x_hat = np.linspace(x.min(), x.max(), 1000)
y_hat = spline(x_hat)

fig, ax = plt.subplots()
ax.plot(x_hat, y_hat, '-', c=light_blue)
ax.set_xlabel(xs)
ax.set_ylabel(ys)
ax.set_title('Compression of architected material')

y_target = 30  # in MPa
ax.plot([data[xs].min(), data[xs].max()], [y_target, y_target],
        linestyle='--', c='tab:orange')

inv_spline = interpolate.interp1d(y, x, kind='cubic')

y_hat = np.linspace(y.min(), y.max(), 1000)
x_hat = inv_spline(y_hat)

fig, ax = plt.subplots()
ax.plot(y, x, '.', c=light_blue)
ax.plot(y_hat, x_hat, '-', c=light_blue)
ax.set_xlabel(ys)
ax.set_ylabel(xs)
ax.set_title('Compression of architected material')
