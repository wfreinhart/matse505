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
# id: Lecture18_a_simple_example
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## A simple example
#
# Consider the function $y = f(x) = x^3$. Say we want to find $f^{-1}(y)$.
# We know the analytical solution: $f^{-1}(y) = x^{1/3}$.
#
# Let's make a spline interpolation and check it against the true values:
#
# Now it turns out we can actually use our `interp1d` to model $f^{-1}$ too. All we need to do is swap the order of $x$ and $y$!
#
# It doesn't work quite as well as the other way around because this function is $x = y^{-1/3}$, whereas the cubic spline is using polynomials the $x, x^2, x^3$ of which one of them is the true function!
#
# Anyways, we can still check that it works:
#
# We can see there is some error here, but it does get us close without knowing anything about the analytical solution.

# %%
x = np.linspace(-1, 1, 21)
y = x**3

fig, ax = plt.subplots()
ax.plot(x, y, '.')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

x_hat = np.linspace(-1, 1, 101)
x3_spline = interpolate.interp1d(x, y, kind='cubic')
y_hat = x3_spline(x_hat)
ax.plot(x_hat, y_hat, '-')
fig

# note the order! y then x:
y3_spline = interpolate.interp1d(y, x, kind='cubic')

y_hat = np.linspace(-1, 1, 101)
x_hat = y3_spline(y_hat)  # call on y data!

fig, ax = plt.subplots()
ax.plot(y, x, '.')
ax.set_xlabel('$y$')  # flipped!
ax.set_ylabel('$x$')

ax.plot(y_hat, x_hat, '-')

x_test = 0.67
print(f'Spline f^-1 = {y3_spline(x_test)}')
print(f'True f^-1   = {x_test**(1/3)}')
