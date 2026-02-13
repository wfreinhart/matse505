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
# id: Lecture18_interp1d
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## `interp1d`
#
# Of course there's an easier way that doing this by hand each time, even easier than using a `for` loop. Let's use `scipy.interpolate.interp1d` for this purpose.
#
# We get a *function* back from our `interp1d` call. We can pass it x values and it will return y values:
#
# As we see, this matches our calculation done by hand above. The advantage of this function is we can also pass a whole array at once:
#
# If we zoom in as we did before, we see the effect of our interpolation in between the observed data points:

# %%
from scipy import interpolate

linear = interpolate.interp1d(data[xs], data[ys])
print(linear)

print(linear(0.253))

x_hat = np.linspace(data[xs].min(), data[xs].max(), 1000)
y_hat = linear(x_hat)

fig2, ax2 = plt.subplots()
ax2.plot(x, y, 'o', c=light_blue, label='Data')
ax2.plot(x_hat, y_hat, '.', c=light_blue, label='Linear')
ax2.set_xlabel(xs)
ax2.set_ylabel(ys)
ax2.set_title('Compression of architected material')
ax2.legend()

ax2.set_xlim(0.22, 0.28)  # zoom in on 0.22 to 0.28 strain
ax2.set_ylim(5, 15)       # zoom in on 5 to 15 MPa stress
fig2
