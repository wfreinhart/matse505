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
# id: Lecture18_linear_interpolation
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Linear interpolation
#
# A better strategy that might be even more intuitive than the one we just described is to assume a straight line between each data point. This is what `pyplot` does when we make a line plot from discrete data points:
#
# That looks a lot smoother and more natural. We can formalize this beyond drawing the chart to actually find arbitrary values, like our 0.253 strain example. Let's use this equation:
#
# $\hat{y} = \frac{y_{i+1} - y_i}{x_{i+1} - x_i} (\hat{x} - x_i) + y_i$
#
# We see that between 0.250 and 0.258 the stress decreases from 11.75 to 10.33 MPa, so this result of 11.24 MPa makes sense.

# %%
fig2, ax2 = plt.subplots()
ax2.plot(data[xs], data[ys], c=light_blue)
ax2.set_xlabel(xs)
ax2.set_ylabel(ys + '(interpolated)')
ax2.set_title('Compression of architected material')

i = np.argmin(np.abs(data[xs] - 0.253))  # find the index of closest x value
print(data.iloc[idx:(idx+2), :])           # print the corresponding row

x = data[xs]
y = data[ys]

x_hat = 0.253  # this is our target strain value

slope = (y[i+1] - y[i]) / (x[i+1] - x[i])
y_hat = y[i] + (x_hat - x[i]) * slope
print(y_hat)  # this is our interpolated stress value
