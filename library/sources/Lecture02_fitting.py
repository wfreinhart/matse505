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
# id: Lecture02_fitting
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Fitting
#
# Pearson R tells us about the strength of correlation but does not give us a model. Let's use `stats.linregress` to do a proper linear regression:
#
# What does this `result` mean? We can see that it is a special data structure called `LinregressResult`. The fields can be accessed using the `.` notation. Let's take a look at the model using these fields:
#
# Note that the `rvalue` is the same as the one we found with `pearsonr`. Also note that the slope is negative, which matches the outcome of the fitted `slope` being negative.
#
# Now let's try plotting this result:

# %%
y_clean = xyz_data.loc[:, y.name]
z_clean = xyz_data.loc[:, z.name]

result = stats.linregress(y_clean, z_clean)
print(result)

print(f'linear model is y = {result.slope} x + {result.intercept}')
print(f'measured correlation is {result.rvalue}')

z_model = result.slope * y + result.intercept

fig, ax = plt.subplots()

ax.plot(y, z, '.', label='Observations')
ax.plot(y, z_model, label='Regression')
ax.legend()  # add the legend from labels

ax.set_xlabel(y.name)  # a shortcut to get the axis label!
ax.set_ylabel(z.name)
