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
# id: Lecture02_back_to_plotting
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Back to plotting
#
# What are the axes of this chart?
# It's useless without them.
#
# Now it's properly labeled, but does it actually communicate anything?
# The $x$ axis is just the row in the `data` variable, but doesn't correspond to anything real.
# Let's check if there is a trend between $x$ (`Atomic Mass`) and $y$ (`Vacancy Formation Energy (eV)`) instead.
#
# This does look like it shows a positive trend between the data, but it's hard to read because the points are connected by lines.
# A line chart is not appropriate for this data because there is no ordering to these points!
# Let's switch to markers instead:
#
# Finally, we get a meaningful chart that shows a weak positive correlation between these two elemental properties.

# %%
fig, ax = plt.subplots()  # create a blank figure
ax.plot(x)

ax.set_ylabel(x.name)  # a nice shortcut using Series attributes!
ax.set_xlabel('Index')

fig  # show the updated figure

fig, ax = plt.subplots()  # create a blank figure
ax.plot(x, y)
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

fig, ax = plt.subplots()  # create a blank figure
ax.plot(x, y, linestyle='none',  # you can break to a new line!
              marker='*')        # this sometimes improves readability
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
