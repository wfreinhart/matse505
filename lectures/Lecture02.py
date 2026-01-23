# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# > Reminder: Homework 1 due Tuesday. You will need to do a little reading on your own to answer Problem 1 (such as either of the suggested textbooks or other online resources).

# %% [markdown]
# Today's topics:
# * Reading data with `pandas`
# * Plotting data with `pyplot`
# * Analyzing data with `numpy`
# * Modeling data with `scipy`

# %% [markdown]
# > Note: This lesson is intended to be a rapid-fire session full of examples for you to review on your own. Please refer to the detailed notes under the `Resources` tab on Canvas for additional material to review on specific topics.

# %% [markdown]
# # Reading data with `pandas`

# %% [markdown]
# ## Downloading from a URL
#
# We first have to acquire a data file.
# This short code downloads the data stored at a URL (hosted on OneDrive via Sharepoint, as you can see in the URL).
# It then writes the data to a `csv` file that we can later read.
#
# > If you already had a `csv` file this step would not be needed. I include it here to avoid having to distribute the file separately from the notebook.

# %%
import requests

# # define the url
# url = 'https://pennstateoffice365-my.sharepoint.com/:x:/g/personal/wfr5091_psu_edu/EU5JYKhddWRLhNaq_frzFS0BJOz9cXZTtxx-zKGJQEhVnw?e=jOmqFq&download=1'
# 
# # fetch the data stored at the url
# r = requests.get(url)
# 
# # write the data to the local file system
# with open('data.csv', 'w') as fid:
#     fid.write(r.text)

# %% [markdown]
# ## Reading with pandas
#
# *Why do I need a special module to read data?*
#
# Data can be stored in many forms.
# We can write our own programs to read special data formats, but for the most common ones it would be redundant.
# Not to mention that we probably couldn't be as thorough as the team of developers working on this specialized project!
# So we might as well take advantage of Python's free and open source modules.
#
# *What can Pandas do?*
#
# First and foremost, Pandas gives us a library for common I/O.
# This includes reading and writing `csv`, `xlsx` (Excel), and other widespread formats.
# Beyond I/O Pandas gives us convenient ways to store, filter, and analyze our data.
# Overall, it makes Python data structures look more like the user experience in Excel.
#
# Now let's use Pandas to read the data from that `csv` file.
# If we search `"pandas read csv"` on Google, we find the [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) function.
# From the documentation, it looks like the signature is:
#
# > `pandas.read_csv(filepath_or_buffer, ...`

# %%
import pandas as pd
import os

# Set the path to the data file
filename = 'elements.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    data = pd.read_csv(github_url)

data                            # show a view of the data file

# %% [markdown]
# This looks a lot like the tables in Excel.
# On the left side are row numbers which are called `indices`.
# Across the top are the names of `columns`.
# Each pair of `index` and `column` holds one entry of the table.

# %% [markdown]
# ## DataFrames
#
# *What is `type` of this variable that Pandas created?*
#
# Let's learn more about how this is stored using the `type` builtin function:

# %%
print(type(data))

# %% [markdown]
# Alright, it looks like this is a `pandas DataFrame`.
# If you Google this it will take you to the [documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
# This lists all the `Attributes` and `Methods` that are built into the `DataFrame` object type.
# For instance, we can see that there are several different `Attributes` that let us access specific data elements:
#
# * `at`: Access a single value for a row/column label pair.
# * `loc`: Access a group of rows and columns by label(s) or a boolean array.
# * `iat`: Access a single value for a row/column pair by integer position.
# * `iloc`: Purely integer-location based indexing for selection by position.
#
# It turns out that `loc` can basically act as `at` by just specifying a single element while the reverse is not true, so we'll just stick to `loc`.
# > Note: it's very common to find multiple different ways to achieve an outcome! Use whichever method you prefer.

# %% [markdown]
# Now that we know to use `loc`, we can achieve the result we were looking for:

# %%
data.loc[0]

# %% [markdown]
# Inside this `0` index is an entire row (vector) of data (making the entire `data` table above a matrix).
# It turns out we can also use slices just as we did with `list`:

# %%
data.loc[:5]

# %%
data.loc[0:10:2]

# %% [markdown]
# ## Accessing elements of a `DataFrame`
#
# What if we want to access only one of these columns instead of the whole row?
# We can stack indices together using `,` to index on each dimension: first row, then column.

# %%
# using `loc[]` with [index, column]
print(data.loc[0, 'Symbol'])         # <- the elemental Symbol in row 0
print(data.loc[2, 'Atomic Number'])  # <- the Atomic Number in row 2 (for Al)

# %% [markdown]
# We can also use the `iloc` command to reference the rows and columns by number instead of by name.
# Note that the columns are (0: `Symbol`, 1: `Bulk Static Energy (eV)`, 2: `Reference Energy (eV)`, ...) and so on.

# %%
# using `iloc[]` with [index_integer, column_integer]
print(data.iloc[0, 0])
print(data.iloc[0, 1])
print(data.iloc[0, 2])

# %% [markdown]
# The difference between `loc` and `iloc` is a little confusing because the `index` in our `DataFrame` is numerical.
# Let's try loading it again with the `index_col` keyword of `read_csv`.
# From the documentation, we can see the following information about `index_col`:
#
# > `index_col`: Column(s) to use as the row labels of the DataFrame, either given as string name or column index ...
#
# In this case it makes the most sense to use the `Element` symbols as the index.
# Let's try it out and see what happens:

# %%
# Load again with index_col using the same smart path logic
if os.path.exists(local_path):
    ele_data = pd.read_csv(local_path, index_col='Symbol')
else:
    ele_data = pd.read_csv(github_url, index_col='Symbol')

ele_data                                                # show a view of the data file

# %% [markdown]
# Now if we want to access information about **Ag** we can use its symbol instead of looking at the table to find its `index`.
# This is often much more convenient.
# Let's try it out:

# %%
# using `loc[]` with [index][column]
print(ele_data.loc['Ag', 'Atomic Number'])

# using `iloc[]` with [index_integer, column_integer]
print(ele_data.iloc[1, 5])

# %% [markdown]
# ## Indexing practice
#
# As we said earlier, in addition to accessing single values, `loc` can access ranges of values.
# This is done with `:` as we saw with `lists`.
# When we use `:`, we get the elements starting with the number to the left and ending with one less than the number on the right.
# So `0:3` accesses elements at indices `[0, 1, 2]`.
# Let's try it out:

# %%
print(ele_data.iloc[0, 0:3])

# %% [markdown]
# We could get all the elements in either a row or column by not specifying any numbers to the side of the `:`:

# %%
print(ele_data.iloc[0, :])  # access a whole column by specifying a row only
print('')  # give a line break between print statements
print(ele_data.iloc[:, 0])  # access a whole row by specifying a column only

# %% [markdown]
# We can also use `:` with the index and column names, which is more intuitive -- in this case pandas actually includes the ending entry too:

# %%
print(ele_data.loc['Al':'Cu', 'Atomic Number':'STP Phase'])

# %% [markdown]
# **IMPORTANT:**
# Note that in this case pandas actually returns a `DataFrame` object which is a subset of the larger `ele_data` `DataFrame`.
#
# Pandas formats this nicely if we drop the `print` statement:

# %%
sub_data = ele_data.loc['Al':'Cu', 'Atomic Number':'STP Phase']
print(type(sub_data))
sub_data

# %% [markdown]
# ## [Check your understanding]
#
# In the code cell below, do the following:
# * Use `loc` to print the `Mendeleev Number` of `Ni`
# * Use `loc` to print all `STP Phase` values of `Al`, `Cu`, and `Ni`
# * Use `iloc` to print the `Atomic Number` of `Ba`
# * Use `iloc` to print the `Bulk Static Energy (eV)` and `Cohesive Energy (eV)` of `Au`

# %%

# %% [markdown]
# # Plotting with `pyplot`
#
# Let's start with the basics:

# %%
from matplotlib import pyplot  # import the pyplot submodule

pyplot.plot()  # create an empty plot

# %% [markdown]
# ## A note on aliasing
#
# Before we go any further, let's talk about `aliases` again.
# As you can see from our very first code cell, accessing the functions of the `pyplot` submodule will require us to type `pyplot.<function>` every time.
# In practice, people have all agreed that it's easier to type `plt` instead:

# %%
from matplotlib import pyplot as plt  # use an alias

plt.plot()  # create an empty plot

# %% [markdown]
# Now we can always use `plt` instead of `pyplot`.
# I bring this up because all the tutorials I link to use this notation!
# You can even see it being done this way on the [official pyplot documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot)!

# %% [markdown]
# ## Making a line chart
#
# We should start be referencing the documentation for `plt.plot`:

# %%
help(plt.plot)

# %% [markdown]
# Let's create a line plot by manually defining a `list` of $y$ values.
# The $x$ values will be implicit if not provided (as indicated by the `[x]` above).

# %%
y = [1, 2, 4, 8]
plt.plot( y )  # we provide 4 y values, x will go from 0 to 3

# %% [markdown]
# We can also provide both $x$ and $y$ values:

# %%
plt.plot([1, 2, 3, 10], [2, 4, 6, 10])

# %% [markdown]
# ## Multiple series
#
# We can plot multiple series by calling `plot` twice:

# %%
plt.plot([2, 4, 6, 10], [1, 2, 3, 10])
plt.plot([1, 2, 3, 10], [2, 4, 6, 10])

# %% [markdown]
# However, watch what happens if we don't call this in the same cell:

# %%
plt.plot([2, 4, 6, 10], [1, 2, 3, 10])

# %%
plt.plot([1, 2, 3, 10], [2, 4, 6, 10])

# %% [markdown]
# ## Making persistent `figures`
#
# *What happened to our line?*
#
# You may have noticed we did not create any variables so far, and have only been calling `pyplot` **functions**.
# This is because `pyplot` uses a concept called a "state machine" to store the chart behind the scenes rather than exposing it to the user.
# Executing a new code cell clears the state and begins a new plot.
#
# *What if we want to access our chart again later?*
#
# There is a simple way to get around the default behavior of `pyplot`, which is to explicitly create variables to store the chart.
# Matplotlib calls the elements of a chart the `Figure` and `Axes`, as illustrated in this diagram:
#
# <img src="https://i1.wp.com/ajaytech.co/wp-content/uploads/2019/05/matplotlib-relationship-hierarchy.png?w=712&ssl=1" alt="Diagram of matplotlib Figure classes." height=400/>
#
# (image credit [ajaytech.co](https://ajaytech.co/matplotlib))

# %% [markdown]
# ## Subplots
#
# In order to access the elements of our chart again later, we need to store the `Figure` and/or `Axes` as a variable rather than letting `pyplot` do this with its state machine.
# We can do this using the [`pyplot.subplots`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) command:

# %%
figure, axes = plt.subplots()
print(figure)
print(axes)

# %% [markdown]
# Now we have a persistent chart object that we can modify as we go.
# Let's try repeating the examples we've already seen, except this time by calling the same methods on `axes` instead of using the default `pyplot`:

# %%
axes.plot([1, 2, 3, 5], [2, 4, 6, 10])

# %% [markdown]
# *Where's our figure?*
#
# Because the `Figure` is being stored persistently, `pyplot` doesn't show it to us every time we call a command on it.
# Instead, we can call our variable to show the figure again.

# %%
figure

# %% [markdown]
# ## Why is it called subplots?
#
# `subplots` is meant for making multiple plots in a single `figure`.
# We specify `nrows` and `ncols`:

# %%
fig, ax = plt.subplots(nrows=2, ncols=3)

# %% [markdown]
# For now we'll just use it with the default `nrows=1` and `ncols=1` to make simple charts.

# %%
# plt.subplots?

# %%
fig, ax = plt.subplots()

# %% [markdown]
# # Basic `pyplot` formatting

# %% [markdown]
# ## Line and marker style
#
# We can also switch the type of line and/or marker we use in our chart:

# %%
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], '--')  # use dashed line
ax.plot([2, 5, 6, 10], '.')  # use dots instead of line
ax.plot([3, 4, 5, 4], 's')  # use squares instead of line

# %%
# i want the chart to have a red line
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], 'r--')

# %% [markdown]
# We can also use both lines and markers by specifying their style with keyword arguments `linestyle` and `marker`:

# %%
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], linestyle='-.', marker='d')  # dash-dot linesyle with diamond markers

# %% [markdown]
# You may have noticed above that we changed the linestyle two different ways:
# * `ax.plot([1, 2, 3], '--')`
# * `ax.plot([1, 2, 3], linestyle='--')`
#
# This is because python supports two different kind of function arguments: **positional** and **keyword** arguments.
# Positional means the variable to set is determined by the position in the list, while keyword indicates this explicitly using `value=`.
# Here's an example of a function signature from `numpy`:
# > `numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)`
#

# %% [markdown]
# ## Adding labels
#
# We can specify labels, titles, and other formatting elements by reading the [documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot) or following a [tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html):

# %%
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 5], [2, 4, 6, 10])
ax.set_title('An example pyplot chart')
ax.set_xlabel('x')
ax.set_ylabel('y')

# %% [markdown]
# ## Adding legends
#
# Legends are essential for good charts (with multiple series).
# Here is my recommended way to add a legend using `ax.legend`:

# %%
fig, ax = plt.subplots()

ax.plot([1, 2, 3, 5], [2, 4, 6, 10], label='Series 1')  # label will be used by legend() later
ax.plot([1, 2, 3, 4], [3, 4, 5, 6], label='Series 2')

ax.legend()  # create the legend based on `label`

# %% [markdown]
# ## [Check your understanding]
#
# In the code cell below:
# * Create lists for $x$ and $y$ data -- make sure they're the same length
# * Create a `figure` and `axis` using `subplots`
# * Add a line plot for $y$ vs $x$, use a green (`'g'`) line color with "dash-dot" style (`'-.'`)
# * Add a title and axis labels
#
# > You could reference [this link](https://matplotlib.org/stable/gallery/color/named_colors.html) for all the named colors

# %%

# %% [markdown]
# On the same `axis`...
# * Add a plot for $y$ vs $x$, except this time use blue square markers with no line
# * Add a legend indicating "line" and "markers" data series
#
# > Make sure to call the `figure` again so it shows the updated version

# %%

# %% [markdown]
# # Plotting data from a `DataFrame`

# %% [markdown]
# ## A shortcut
#
# We can start by defining some data of interest.
# I highly recommend a more declarative style of programming with variables defined whenever something will be reused.
# This makes it easier to read and debug.
# You can always optimize for speed later (we'll discuss...).

# %%
x = data.loc[:, 'Atomic Mass']             # one option for indexing
y = data['Vacancy Formation Energy (eV)']  # another option for indexing
z = data.loc[:, 'Bulk Static Energy (eV)'] # I prefer this one -- more explicit

# %% [markdown]
# Let's take a second to investigate these new variables:

# %%
print(type(x))

# %% [markdown]
# A `Series` is kind of like a `DataFrame` but only for one column.
# It retains some useful features:

# %%
print(x.name)    # an attribute
print(x.shape)   # another attribute
print(x.mean())  # a method (albeit with a static value)

# %% [markdown]
# ## Back to plotting

# %%
fig, ax = plt.subplots()  # create a blank figure
ax.plot(x)

# %% [markdown]
# What are the axes of this chart?
# It's useless without them.

# %%
ax.set_ylabel(x.name)  # a nice shortcut using Series attributes!
ax.set_xlabel('Index')

fig  # show the updated figure

# %% [markdown]
# Now it's properly labeled, but does it actually communicate anything?
# The $x$ axis is just the row in the `data` variable, but doesn't correspond to anything real.
# Let's check if there is a trend between $x$ (`Atomic Mass`) and $y$ (`Vacancy Formation Energy (eV)`) instead.

# %%
fig, ax = plt.subplots()  # create a blank figure
ax.plot(x, y)
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

# %% [markdown]
# This does look like it shows a positive trend between the data, but it's hard to read because the points are connected by lines.
# A line chart is not appropriate for this data because there is no ordering to these points!
# Let's switch to markers instead:

# %%
fig, ax = plt.subplots()  # create a blank figure
ax.plot(x, y, linestyle='none',  # you can break to a new line!
              marker='*')        # this sometimes improves readability
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

# %% [markdown]
# Finally, we get a meaningful chart that shows a weak positive correlation between these two elemental properties.

# %% [markdown]
# ## Histograms
#
# Let's quickly add some additional plot styles to our repertoire.
# **Histograms** can be created using `plt.hist`.
# Remember that histograms visualize distributions of a single variable, so it takes only one variable as input:

# %%
fig, ax = plt.subplots()
ax.hist(x)
ax.set_xlabel(x.name)
ax.set_ylabel('Frequency')

# %% [markdown]
# ## Bar charts
#
# **Bar charts** look like histograms but show labeled values instead of the distribution of values.
# The `plt.bar` function takes arguments `(labels, values)` like so:

# %%
sym = data['Symbol']

fig, ax = plt.subplots()
ax.bar(sym, y)
ax.set_xlabel(sym.name)
ax.set_ylabel(y.name)

# %% [markdown]
# This is a little crammed so let's expand the width of the figure to read the symbols more clearly.

# %%
fig.set_figwidth(12)

# %% [markdown]
# What happened?
# Green checkmark means the code executed.
# But we didn't ask for it to show us anything -- we need to render the figure again:

# %%
fig

# %% [markdown]
# ## Scatter
#
# We can add color to our charts with `scatter`, by specifying a `c` keyword argument (short for "color").

# %%
fig, ax = plt.subplots()

ax.scatter(x, y, c=z)

ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

# %% [markdown]
# This isn't helpful without a colorbar indicating the values of the colors and a label to describe it:

# %%
fig, ax = plt.subplots()

im = ax.scatter(x, y, c=z)  # save the output here (scatter object)

cb = plt.colorbar(im)  # save the output again (colorbar object)
cb.set_label(z.name)   # label the colorbar

ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

# %% [markdown]
# ## Using `pandas` builtins
#
# I personally don't use `pandas` for plotting, but you should know its objects include handy wrappers for `pyplot`.
#
# For example, a method of the `Series` object:

# %%
x.hist()

# %% [markdown]
# We can plot multiple `Series` using a method of the `DataFrame`:

# %%
data.plot.bar('Symbol', 'Vacancy Formation Energy (eV)')

# %% [markdown]
# And finally we can introduce the (x,y)+color scatter plot from above:

# %%
data.plot.scatter(x.name, y.name, c=z.name)

# %% [markdown]
# Note the `type` that is returned -- `matplotlib.axes._subplots.AxesSubplot`!
# All this is doing is sending some predefined commands to `pyplot`.
# If you find this syntax helpful you are more than welcome to use it.
# I don't simply because I don't always use `DataFrame` objects to store my data.

# %% [markdown]
# ## [Check your understanding]
#
# In the code cell below:
# * Create a plot with markers that visualizes the relationship between any two elemental properties in the `DataFrame`.
# * Create a bar chart that visualizes the values of any one elemental property in the `DataFrame`.
# * Create a histogram that visualizes the distribution of any one elemental property in the `DataFrame`.
#
# (you can put them all together in one code cell or use separate code cells for each)
#
# > Tip: to review which columns are present in the `DataFrame`, you could look at the `columns` attribute
#

# %%

# %% [markdown]
# # Analysis with `numpy`
#
# NumPy is a popular library for scientific computing that contains fast and convenient functions for linear algebra.
# Let's start by importing the [`numpy` module](https://numpy.org/doc/stable/).
# This will give access to all the features the `numpy` authors have implemented for us.
# Let's use the `as` keyword to keep the module name short again.
# Trust me, this will be worth it -- we are going to reference `numpy` in almost every line!

# %%
import numpy as np

# %% [markdown]
# ## Math with arrays

# %% [markdown]
# In addition to math functions, NumPy can compute all sorts of statistics as well.
# Let's take a look:

# %%
# I generated this list of random numbers ahead of time
my_data = np.array([-0.44,  0.77,  0.6 ,  0.94,  2.1 , -0.74,  0.52, -0.67,  0.18, -0.24, -1.45, -1.55,  1.52, -0.21,  0.3 ,  0.38,  0.82,  0.57, -0.49,  0.72])
print(my_data)
print(f'Mean:   {np.mean(my_data)}')  # the f is called f-string
print(f'Median: {np.median(my_data):.4f}')  # :.4f gives 4 digits after decimal
print(f'StDev:  {np.std(my_data):.4f}')

# %% [markdown]
# However, one common use case for these functions will be computing statistics for each column in a `DataFrame`.
# Rather than compute them all individually or even using a `for` loop, we can use the `axis` keyword argument.
# Let's start by making `my_data` into a matrix using `reshape`:

# %%
my_matrix = np.reshape(my_data, [4, 5])
print(my_matrix)

# %% [markdown]
# We can verify that calling `np.mean` on the whole matrix returns the same value, `0.1815`:

# %%
print(f'The matrix mean is: {np.mean(my_matrix)}')
print(f'mean(my_matrix) == mean(my_data): {np.mean(my_matrix) == np.mean(my_data)}')

# %% [markdown]
# We can also take these statistics over rows or columns with `axis`.
# This keyword argument indicates which *direction* the mean should be taken over.
# So `axis=0` indicates taking the mean over a column (i.e., collect values from the up-down direction):

# %%
print(f'Column mean: {np.mean(my_matrix, axis=0)}')
print(f'Row mean: {np.mean(my_matrix, axis=1)}')

# %% [markdown]
# Note that `axis=0` returns 5 values while `axis=1` returns 4 values, consistent with the 5 columns and 4 rows of the matrix:

# %%
print(my_matrix.shape)

# %% [markdown]
# You can think of this operation as eliminating the shape in that spot of `shape`.
# For instance, with `axis=0`, we get a `(,5)`-shaped object (eliminating `shape[0]` which is 4):

# %%
print(np.mean(my_matrix, axis=0).shape)

# %% [markdown]
# Whereas for `axis=1` we get a `(4,)`-shaped object (eliminating `shape[1]` which is 5):

# %%
print(np.mean(my_matrix, axis=1).shape)

# %% [markdown]
# ## Central tendency
#
# There are a few key quantities we might calculate for a single `Series`.
# The first category would be "central tendency" like `mean` and `median`:
#

# %%
print(np.mean(x), np.median(x))

# %% [markdown]
# We can get these same values using the `mean()` and `median` methods of the `Series` object:

# %%
print(x.mean(), x.median())

# %% [markdown]
# Note this is just a wrapper for the `numpy` function that utilizes the data stored in the `Series`.
# We can even see this by accessing the `values` attribute of the `Series`:

# %%
print(type(x.values))
print(x.values)

# %% [markdown]
# There it is -- `Series` is just wrapping a `numpy.ndarray` object under the hood!
#
# What about on the whole `DataFrame`?

# %%
print(type(data.values))

# %% [markdown]
# Again, `DataFrame` is storing a bunch of `numpy.ndarray` objects.
#
# Anyways, we can continue to perform calculations with either the `numpy` functions directly or using the wrapper methods:

# %%
np.mean(data, axis=0)

# %% [markdown]
# > The warning is telling us that "nuisance columns" (non-numerical data) are being ignored but that future versions of `pandas` will not follow this behavior.

# %%
data.mean(axis=0)

# %% [markdown]
# ## Variance
#
# Exactly the same except with either `np.var` (variance) or `np.std` (standard deviation):

# %%
print(np.std(x))
print(np.var(x))
print(np.sqrt(np.var(x)))  # demonstrate the definition of std dev

# %% [markdown]
# We can also use the `axis` keyword as always:

# %%
print(data.std(axis=0))

# %% [markdown]
# ## Statistical moments
#
# We might be interested in some higher moments of data.
# Here are the skew and kurtosis for our `Series`:

# %%
print(x.skew())
print(x.kurt())

# %% [markdown]
# These can be computed manually using the definition for [standardized moments](https://en.wikipedia.org/wiki/Standardized_moment):
#
# $\bar{\mu}_3 = E[(X-\mu)^3] / \sigma^3$
#
# $\bar{\mu}_4 = E[(X-\mu)^4] / \sigma^4$
#
# where $\mu$ is the mean and $\sigma$ is the standard deviation.

# %%
mu = np.mean(x)
sigma = np.std(x)
skew = np.mean((x-mu)**3) / sigma**3
print(skew)

kurt = np.mean((x-mu)**4) / sigma**4
print(kurt)

# %% [markdown]
# You will notice that the skew is pretty close, but the kurtosis is way off (even a different sign).
# We can look at the documentation for `kurt` to find out why:

# %%
help(x.kurt)

# %% [markdown]
# The Fisher's definition of kurtosis uses a normal distribution as the baseline, so we need to subtract 3:

# %%
kurt_fisher = np.mean((x-mu)**4) / sigma**4 - 3
print(kurt_fisher)

# %% [markdown]
# ## Correlation
#
# Let's quantify the relationship between several variables using the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
# We can compute this in Python with `scipy.stats.pearsonr`.
# If we look at the docstring for `pearsonr`, we see the following:

# %%
from scipy import stats

help(stats.pearsonr)

# %% [markdown]
# This means we should expect two outputs: `r` and `p-value`.
# We can assign these like we do `figure` and `axis` from `plt.subplots`, using a `,` to separate the two variable assignments:

# %%
r, p = stats.pearsonr(x, y)
print(r)

# %% [markdown]
# If we don't do this, we'll get a `tuple` of outputs (like when we defined our own function with multiple variables after the `return` statement):

# %%
out = stats.pearsonr(x, y)
print(out)
print(type(out))

# %% [markdown]
# Now let's try computing the Pearson R between all pairs of these variables:

# %%
r, p = stats.pearsonr(x, y)
print(r)

r, p = stats.pearsonr(x, z)
print(r)

r, p = stats.pearsonr(y, z)
print(r)

# %% [markdown]
# ## Removing NaN values

# %% [markdown]
# What happened? The error message says:
# > `ValueError: array must not contain infs or NaNs`
#
# If you look back at the table above, you will see some entries with `NaN`.
# This stands for "**N**ot **a** **N**umber".
# Here it means there was no entry for this cell in the original data file.
# If there is no entry, the `pearsonr` function cannot include it in the calculation and it returns a `ValueError` for the user to deal with.
#
# We can use the `np.isfinite` function to find these problematic entries (returns `False` for `nan` and `inf` values, as requested by `pearsonr`).

# %%
indices = np.isfinite( data.loc[:, 'Bulk Static Energy (eV)'] )
data_filtered = data.loc[indices, :]
print(data.shape, '->', data_filtered.shape)

# %% [markdown]
# You can see here we dropped 10 rows from the `DataFrame`.
# Now we can resume our calculations of correlation:

# %%
x = data_filtered[x.name]
y = data_filtered[y.name]
z = data_filtered[z.name]

r, p = stats.pearsonr(x, y)
print(r)

r, p = stats.pearsonr(x, z)
print(r)

r, p = stats.pearsonr(y, z)
print(r)

# %% [markdown]
# We see that the numerical values agree with our qualitative assessment from the top:
#
# It looks like `Bulk Static Energy` and `Vacancy Formation Energy` have a strong negative correlation, but `Atomic Mass` does not correlate strongly with either of these.
#
# Actually now we can also see that the level of correlation is similar for `Atomic Mass` with both the other properties, though the sign is opposite.

# %% [markdown]
# We can do something similar using the `dropna` method of the `DataFrame`.
# This is designed to "drop" (remove) the "na" (NaN) values from the `DataFrame`.
# Let's try it:

# %%
data_drop = data.dropna()
print(data_drop.shape)

# %% [markdown]
# Wait, we only have 9 rows. What happened?
# Well, `dropna` drops rows with *any NaN value*.
# There must have been NaN in other columns besides the `'Bulk Static Energy (eV)'` that got dropped.
#
# Let's try a different strategy: first create a subset of our data, then use `dropna` on that.

# %%
# create a DataFrame with 3 columns:
xyz_data = data.loc[:, [x.name, y.name, z.name]]

# drop NaN from this smaller DataFrame:
xyz_data = xyz_data.dropna()

# print the shape of the resulting DataFrame:
print(data.shape, '->', xyz_data.shape)

# %% [markdown]
# We can see that we have downselected from 52 rows to 42 rows, and only kept 3 columns.
# This number of rows corresponds to the result from earlier using `np.isfinite`.
# Now we can repeat our previous calculation:

# %%
r, p = stats.pearsonr(xyz_data.loc[:, x.name], xyz_data.loc[:, y.name])
print(r)

# %% [markdown]
# Of course we don't always have to make a new `DataFrame` with only a few columns.
# The other option is to use the `subset` keyword, indicating which columns should be considered when looking for `NaN`.
# All the columns will be kept, and only rows with `NaN` in the `subset` will be dropped.

# %%
data_drop = data.dropna(subset=[x.name, y.name, z.name])
print(data_drop.shape)

# %% [markdown]
# We can check that we get the same value using `pearsonr`:

# %%
r, p = stats.pearsonr(data_drop.loc[:, x.name], data_drop.loc[:, y.name])
print(r)

# %% [markdown]
# ## With `pandas` builtins
#
# There is a huge shortcut we can take using `pandas` again:

# %%
xyz_data.corr()

# %% [markdown]
# This returns all the correlations between pairs of columns.
# It is itself a `DataFrame` so we can filter using `loc`:

# %%
xyz_data.corr().loc[:, 'Atomic Mass']

# %% [markdown]
# # Linear regression

# %% [markdown]
# ## Fitting
#
# Pearson R tells us about the strength of correlation but does not give us a model. Let's use `stats.linregress` to do a proper linear regression:

# %%
y_clean = xyz_data.loc[:, y.name]
z_clean = xyz_data.loc[:, z.name]

result = stats.linregress(y_clean, z_clean)
print(result)

# %% [markdown]
# What does this `result` mean? We can see that it is a special data structure called `LinregressResult`. The fields can be accessed using the `.` notation. Let's take a look at the model using these fields:

# %%
print(f'linear model is y = {result.slope} x + {result.intercept}')
print(f'measured correlation is {result.rvalue}')

# %% [markdown]
# Note that the `rvalue` is the same as the one we found with `pearsonr`. Also note that the slope is negative, which matches the outcome of the fitted `slope` being negative.
#
# Now let's try plotting this result:

# %%
z_model = result.slope * y + result.intercept

fig, ax = plt.subplots()

ax.plot(y, z, '.', label='Observations')
ax.plot(y, z_model, label='Regression')
ax.legend()  # add the legend from labels

ax.set_xlabel(y.name)  # a shortcut to get the axis label!
ax.set_ylabel(z.name)

# %% [markdown]
# ## Evaluating goodness of fit
#
# A common metric for goodness of fit is the $R^2$:
#
# $R^2 = 1 - \frac{SS_\mathrm{residual}}{SS_\mathrm{total}}$
#
# where $SS_\mathrm{residual} = \sum_i (y_i - \hat{y}_i)^2$ and $SS_\mathrm{total} = \sum_i (y_i - \mu)^2$ with $\hat{y}$ being the model prediction and $\mu$ being the data mean.
#
# This is also sometimes called "explained variance" because you can use this alternative formulation:
#
# $R^2 = 1 - \frac{ \mathrm{Var}(y_\mathrm{residual}) } { \mathrm{Var}(y_\mathrm{data}) }$
#
#
#
#

# %%
residual = z - z_model
Rsquare = 1 - np.var(residual) / np.var(z)
print(Rsquare)

# %% [markdown]
# We can check that we get the same answer as provided by `stats.linregress`:

# %%
print(result.rvalue**2)

# %% [markdown]
# ## [Check your understanding]
#
# Perform a linear regression for `Cohesive Energy (eV)` versus `Vacancy Formation Energy (eV)` and compute the $R^2$ manually. Then compare to the result of `linregress`.

# %%
