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
# id: Lecture02_reading_with_pandas
# type: Foundational
# parent_lecture: Lecture02
# ---
#
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
#
# This looks a lot like the tables in Excel.
# On the left side are row numbers which are called `indices`.
# Across the top are the names of `columns`.
# Each pair of `index` and `column` holds one entry of the table.

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
