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
# id: Lecture01_python_syntax
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# # Python syntax
#
# With the basics out of the way, let's talk about how to actually write and run Python code!
#
# [Python syntax](https://www.w3schools.com/python/python_syntax.asp) will follow a recipe like `variable_name = value` to create or modify variables, `function(variable)` to run functions or methods, or `#` to leave a comment.
# Here's an example with all three elements:
#
# **Note:** The grey box indicates a *Code Cell* as opposed to the *Text Cell* that we've been using so far.
# It's actual Python code and can be run directly in your browser (thanks to Google Colab) to see the result.
#
# Go ahead and run the code above by clicking the arrow that appears inside the brackets `[ ]` on the left side of the cell or by pressing `Shift + Enter` with the cell selected.
# It should just give the same answer again because we stored the output.

# %%
# let's print a number:
x = 5
print(x)
