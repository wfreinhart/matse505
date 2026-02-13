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
# id: Lecture01_functions
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# # Functions
#
# [Functions](https://www.w3schools.com/python/python_functions.asp) do something to variables.
# You may have learned in math about functions like $f(x) = y$.
# The Python version of a function is exactly the same, it produces some output $y$ from some input $x$.
#
# There are a bunch of built-in functions including `print` that we have already seen.
# Let's try using print in a few different ways:
#
# Functions are called by writing the name followed by `()`.
# The function `arguments` go inside the parentheses.
# Some functions take no arguments and would look like this: `my_function()`.
#
# Another builtin function is `type`:
#
# Note that `x` is an integer because its assigned value did not include a decimal point.
# If we use a decimal point, it will become a floating point number (`float`):

# %%
print(x)            # this will write "1" (from x = 1 above)
print(my_variable)  # this will write "abc" (from my_variable = 'abc' above)
print(5)
print("some text")  # we can also call print on an input that's not a variable
print(x, 'text')    # print can take multiple arguments separated by commas

print(print)

type(x)  # x should be of type "integer," or "int" for short

x = 1.0
type(x)
