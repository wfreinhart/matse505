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
# id: Lecture01_defining_functions
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Defining functions
#
# Using functions from open source modules will save us a lot of time and effort by providing well known formulas and functionalities without us having to design them ourselves.
# Here's how we can define our own functions with `def`:
#
# This function does not send the value back to us, it only prints something. We can make this behavior more explicit like so:
#
# To keep the value, we need to use `return`:

# %%
# def is the python keyword
# my_function is the name
# arg is a positional argument
def my_function(arg):
    # do something with the argument
    print(arg)
    # the extent is indicated by indent

# call the function outside the indent
out = my_function('hello')
print(out)

out = print('hello')
print(out)

out = my_function('hello')
print(f'out = {out}')

def my_function(arg):
    arg_plus = arg + ' from my_function'
    print(arg_plus)
    return arg_plus  # this gets sent back to the main program

out = my_function('hello')  # out will be set to `arg_plus`
print(out)
