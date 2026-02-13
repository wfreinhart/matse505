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
# id: Lecture01_variables
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Variables
#
# ### Creating variables
#
# Anything we assign a value to becomes a [variable](https://www.w3schools.com/python/python_variables.asp).
# Variables are created when you first assign something to them and persist until the program completes or you manually delete them.
# From that point forward you can reference the variable to access the stored value.
#
# > There is something called scope that will make this statement more nuanced. We'll talk about it later.
#
# Let's try creating a variable in a few different ways:
#
# Variables can have any name that follows these rules:
# * not a [reserved keyword](https://www.w3schools.com/python/python_ref_keywords.asp) (one of the builtin parts of the language)
# * no spaces
# * starts with a letter (not a number)
#
# Variables are also *case sensitive*.
# For variables that make sense as multiple words, we usually use an underscore `_` instead of a space.
# Here are some examples of valid variable names:
#
# Let's see what happens if we don't respect the case sensitivity:
#
# Oops, we get a `NameError`.
# This is Python trying to tell us that it doesn't understand our command in as much detail as it can.
# We'll learn more about [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html) later.
#
# ### Manipulating variables
#
# We can manipulate variables using math.
# You can imagine Python as a calculator if you're just using some of the basic functionality:
#
# Multiple variables can be involved in these math operations:
#
# Once a variable is defined, its value persists between code blocks.
# For example, do remember what value we assigned to `a_variable_with_a_long_name` from before?
# The Python interpreter does!
#
# When manipulating a variable like this, it can get tiring to write expressions like `my_variable = my_variable + 1`.
# We can use a shorthand `+=` or `-=` to *increment* or *decrement* the value like so:

# %%
x = 1  # this changes the value from before!
my_variable = "abc"

a_variable_with_a_long_name = 1
MyMixedCaseVariable = 2
c = 3
my_var_4 = 4

print(a_variable_with_a_long_name)
print(MyMixedCaseVariable)
print(c)
print(my_var_4)

print(mymixedcasevariable)

x = 1 * 2  # x will be 2
x = x + 3  # x will be 2 + 3 = 5
print(x)   # print the value to see

x = 1
y = 2
z = 3
print(x * y + z)

print(a_variable_with_a_long_name)
print(a_variable_with_a_long_name * 2)
print(a_variable_with_a_long_name + 5)

x = 1
print(x)
x += 1
print(x)
x += x  # we can also use variable values
print(x)
x *= 2  # we can use +, -, *, and / this way
print(x)
