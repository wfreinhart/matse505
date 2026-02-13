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
# id: Lecture01_arguments
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Arguments
#
# We have already discussed the difference between *positional* and *keyword* arguments when using existing functions. Now we will see how it works when we define our own.
#
# ### Positional arguments
#
# Here we define a function with two positional arguments. Which input value goes to `a` and which goes to `b` is therefore determined by the *order* of the inputs:
#
# There is no limit to the number of positional arguments we provide. However, from a practical perspective it gets annoying to provide more than 3 or 4:
#
# ### Returning multiple values
#
# We can also `return` multiple values. We've used functions with this behavior before, like `subplots()` and `pearsonr`. All we need to do is list multiple values separated by commas:
#
# This works exactly like positional arguments where the order determines which variable gets which value. Also like positional arguments, we should practically limit the number of values being returned so as not to annoy the user (likely ourselves!):
#
# We can also use conditional statements to control which variables get returned.
#
# > Note: this can make the interface to your function rather confusing or inconsistent. Think carefully about this design choice.
#
# ### Keyword arguments
#
# In addition to positional arguments, we can use *keywords* to assign specific values to specific variables inside our function. This allows us to:
# * mix the order of inputs
# * give default values so not every parameter needs to be specified
#
# Let's look at an example:
#
# Notice how the value `b = 2` was not provided in the function call, it happens automatically in the function itself. We could also provide a different value and it will use that. Either position or the keyword itself can set the value of `b`.
#
# We can do the same thing with several keyword arguments:
#
# We also don't have to specify both values, we can do any combination of them:
#
# Finally, we can also refer to the positional arguments by their names if we want to reference them out of order:
#
# ### Sidenote: unpacking arguments
#
# Python also supports a pretty crazy syntax for "argument unpacking." Let's say we put a bunch of arguments in a `list`. Then we could do a function call using each element of this `list` in this way:
#
# What if we wanted to tell `my_function` that the `list` is *the list of arguments*? So instead of `args[0], args[1], args[2]` we could just say "the positional arguments are in order in this `list`. We can do exactly that with the `*` operator:

# %%
def my_function(a, b):
    return (a + 1) * b

print(my_function(1, 2))  # a = 1, b = 2
print(my_function(2, 1))  # a = 2, b = 1

def my_function(a, b, c, d, e, f, g, h, i, j):
    # this should just take a list!!!
    return a + b + c + d + e + f + g + h + i + j

print(my_function(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

def my_function(a, b):
    return a + b, a * b

out = my_function(2, 3)
print(out)

plus, mult = out
print(plus)
print(mult)

def my_function(a):
    return a + 1, a + 2, a + 3, a + 4, a + 5, a + 6

a1, a2, a3, a4, a5, a6 = my_function(1)
print(a1, a2, a3, a4, a5, a6)

def my_function(a):
    return 0
    # these don't run after "return"
    a += 1
    return a

my_function(10)

def my_function(a, return_a):
    b = a + 1
    if return_a:
        return a, b
    else:
        return b

print(my_function(1, False))
print(my_function(1, True))

def my_function(a, b=2):
    return a * b

print(my_function(3))
print(my_function(3, 3))
print(my_function(3, b=4))

def my_function(a, b=2, c=3):
    return a * b + c

print(my_function(2))            # use default values
print(my_function(2, 3, 2))      # in order as positional args
print(my_function(2, c=3, b=2))  # out of order as keyword args!

print(my_function(2, c=2))
print(my_function(2, b=2))

# the function def from above:
# def my_function(a, b=2, c=3):
print(my_function(b=2, c=3, a=2))  # refer to a using `a=` keyword!

def my_function(a, b, c):
    return a + b + c

args = [1, 2, 3]
my_function(args[0], args[1], args[2])

my_function(*args)
