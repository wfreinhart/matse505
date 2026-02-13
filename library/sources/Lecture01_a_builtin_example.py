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
# id: Lecture01_a_builtin_example
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## A builtin example
#
# Consider the `list` data type:
#
# At first glance, we may think of `list` as this series of numbers. But we can ask `list` to do things:
#
# Notice how `append` isn't a Python builtin function, but instead it actually comes after `my_list` plus a `.` (dot)
#
# What does this mean?
# It means that `list` is more than just a series of numbers, it is an **object**. That object is capable of *doing things* and it also *knows information about itself*.
#
# Here's an example: `sort`
#
# Weird, it looks like nothing happened...let's check on `my_list` again:
#
# The `list` is sorted now! What happened?
#
# Well, we asked `my_list` to sort its entries using `sort()`.
# In fact, `sort` is something that all `list` objects know how to do to their own data. Let's see what else `my_list` can do using `dir`:
#
# Woah, `my_list` can do a lot of things!
#
# What's up with all those `__` entries? Those are `builtins`, and they are named that way deliberately to "hide" them from you as a user. Notice that one of them is called `__len__`, which is suspiciously similar to the `len()` builtin function.
# It turns out that the builtin `len()` simply calls `list.__len__()`, and we can demonstrate that here:

# %%
my_list = [5, 4, 3, 2, 1]
print(my_list)

my_list.append(6)
print(my_list)

my_list.sort()

print(my_list)

dir(my_list)

my_list.__len__()

len(my_list)
