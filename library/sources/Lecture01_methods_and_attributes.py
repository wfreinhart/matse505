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
# id: Lecture01_methods_and_attributes
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Methods and attributes
#
# Let's build our vocabulary up a bit more.
# At a basic level, objects can have two types of "things" -- methods and attributes.
#
# * **methods** are functions that belong to the object and can access variables inside it
# * **attributes** are variables that belong to the object
#
# Using our `list` example, `append` and `sort` are methods. `list` does not have any attributes. `np.array` does, however:
#
# We can access methods and attributes inside an object using the same `.` notation. The only difference is what is returned: a callable in the case of methods and an object in the case of attributes. If we want to use the `mean` method, for instance, we need to add the `()` as we do for any function.

# %%
import numpy as np  # this is a module, with an alias

my_arr = np.array(my_list)
print(my_arr)

print(my_arr.mean)   # this is a method
print(my_arr.shape)  # this is an attribute

print(my_arr.mean())
print(np.mean(my_arr))
